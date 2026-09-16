# > A slow and inefficient implementation of a slightly modified Trompt model
# > From the ICLM 2023 paper https://arxiv.org/abs/2305.18446


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import os
import urllib.request
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import time


class TromptCell(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model):
        super().__init__()
        # Embeddings (Figure 3.2)
        self.feature_emb_weight = nn.Parameter(torch.empty(n_columns, d_model))
        self.feature_emb_bias = nn.Parameter(torch.empty(n_columns, d_model))
        self.ln_emb = nn.LayerNorm(d_model)

        # Importance Getter (Figure 3.1)
        self.ln_col = nn.LayerNorm(d_model)
        self.ln_prompt = nn.LayerNorm(d_model)
        self.dense_imp = nn.Linear(2 * d_model, d_model)

        self.emb_column = nn.Parameter(torch.empty(n_columns, d_model))
        self.emb_prompt = nn.Parameter(torch.empty(n_prompts, d_model))

        # Modified expansion block (Figure 3.3)
        # Without non-linearities! This is important to make significant speed-ups possible.
        self.dense_expand = nn.Linear(1, n_prompts)

        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.feature_emb_weight.shape[1] ** -0.5
        nn.init.uniform_(self.feature_emb_weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.feature_emb_bias, -d_rsqrt, d_rsqrt)
        nn.init.normal_(self.emb_column, std=0.01)
        nn.init.normal_(self.emb_prompt, std=0.01)

    def forward(self, x: torch.Tensor, prev_cell_out: torch.Tensor) -> torch.Tensor:
        x_emb = x[..., None] * self.feature_emb_weight + self.feature_emb_bias
        x_emb = F.relu(x_emb)
        x_emb = self.ln_emb(x_emb)

        x_prompt = self.emb_prompt
        x_prompt = self.dense_imp(torch.cat([self.ln_prompt(x_prompt), prev_cell_out], dim=-1)) + x_prompt
        x_column = self.ln_col(self.emb_column)
        scores = x_prompt @ x_column.T
        mask = torch.softmax(scores, dim=-1)

        a = self.dense_expand.weight[:, 0].view(1, -1, 1)
        q = self.dense_expand.bias.view(1, -1, 1)

        x_out = (1.0 + a) * (mask @ x_emb) + q
        return x_out


class TromptDownstream(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense0 = nn.Linear(d_model, 1)
        self.dense1 = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dense_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pw = torch.softmax(self.dense0(x).squeeze(-1), dim=-1)
        xnew = (pw.unsqueeze(-1) * x).sum(dim=-2)
        return self.dense_out(self.ln(F.relu(self.dense1(xnew))))


class Trompt(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model, n_cycles):
        super().__init__()
        self.tcells = nn.ModuleList([TromptCell(n_columns, n_prompts, d_model) for _ in range(n_cycles)])
        self.tdown = TromptDownstream(d_model)
        self.prompt = nn.Parameter(torch.empty(n_prompts, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.prompt, std=0.01)

    def forward(self, x):
        x_prompt = self.prompt
        outputs = []
        for cell in self.tcells:
            outputs.append(self.tdown(cell(x, x_prompt)))
        return torch.stack(outputs, dim=1).squeeze(-1)


def load_from_url(url, cache_dir='.'):
    filename = os.path.join(cache_dir, url.split('/')[-1])
    if not os.path.exists(filename):
        with tqdm(unit='B', unit_scale=True, desc=filename) as pbar:
            urllib.request.urlretrieve(url, filename, reporthook=lambda _, b, t: pbar.update(b))
    return torch.load(filename, map_location=torch.device('cpu'), weights_only=True)


TRAIN_DATA = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/train_dataset.pt"
VAL_DATA = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/val_dataset.pt"

AMP = True
EPOCHS = 5
N_CYCLES = 6
BATCH_SIZE = 128  

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()

    torch.manual_seed(0)

    if rank == 0:
        train_data = load_from_url(TRAIN_DATA)
        val_data = load_from_url(VAL_DATA)

    dist.barrier(device_ids=[local_rank])

    if rank != 0:
        train_data = load_from_url(TRAIN_DATA)

    train_dataset = torch.utils.data.TensorDataset(*map(torch.nan_to_num, train_data))

    val_dataset = None
    val_dl = None
    if rank == 0:
        val_dataset = torch.utils.data.TensorDataset(*map(torch.nan_to_num, val_data))
        val_dl = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=1024)

    Y_mean = train_dataset.tensors[1].mean()
    Y_std = train_dataset.tensors[1].std()
    train_dataset.tensors = (train_dataset.tensors[0],(train_dataset.tensors[1] - Y_mean) / Y_std,)

    model = Trompt(n_columns=train_dataset.tensors[0].shape[1], n_prompts=128, d_model=128, n_cycles=N_CYCLES,).to(device)
    model.compile(mode="reduce-overhead")
    model = DDP(model, device_ids=[local_rank])

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dl = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=BATCH_SIZE, sampler=train_sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    x_warm, y_warm = next(iter(train_dl))
    x_warm, y_warm = x_warm.to(device), y_warm.to(device)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            pred = model(x_warm)
            loss = F.mse_loss(pred, y_warm.unsqueeze(1).expand(-1, N_CYCLES))
        scaler.scale(loss).backward()
    optimizer.zero_grad(set_to_none=True)

    throughputs = []
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        samples_processed = 0

        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[local_rank])
        start_time = time.perf_counter()

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                pred = model(x)
                loss = F.mse_loss(pred, y.unsqueeze(1).expand(-1, N_CYCLES))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            samples_processed += x.shape[0]

        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[local_rank])
        train_time = time.perf_counter() - start_time

        total_samples = torch.tensor(samples_processed, device=device, dtype=torch.long)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        throughput = total_samples.item() / train_time

        if rank == 0:
            throughputs.append(throughput)
            print(
                f"Epoch {epoch + 1}/{EPOCHS}; "
                f"loss = {loss.item():.5f}; "
                f"time = {train_time:.2f}s; "
                f"throughput = {throughput:.2f} samples/sec",
                flush=True,
            )

    if rank == 0 and val_dataset is not None and val_dl is not None:
        print(f"Peak throughput: {max(throughputs):.2f} samples/sec")
        print(f"Average throughput: {sum(throughputs) / len(throughputs):.2f} samples/sec")

        model.module.eval()
        mae = 0.0
        Y_mean, Y_std = Y_mean.to(device), Y_std.to(device)
        with torch.inference_mode():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                    pred = model.module(x)
                mae += (pred.mean(dim=-1) * Y_std + Y_mean - y).abs().sum().item()
        print(f"Validation MAE = {mae / len(val_dataset):.5f}")

    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()

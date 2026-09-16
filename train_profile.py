# > A slow and inefficient implementation of a slightly modified Trompt model
# > From the ICLM 2023 paper https://arxiv.org/abs/2305.18446

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import os
import urllib.request
from tqdm import tqdm

from torch.profiler import profile, ProfilerActivity, record_function
from torch.profiler import schedule

import time

PROFILE = True
TRAIN = False
VALIDATE = False
COMPILE = False
AMP = False


EPOCHS = 5

N_CYCLES = 6
BATCH_SIZE = 64

PROFILE_SKIP = 5
PROFILE_WARMUP = 1
PROFILE_ACTIVE = 10


PROFILE_STEPS = (PROFILE_SKIP + PROFILE_WARMUP + PROFILE_ACTIVE)


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
        with record_function("cell_feature_embedding"):
            x_emb = x.unsqueeze(-1) * self.feature_emb_weight + self.feature_emb_bias.unsqueeze(0) # (8, 984, 128)
            x_emb = F.relu(x_emb) # (8, 984, 128)
            x_emb = self.ln_emb(x_emb) # (8, 984, 128)

        with record_function("cell_importance_getter"):
            with record_function("prompt_normalization"):
                # x_prompt = self.emb_prompt.unsqueeze(0).repeat(x_emb.shape[0], 1, 1) # (8, 128, 128)
                x_prompt = self.emb_prompt

            with record_function("importance_dense"):
                x_prompt = self.dense_imp(torch.cat([self.ln_prompt(x_prompt), prev_cell_out], dim=-1)) + x_prompt # (8, 128, 128)

            with record_function("column_normalization"):
                # x_column = self.ln_col(self.emb_column.unsqueeze(0).repeat(x_emb.shape[0], 1, 1)) # (8, 984, 128)
                x_column = self.ln_col(self.emb_column)

            with record_function("attention_scores"):
                scores = x_prompt @ x_column.T

            with record_function("attention_softmax"):
                # mask = torch.softmax(x_prompt @ x_column.transpose(1,2), dim=-1) # (8, 128, 984) 
                # mask = torch.softmax(x_prompt @ x_column.T, dim=-1,)
                mask = torch.softmax(scores, dim=-1)

        # with record_function("cell_expansion"):
            # x_emb = x_emb.unsqueeze(1) + self.dense_expand(x_emb.unsqueeze(-1)).permute(0, 3, 1, 2)

        with record_function("cell_weighted_sum"):
            # x_out = (mask.unsqueeze(-1) * x_emb).sum(dim=2)
            a = self.dense_expand.weight[:, 0].view(1, -1, 1)
            q = self.dense_expand.bias.view(1, -1, 1)

            x_out = (1.0 + a) * (mask @ x_emb) + q # (8, 128, 984) @ (8, 984, 128)

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
        # x_prompt = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1)
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

if __name__ == "__main__":
    device = torch.device("cuda:0")

    torch.manual_seed(0)

    train_dataset = torch.utils.data.TensorDataset(*map(torch.nan_to_num, load_from_url(TRAIN_DATA)))

    val_dataset = torch.utils.data.TensorDataset(*map(torch.nan_to_num, load_from_url(VAL_DATA)))

    Y_mean = train_dataset.tensors[1].mean()
    Y_std = train_dataset.tensors[1].std()
    train_dataset.tensors = (train_dataset.tensors[0],(train_dataset.tensors[1] - Y_mean) / Y_std)

    model = Trompt(n_columns=train_dataset.tensors[0].shape[1],n_prompts=128, d_model=128, n_cycles=N_CYCLES,).to(device)

    if COMPILE:
        model.compile(mode="reduce-overhead")

    train_dl = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=1024)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities += [ProfilerActivity.CUDA]

    sort_by_keyword = "cuda_time_total"

    def trace_handler(p):
        time_table = p.key_averages(group_by_input_shape=True, group_by_stack_n=5,
                                    ).table(
                                        sort_by=sort_by_keyword,
                                        row_limit=20,
                                        max_name_column_width=60,
                                        max_shapes_column_width=100,
                                    )
        memory_table = p.key_averages(group_by_input_shape=True,
                                    ).table(
                                        sort_by="self_cuda_memory_usage",
                                        row_limit=20,
                                        max_name_column_width=60,
                                        max_shapes_column_width=100,
                                    )
        
        print("top operations by cuda time")
        print(time_table)
        print()
        print("top operations by cuda memory")
        print(memory_table)

        with open("profile_time.txt", "w") as f:
            f.write(time_table)

        with open("profile_memory.txt", "w") as f:
            f.write(memory_table)

        p.export_chrome_trace("profile_trace.json")

    my_schedule = schedule(skip_first=PROFILE_SKIP, wait=0, warmup=PROFILE_WARMUP, active=PROFILE_ACTIVE,repeat=1,)

    if PROFILE:
        model.train()

        profiler_context = profile(
            activities=activities,
            schedule=my_schedule,
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        with profiler_context as p:
            for i, batch in enumerate(tqdm(train_dl)):
                x, y = batch

                with record_function("to_device"):
                    x = x.to(device)
                    y = y.to(device)

                with record_function("zero_grad"):
                    optimizer.zero_grad()

                with record_function("forward"):
                    pred = model(x)

                with record_function("loss"):
                    loss = F.mse_loss(pred,y.unsqueeze(1).expand(-1, N_CYCLES))

                with record_function("backward"):
                    loss.backward()

                with record_function("optimizer_step"):
                    optimizer.step()

                p.step()

                if i + 1 >= PROFILE_STEPS:
                    break

    if TRAIN:
        model.train()

        if COMPILE:
            x_warm, y_warm = next(iter(train_dl))

            x_warm = x_warm.to(device)
            y_warm = y_warm.to(device)

            for _ in range(EPOCHS):
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                    pred = model(x_warm)
                    loss = F.mse_loss(pred, y_warm.unsqueeze(1).expand(-1, N_CYCLES))
                scaler.scale(loss).backward()
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize(device)

        throughputs = []

        for epoch in range(EPOCHS):
            samples_processed = 0

            torch.cuda.synchronize(device)

            start_time = time.perf_counter()

            for x, y in train_dl:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                    pred = model(x)
                    loss = F.mse_loss(pred, y.unsqueeze(1).expand(-1, N_CYCLES))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                samples_processed += x.shape[0]

            torch.cuda.synchronize(device)

            train_time = time.perf_counter() - start_time
            samples_per_sec = samples_processed / train_time
            throughputs.append(samples_per_sec)

            print(
                f"Epoch {epoch + 1}/{EPOCHS}; "
                f"loss = {loss.item():.5f}; "
                f"time = {train_time:.2f}s; "
                f"samples = {samples_processed}; "
                f"throughput = {samples_per_sec:.2f} samples/sec"
            )

        print(f"Peak throughput: {max(throughputs):.2f} samples/sec")
        print(f"Average throughput: {sum(throughputs) / len(throughputs):.2f} samples/sec")


    if VALIDATE:
        model.eval()
        mae = 0

        with torch.inference_mode():
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                    pred = model(x)

                mae += (pred.mean(dim=-1) * Y_std + Y_mean - y).abs().sum().item()

        mae /= len(val_dataset)
        print(f"Validation MAE = {mae:.5f}")

import os
import time
import urllib.request

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import Trompt


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
BATCH_SIZE = 64  

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()

    torch.manual_seed(0)

    if dist.get_rank() == 0:
        load_from_url(TRAIN_DATA)
        load_from_url(VAL_DATA)
    dist.barrier(device_ids=[local_rank])

    train_dataset = torch.utils.data.TensorDataset(*map(torch.nan_to_num, load_from_url(TRAIN_DATA)))

    if dist.get_rank() == 0:
        val_dataset = torch.utils.data.TensorDataset(*map(torch.nan_to_num, load_from_url(VAL_DATA)))
        val_dl = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=1024)

    Y_mean = train_dataset.tensors[1].mean()
    Y_std = train_dataset.tensors[1].std()
    train_dataset.tensors = (train_dataset.tensors[0],(train_dataset.tensors[1] - Y_mean) / Y_std,)

    model = Trompt(n_columns=train_dataset.tensors[0].shape[1], n_prompts=128, d_model=128, n_cycles=N_CYCLES,).to(device)
    model.compile(mode="reduce-overhead")
    model = DDP(model, device_ids=[local_rank])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True)
    train_dl = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=BATCH_SIZE, sampler=train_sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    x_warm, y_warm = next(iter(train_dl))
    x_warm, y_warm = x_warm.to(device), y_warm.to(device)
    for _ in range(EPOCHS):
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

        if dist.get_rank() == 0:
            throughputs.append(throughput)
            print(
                f"Epoch {epoch + 1}/{EPOCHS}; "
                f"loss = {loss.item():.5f}; "
                f"time = {train_time:.2f}s; "
                f"samples = {total_samples.item()}; "
                f"throughput = {throughput:.2f} samples/sec",
                flush=True,
            )

    if dist.get_rank() == 0:
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

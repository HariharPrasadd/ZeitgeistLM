"""Measure end-to-end temporal GPT training throughput on one Modal GPU."""

import json
import os
from pathlib import Path
import statistics
import time

import torch

import train_gpt2
from train_gpt2 import GPT, GPTConfig
from train_zeitgeist import TimestampedLoader


def select_shard(root: Path) -> tuple[dict, dict]:
    """Choose the largest completed training shard already on the Volume."""
    candidates = []
    for path in root.glob("20[1-2][0-9]/[0-1][0-9]/*/audit.json"):
        report = json.loads(path.read_text())
        if report.get("year", 9999) > 2020 or report.get("tokenizer_version") != 2:
            continue
        for spec in report["files"]:
            candidates.append((int(spec["windows"]), path.parent, spec))
    if not candidates:
        raise FileNotFoundError("No completed 2011-2020 token shard to benchmark")
    windows, directory, spec = max(candidates, key=lambda item: item[0])
    manifest = {
        "train_start_utc": 1293840000, "train_end_utc": 1609459199,
        "splits": {"train": [{"tokens": str((directory / spec["tokens"]).relative_to(root)),
                              "created_utc": str((directory / spec["created_utc"]).relative_to(root))}]},
    }
    return manifest, {"path": str(directory.relative_to(root)), "windows": windows}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark needs a Modal GPU")
    root = Path(os.environ.get("ZEITGEIST_DATA_ROOT", "/data/tokenized"))
    manifest, chosen = select_shard(root)
    batch_size = 16
    sequence_length = 1024
    accumulation = 16
    warmup_steps = 2
    timed_steps = 5
    train_gpt2.master_process = False
    torch.manual_seed(1337)
    loader = TimestampedLoader(root, manifest, "train", batch_size,
                               sequence_length, 1337)
    model = GPT(GPTConfig(block_size=sequence_length, vocab_size=50304)).cuda()
    optimizer = model.configure_optimizers(0.1, 6e-4, "cuda")
    times = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(warmup_steps + timed_steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            x, y, tau = loader.next_batch()
            x, y, tau = x.cuda(non_blocking=True), y.cuda(non_blocking=True), tau.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, tau)
            (loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        if step >= warmup_steps:
            times.append(time.perf_counter() - started)
    tokens_per_step = batch_size * sequence_length * accumulation
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shard": chosen,
        "micro_batch": batch_size, "sequence_length": sequence_length,
        "accumulation": accumulation, "tokens_per_step": tokens_per_step,
        "warmup_steps": warmup_steps, "timed_steps": timed_steps,
        "step_seconds": times,
        "median_tokens_per_second": tokens_per_step / statistics.median(times),
        "mean_tokens_per_second": tokens_per_step / statistics.mean(times),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()

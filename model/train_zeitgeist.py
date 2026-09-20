"""Train the temporal GPT from timestamped, fixed-length document windows."""

import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import train_gpt2
from train_gpt2 import GPT, GPTConfig


class TimestampedLoader:
    """Sample prepacked windows with a numeric mean source timestamp."""

    def __init__(self, root: Path, manifest: dict, split: str, batch_size: int,
                 sequence_length: int, seed: int):
        self.root = root
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.start = int(manifest["train_start_utc"])
        self.end = int(manifest["train_end_utc"])
        if self.end <= self.start:
            raise ValueError("Training time interval must be positive")
        self.rng = np.random.default_rng(seed)
        self.shards = []
        counts = []
        for spec in manifest["splits"][split]:
            # Each row is one T+1-token window packed from a single month.
            tokens = np.load(root / spec["tokens"], mmap_mode="r")
            dates = np.load(root / spec["created_utc"], mmap_mode="r")
            if tokens.ndim != 2 or tokens.shape[1] != sequence_length + 1:
                raise ValueError(f"Bad token shape in {spec['tokens']}")
            if dates.shape != (tokens.shape[0],):
                raise ValueError(f"Timestamp count differs in {spec['created_utc']}")
            if tokens.shape[0]:
                self.shards.append((tokens, dates))
                counts.append(tokens.shape[0])
        if not counts:
            raise ValueError(f"No {split} windows in manifest")
        self.cumulative = np.cumsum(counts)

    def next_batch(self):
        # Uniform sampling over all windows prevents small shards dominating.
        picks = self.rng.integers(0, int(self.cumulative[-1]), size=self.batch_size)
        shard_ids = np.searchsorted(self.cumulative, picks, side="right")
        x = np.empty((self.batch_size, self.sequence_length), dtype=np.int64)
        y = np.empty_like(x)
        dates = np.empty(self.batch_size, dtype=np.float32)
        for n, (pick, shard_id) in enumerate(zip(picks, shard_ids)):
            base = 0 if shard_id == 0 else self.cumulative[shard_id - 1]
            tokens, stamps = self.shards[shard_id]
            row = int(pick - base)
            x[n] = tokens[row, :-1]
            y[n] = tokens[row, 1:]
            dates[n] = (int(stamps[row]) - self.start) / (self.end - self.start)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(dates)


def main():
    root = Path(os.environ.get("ZEITGEIST_DATA_ROOT", "/data/tokenized"))
    output = Path(os.environ.get("ZEITGEIST_OUTPUT_ROOT", "/data/checkpoints"))
    manifest = json.loads((root / "manifest.json").read_text())
    B = int(os.environ.get("ZEITGEIST_MICRO_BATCH", "16"))
    T = int(os.environ.get("ZEITGEIST_SEQUENCE_LENGTH", "1024"))
    total_batch = int(os.environ.get("ZEITGEIST_TOTAL_BATCH_TOKENS", "262144"))
    if total_batch % (B * T):
        raise ValueError("Total batch tokens must be divisible by micro batch tokens")
    accumulation = total_batch // (B * T)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    train_gpt2.master_process = True
    torch.manual_seed(1337)
    train = TimestampedLoader(root, manifest, "train", B, T, 1337)
    val = TimestampedLoader(root, manifest, "val", B, T, 7331)
    config = GPTConfig(
        block_size=T,
        vocab_size=int(os.environ.get("ZEITGEIST_VOCAB_SIZE", "50304")),
        n_layer=int(os.environ.get("ZEITGEIST_N_LAYER", "12")),
        n_head=int(os.environ.get("ZEITGEIST_N_HEAD", "12")),
        n_embd=int(os.environ.get("ZEITGEIST_N_EMBD", "768")),
    )
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(0.1, 6e-4, device_type)
    max_steps = int(os.environ.get("ZEITGEIST_MAX_STEPS", "0"))
    if max_steps <= 0:
        train_tokens = int(manifest["train_tokens"])
        max_steps = max(1, math.ceil(train_tokens / total_batch))
    warmup = min(715, max(1, max_steps // 20))
    lr_scale = float(os.environ.get("ZEITGEIST_LR_SCALE", "1"))
    if not 0 < lr_scale <= 1:
        raise ValueError("ZEITGEIST_LR_SCALE must be in (0, 1]")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "latest.pt"
    # Milestones are thresholds; the saved step may pass one by a partial batch.
    milestone_text = os.environ.get("ZEITGEIST_MILESTONES", "100000000,250000000,500000000,1000000000")
    milestones = sorted({int(value.strip()) for value in milestone_text.split(",") if value.strip()})
    if any(value <= 0 for value in milestones):
        raise ValueError("Checkpoint milestones must be positive token counts")
    start_step = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train.rng.bit_generator.state = checkpoint["train_rng"]
        val.rng.bit_generator.state = checkpoint["val_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
        start_step = int(checkpoint["next_step"])
        print(f"Resuming from step {start_step}", flush=True)

    def learning_rate(step):
        # Preserve the schedule shape when lowering LR after a resumed run.
        if step < warmup:
            return lr_scale * 6e-4 * (step + 1) / warmup
        fraction = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        return lr_scale * (6e-5 + 0.5 * (6e-4 - 6e-5) *
                           (1 + math.cos(math.pi * fraction)))

    def save(next_step, crossed_milestones):
        # Atomic replacement makes a stopped GPU job restart from the last full step.
        temporary = output / "latest.pt.tmp"
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "next_step": next_step, "train_rng": train.rng.bit_generator.state,
            "val_rng": val.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
            "config": model.config, "time_bounds": (train.start, train.end),
            "tokens_seen": next_step * total_batch,
        }, temporary)
        temporary.replace(checkpoint_path)
        for milestone in crossed_milestones:
            # Keep an immutable copy for comparing temporal structure over training.
            label = (f"{milestone // 1000000}M" if milestone % 1000000 == 0
                     else f"{milestone}tokens")
            target = output / f"tokens_{label}.pt"
            if not target.exists():
                temporary_target = output / f"{target.name}.tmp"
                shutil.copy2(checkpoint_path, temporary_target)
                temporary_target.replace(target)
                print(f"Saved {target.name} at {next_step * total_batch:,} tokens", flush=True)
        if os.environ.get("ZEITGEIST_ON_MODAL"):
            import modal
            modal.Volume.from_name("zeitgeistlm-data").commit()

    if start_step:
        # Repair a snapshot if interruption occurred after latest.pt was replaced.
        pending = [value for value in milestones
                   if (start_step - 1) * total_batch < value <= start_step * total_batch]
        if pending:
            save(start_step, pending)

    torch.set_float32_matmul_precision("high")
    for step in range(start_step, max_steps):
        started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(accumulation):
            x, y, tau = train.next_batch()
            x, y, tau = x.to(device), y.to(device), tau.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                enabled=device_type == "cuda"):
                _, loss = model(x, y, tau)
            (loss / accumulation).backward()
            loss_total += loss.detach().item() / accumulation
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        print(f"step {step} loss {loss_total:.4f} lr {lr:.3g} norm {norm:.3f} "
              f"seconds {time.time() - started:.1f}", flush=True)
        if (step + 1) % 250 == 0 or step + 1 == max_steps:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(20):
                    x, y, tau = val.next_batch()
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                        enabled=device_type == "cuda"):
                        _, loss = model(x.to(device), y.to(device), tau.to(device))
                    losses.append(loss.item())
            print(f"validation step {step + 1} loss {sum(losses) / len(losses):.4f}", flush=True)
        tokens_before = step * total_batch
        tokens_after = (step + 1) * total_batch
        crossed = [value for value in milestones if tokens_before < value <= tokens_after]
        # Frequent early recovery points, then longer intervals as training settles.
        save_interval = 50 if tokens_after < 100000000 else 100 if tokens_after < 250000000 else 250
        if crossed or (step + 1) % save_interval == 0 or step + 1 == max_steps:
            save(step + 1, crossed)


if __name__ == "__main__":
    main()

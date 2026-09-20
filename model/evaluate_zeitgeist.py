"""Measure chronological validation or explicitly requested future-test loss."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from zeitgeist_inference import load_model, normalized_time


def evaluate(model, manifest: dict, root: Path, split: str, batch_size: int,
             max_windows: int, device: str) -> dict:
    """Score windows once each, preserving their stored time coordinates."""
    totals = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0, "windows": 0})
    processed = 0
    with torch.inference_mode():
        for spec in manifest["splits"][split]:
            token_rows = np.load(root / spec["tokens"], mmap_mode="r")
            dates = np.load(root / spec["created_utc"], mmap_mode="r")
            for offset in range(0, len(dates), batch_size):
                if max_windows and processed >= max_windows:
                    break
                stop = min(offset + batch_size, len(dates))
                if max_windows:
                    stop = min(stop, offset + max_windows - processed)
                batch = np.asarray(token_rows[offset:stop], dtype=np.int64)
                x = torch.from_numpy(batch[:, :-1]).to(device)
                y = torch.from_numpy(batch[:, 1:]).to(device)
                tau = torch.tensor([normalized_time(int(date), manifest)
                                    for date in dates[offset:stop]], device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=device == "cuda"):
                    logits, _ = model(x, time_value=tau)
                    loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                               y.reshape(-1), reduction="sum").item()
                count = y.numel()
                entry = totals[spec["kind"]]
                entry["loss_sum"] += loss_sum
                entry["tokens"] += count
                entry["windows"] += stop - offset
                processed += stop - offset
            if max_windows and processed >= max_windows:
                break
    total_tokens = sum(item["tokens"] for item in totals.values())
    total_loss = sum(item["loss_sum"] for item in totals.values())
    if not total_tokens:
        raise ValueError(f"No windows in {split} split")
    return {"split": split, "loss": total_loss / total_tokens,
            "tokens": total_tokens, "windows": processed,
            "by_kind": {kind: {"loss": value["loss_sum"] / value["tokens"],
                               "tokens": value["tokens"], "windows": value["windows"]}
                        for kind, value in totals.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Future test is locked; pass --allow-test after model selection")
    if args.batch_size <= 0 or args.max_windows < 0:
        parser.error("Batch size must be positive and max-windows nonnegative")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, manifest, checkpoint = load_model(args.checkpoint, args.manifest, device)
    report = evaluate(model, manifest, args.manifest.parent, args.split,
                      args.batch_size, args.max_windows, device)
    report["checkpoint_tokens_seen"] = checkpoint.get("tokens_seen")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

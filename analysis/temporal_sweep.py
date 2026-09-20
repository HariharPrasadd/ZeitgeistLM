"""Score unchanged held-out token windows over a continuous time grid."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiment import sample_windows
from zeitgeist_inference import load_model, normalized_time, parse_date


def run(root: Path, checkpoint_name: str = "tokens_1000M.pt") -> dict:
    """Return per-window likelihood curves with tokens and checkpoint fixed."""
    model, manifest, checkpoint = load_model(root / "checkpoints" / checkpoint_name,
                                              root / "tokenized/manifest.json", "cuda")
    windows = [(split, *row) for split in ("val", "test")
               for row in sample_windows(root / "tokenized", manifest, split)]
    dates = [f"{year}-07-01" for year in range(2011, 2024)]
    grid = [normalized_time(parse_date(date), manifest) for date in dates]
    losses = np.empty((len(windows), len(grid)), dtype=np.float32)
    for offset in range(0, len(windows), 8):
        chunk = windows[offset:offset + 8]
        tokens = torch.as_tensor(np.stack([row[1] for row in chunk]), device="cuda")
        for j, tau in enumerate(grid):
            times = torch.full((len(chunk),), tau, device="cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(tokens[:, :-1], time_value=times)
                per_token = F.cross_entropy(logits.float().transpose(1, 2),
                                            tokens[:, 1:], reduction="none")
            losses[offset:offset + len(chunk), j] = per_token.mean(1).cpu().numpy()
    records = [{"split": row[0], "year": row[3], "month": row[4],
                "kind": row[5], "created_utc": row[2],
                "loss_by_date": losses[i].tolist()}
               for i, row in enumerate(windows)]
    report = {"checkpoint_name": checkpoint_name,
              "checkpoint_tokens_seen": checkpoint["tokens_seen"],
              "grid_dates": dates, "windows": len(records),
              "records": records,
              "mean_by_split": {split: losses[[r["split"] == split for r in records]].mean(0).tolist()
                                for split in ("val", "test")}}
    return report

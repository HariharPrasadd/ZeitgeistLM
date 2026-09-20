"""Shared checkpoint loading and continuous-time conversion for demos."""

from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from train_gpt2 import GPT


def load_model(checkpoint_path: Path, manifest_path: Path, device: str):
    """Load trained weights and verify the saved training-time interval."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    manifest = json.loads(Path(manifest_path).read_text())
    bounds = (int(manifest["train_start_utc"]), int(manifest["train_end_utc"]))
    if tuple(checkpoint["time_bounds"]) != bounds:
        raise ValueError("Checkpoint and manifest use different time bounds")
    model = GPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, manifest, checkpoint


def normalized_time(created_utc: int, manifest: dict) -> float:
    """Extrapolate naturally beyond the training interval without clipping."""
    start = int(manifest["train_start_utc"])
    end = int(manifest["train_end_utc"])
    return (created_utc - start) / (end - start)


def parse_date(value: str) -> int:
    """Accept a UTC year or ISO calendar date for fixed demo coordinates."""
    if len(value) == 4 and value.isdigit():
        value += "-07-01"
    date = datetime.fromisoformat(value)
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return int(date.timestamp())

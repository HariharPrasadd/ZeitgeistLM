"""Build a verified chronological manifest from completed Modal Volume shards."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


TOKENIZER_VERSION = 2
SEQUENCE_LENGTH = 1024
TRAIN_END_YEAR = 2020
VALIDATION_YEAR = 2021
TEST_YEAR = 2022
TEST_LAST_MONTH = 8


def expected_partitions():
    """Yield the exact month/type set requested for this experiment."""
    for year in range(2011, TEST_YEAR + 1):
        last_month = TEST_LAST_MONTH if year == TEST_YEAR else 12
        for month in range(1, last_month + 1):
            for kind in ("submissions", "comments"):
                yield year, month, kind


def split_for_year(year: int) -> str:
    """Assign splits only by publication year."""
    if year <= TRAIN_END_YEAR:
        return "train"
    if year == VALIDATION_YEAR:
        return "val"
    if year == TEST_YEAR:
        return "test"
    raise ValueError(f"Year outside corpus: {year}")


def build_manifest(token_root: Path, clean_root: Path, require_complete: bool = True) -> dict:
    """Validate every expected partition and publish no incomplete training set."""
    token_root, clean_root = Path(token_root), Path(clean_root)
    if require_complete and not (clean_root / "COMPLETE.json").is_file():
        raise RuntimeError("Cleaning has not written COMPLETE.json")

    splits = {"train": [], "val": [], "test": []}
    by_year = defaultdict(lambda: {"submissions": 0, "comments": 0})
    by_split = defaultdict(lambda: {"submissions": 0, "comments": 0})
    documents = defaultdict(lambda: {"submissions": 0, "comments": 0})
    missing_source = []
    pending = []
    for year, month, kind in expected_partitions():
        clean_dir = clean_root / f"{year:04d}" / f"{month:02d}"
        source_audit = clean_dir / f"{kind}_audit.json"
        label = f"{year:04d}-{month:02d}/{kind}"
        if not source_audit.is_file():
            pending.append(label + ": clean audit")
            continue
        source_report = json.loads(source_audit.read_text())
        if source_report.get("status") == "missing_source":
            missing_source.append(label)
            continue
        if not (clean_dir / f"{kind}.parquet").is_file():
            pending.append(label + ": clean Parquet")
            continue
        part_dir = token_root / f"{year:04d}" / f"{month:02d}" / kind
        audit_path = part_dir / "audit.json"
        if not audit_path.is_file():
            pending.append(label + ": token audit")
            continue
        audit = json.loads(audit_path.read_text())
        if audit.get("tokenizer_version") != TOKENIZER_VERSION:
            pending.append(label + ": tokenizer version")
            continue
        if audit.get("sequence_length") != SEQUENCE_LENGTH:
            raise ValueError(f"Wrong sequence length in {audit_path}")
        if (audit.get("year"), audit.get("month"), audit.get("kind")) != (year, month, kind):
            raise ValueError(f"Partition identity mismatch in {audit_path}")

        split = split_for_year(year)
        window_count = 0
        for spec in audit["files"]:
            tokens_path = part_dir / spec["tokens"]
            dates_path = part_dir / spec["created_utc"]
            if not tokens_path.is_file() or not dates_path.is_file():
                pending.append(label + ": token files")
                continue
            tokens = np.load(tokens_path, mmap_mode="r")
            dates = np.load(dates_path, mmap_mode="r")
            if tokens.shape != (spec["windows"], SEQUENCE_LENGTH + 1):
                raise ValueError(f"Token shape mismatch in {tokens_path}")
            if dates.shape != (spec["windows"],):
                raise ValueError(f"Date shape mismatch in {dates_path}")
            start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
            end_year, end_month = (year + 1, 1) if month == 12 else (year, month + 1)
            end = int(datetime(end_year, end_month, 1, tzinfo=timezone.utc).timestamp())
            if len(dates) and (int(dates.min()) < start or int(dates.max()) >= end):
                raise ValueError(f"Timestamp outside source month in {dates_path}")
            count = int(spec["windows"])
            window_count += count
            splits[split].append({
                "tokens": str(tokens_path.relative_to(token_root)),
                "created_utc": str(dates_path.relative_to(token_root)),
                "year": year, "month": month, "kind": kind, "windows": count,
            })
        if window_count != audit["training_windows"]:
            raise ValueError(f"Window count mismatch in {audit_path}")
        count = window_count * SEQUENCE_LENGTH
        by_year[str(year)][kind] += count
        by_split[split][kind] += count
        documents[str(year)][kind] += int(audit["documents"])

    if pending and require_complete:
        raise RuntimeError(f"{len(pending)} partitions incomplete; first: {pending[:8]}")
    if require_complete and (not splits["train"] or not splits["val"] or not splits["test"]):
        raise RuntimeError("All three chronological splits need usable windows")

    first = int(datetime(2011, 1, 1, tzinfo=timezone.utc).timestamp())
    last = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()) - 1
    return {
        "schema_version": 1, "tokenizer_version": TOKENIZER_VERSION,
        "sequence_length": SEQUENCE_LENGTH,
        "train_start_utc": first, "train_end_utc": last,
        "train_tokens": sum(by_split["train"].values()),
        "splits": splits,
        "stats": {"tokens_by_year_and_kind": dict(by_year),
                  "tokens_by_split_and_kind": dict(by_split),
                  "documents_by_year_and_kind": dict(documents),
                  "missing_source": missing_source, "pending": pending},
    }


def write_manifest(token_root: Path, clean_root: Path) -> dict:
    """Atomically publish the final manifest after every partition validates."""
    manifest = build_manifest(token_root, clean_root, require_complete=True)
    target = Path(token_root) / "manifest.json"
    temporary = target.with_name("manifest.json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(target)
    return manifest

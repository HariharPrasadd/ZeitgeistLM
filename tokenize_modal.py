"""Tokenize completed Reddit months on Modal without local data downloads."""

from pathlib import Path

import modal


app = modal.App("zeitgeistlm-tokenize")
volume = modal.Volume.from_name("zeitgeistlm-data")
jobs = modal.Dict.from_name("zeitgeistlm-token-jobs", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "numpy>=2,<3", "pyarrow>=20,<24", "tiktoken>=0.9,<1"
).add_local_python_source("data_manifest")
SEQUENCE_LENGTH = 1024
WINDOWS_PER_FILE = 8192
TOKENIZER_VERSION = 2


@app.function()
def version() -> int:
    """Expose the deployed tokenizer contract for a cheap version check."""
    return TOKENIZER_VERSION


@app.function(image=image, volumes={"/data": volume}, cpu=2, memory=4096,
              timeout=43200, max_containers=8, retries=2)
def tokenize_month(year: int, month: int, kind: str) -> dict:
    """Write time-conditioned training windows for one completed partition."""
    from collections import deque
    from itertools import islice
    import json
    import shutil

    import numpy as np
    import pyarrow.parquet as pq
    import tiktoken

    volume.reload()
    source = Path(f"/data/cleaned/{year:04d}/{month:02d}/{kind}.parquet")
    source_audit = source.with_name(f"{kind}_audit.json")
    if not (source.is_file() and source_audit.is_file()):
        raise FileNotFoundError(f"Cleaned partition is incomplete: {source}")
    destination = Path(f"/data/tokenized/{year:04d}/{month:02d}/{kind}")
    audit_path = destination / "audit.json"
    if audit_path.is_file():
        existing = json.loads(audit_path.read_text())
        if existing.get("tokenizer_version") == TOKENIZER_VERSION:
            return existing

    # Incomplete files from an interrupted attempt are safe to replace.
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    encoder = tiktoken.get_encoding("gpt2")
    end_token = encoder.eot_token
    token_buffer = deque()
    date_buffer = deque()
    window_tokens = []
    window_dates = []
    files = []
    documents = 0
    text_tokens = 0
    dropped_tail_tokens = 0

    def flush():
        if not window_tokens:
            return
        part = len(files)
        token_name = f"part-{part:05d}-tokens.npy"
        date_name = f"part-{part:05d}-created_utc.npy"
        np.save(destination / token_name, np.asarray(window_tokens, dtype=np.uint16))
        np.save(destination / date_name, np.asarray(window_dates, dtype=np.int64))
        files.append({"tokens": token_name, "created_utc": date_name,
                      "windows": len(window_tokens)})
        window_tokens.clear()
        window_dates.clear()

    for batch in pq.ParquetFile(source).iter_batches(
        batch_size=4096, columns=["created_utc", "subreddit", "text"]
    ):
        for row in batch.to_pylist():
            documents += 1
            # Reddit IDs remain in the clean Parquet; the model sees a text delimiter.
            prefix = f"<|subreddit_{row['subreddit']}|> "
            tokens = encoder.encode_ordinary(prefix + row["text"])
            text_tokens += len(tokens)
            created = int(row["created_utc"])
            token_buffer.extend(tokens)
            token_buffer.append(end_token)
            date_buffer.extend([created] * (len(tokens) + 1))
            while len(token_buffer) >= SEQUENCE_LENGTH + 1:
                window_tokens.append(list(islice(token_buffer, SEQUENCE_LENGTH + 1)))
                # A packed window may hold several posts; use its mean source time.
                window_dates.append(round(sum(islice(date_buffer, SEQUENCE_LENGTH)) /
                                          SEQUENCE_LENGTH))
                for _ in range(SEQUENCE_LENGTH):
                    token_buffer.popleft()
                    date_buffer.popleft()
                if len(window_tokens) >= WINDOWS_PER_FILE:
                    flush()
    dropped_tail_tokens = len(token_buffer)
    flush()
    report = {
        "year": year, "month": month, "kind": kind,
        "tokenizer_version": TOKENIZER_VERSION, "documents": documents,
        "text_tokens": text_tokens, "training_windows": sum(x["windows"] for x in files),
        "training_tokens": sum(x["windows"] for x in files) * SEQUENCE_LENGTH,
        "dropped_tail_tokens": dropped_tail_tokens, "timestamp_resolution": "mean source time per window",
        "sequence_length": SEQUENCE_LENGTH, "files": files,
    }
    # The audit is the completion marker; a failed worker leaves only partial files.
    audit_path.write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(volumes={"/data": volume}, schedule=modal.Period(minutes=5),
              max_containers=1, timeout=600)
def dispatch_available() -> dict:
    """Queue every newly committed clean partition, tracking in-flight calls."""
    import json

    volume.reload()
    ready = []
    cleaned_root = Path("/data/cleaned")
    for audit in sorted(cleaned_root.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/*_audit.json")):
        kind = audit.name.removesuffix("_audit.json")
        if kind not in {"submissions", "comments"}:
            continue
        if json.loads(audit.read_text()).get("status") == "missing_source":
            continue
        if not audit.with_name(f"{kind}.parquet").is_file():
            continue
        year, month = int(audit.parent.parent.name), int(audit.parent.name)
        token_audit = Path(f"/data/tokenized/{year:04d}/{month:02d}/{kind}/audit.json")
        if token_audit.is_file() and json.loads(token_audit.read_text()).get(
            "tokenizer_version") == TOKENIZER_VERSION:
            continue
        ready.append((year, month, kind))
    queued = []
    running = 0
    failed = []
    for year, month, kind in ready:
        key = f"v{TOKENIZER_VERSION}/{year:04d}/{month:02d}/{kind}"
        prior = jobs.get(key)
        attempts = prior["attempts"] if prior else 0
        if prior:
            try:
                modal.FunctionCall.from_id(prior["call_id"]).get(timeout=0)
            except TimeoutError:
                running += 1
                continue
            except Exception:
                # A failed worker can be retried on the next scheduled scan.
                pass
            else:
                # A successful worker committed its audit; a later scan sees it.
                continue
        if attempts >= 3:
            failed.append(key)
            continue
        call = tokenize_month.spawn(year, month, kind)
        jobs[key] = {"call_id": call.object_id, "attempts": attempts + 1}
        queued.append({"year": year, "month": month, "kind": kind,
                       "call_id": call.object_id})
    manifest_finalized = False
    if not ready and (cleaned_root / "COMPLETE.json").is_file():
        manifest_path = Path("/data/tokenized/manifest.json")
        if not manifest_path.is_file():
            # Validation will refuse to publish if any expected partition is absent.
            manifest_report.remote(True)
            manifest_finalized = True
    print(f"clean-ready={len(ready)} queued={len(queued)} running={running} failed={len(failed)}")
    return {"queued": queued, "running": running, "failed": failed,
            "manifest_finalized": manifest_finalized}


@app.function(image=image, volumes={"/data": volume}, timeout=3600)
def manifest_report(finalize: bool = False) -> dict:
    """Preview corpus balance now, or publish the final manifest when complete."""
    from data_manifest import build_manifest, write_manifest

    volume.reload()
    token_root = Path("/data/tokenized")
    clean_root = Path("/data/cleaned")
    manifest = (write_manifest(token_root, clean_root) if finalize else
                build_manifest(token_root, clean_root, require_complete=False))
    if finalize:
        volume.commit()
    return {"train_tokens": manifest["train_tokens"],
            "shards_by_split": {split: len(files) for split, files in manifest["splits"].items()},
            **manifest["stats"]}


@app.local_entrypoint()
def start() -> None:
    """Run one immediate scan; scheduled scans continue every five minutes."""
    import json

    print(json.dumps(dispatch_available.remote(), indent=2))

"""Modal entry point for ZeitgeistLM's remote data and training jobs."""

import modal


app = modal.App("zeitgeistlm")
data_volume = modal.Volume.from_name("zeitgeistlm-data", create_if_missing=True)
clean_jobs = modal.Dict.from_name("zeitgeistlm-clean-jobs", create_if_missing=True)
cleaning_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub>=0.34,<2", "pyarrow>=20,<24", "tiktoken>=0.9,<1")
    .add_local_python_source("clean_reddit")
)


@app.function(volumes={"/data": data_volume})
def status() -> str:
    """Verify that the deployed app can mount its persistent data volume."""
    from pathlib import Path

    # Later ingestion and training functions will share this volume mount.
    assert Path("/data").is_dir(), "ZeitgeistLM data volume is not mounted"
    return "ZeitgeistLM ready"


@app.function(
    image=cleaning_image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("zeitgeistlm")],
    cpu=2,
    memory=4096,
    timeout=43200,
    max_containers=16,
    retries=2,
)
def clean_month(year: int, month: int, kind: str, sample_shards: int = 0) -> dict:
    """Filter a monthly Arctic partition, optionally sampling source shards."""
    import json
    import os
    from pathlib import Path
    import shutil

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import tiktoken

    from clean_reddit import Cleaner, SUBREDDITS

    if kind not in {"comments", "submissions"} or not 1 <= month <= 12:
        raise ValueError("Expected a valid month and comments or submissions")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("The zeitgeistlm Modal Secret must contain HF_TOKEN")
    repo_id = "open-index/arctic"
    partition = f"data/{kind}/{year:04d}/{month:02d}"
    try:
        source_shards = sorted(
            (item for item in HfApi(token=token).list_repo_tree(
                repo_id, repo_type="dataset", path_in_repo=partition
            ) if item.path.endswith(".parquet")),
            key=lambda item: item.path,
        )
    except RemoteEntryNotFoundError:
        source_shards = []
    if not source_shards:
        # Some Arctic month/type combinations are absent; record the gap.
        report = {"year": year, "month": month, "kind": kind,
                  "status": "missing_source", "source_partition": partition}
        dataset_dir = "pilots" if sample_shards else "cleaned"
        destination = Path(f"/data/{dataset_dir}/{year:04d}/{month:02d}")
        destination.mkdir(parents=True, exist_ok=True)
        audit_path = destination / f"{kind}_audit.json"
        temporary_audit = destination / f"{kind}_audit.json.tmp"
        temporary_audit.write_text(json.dumps(report, indent=2))
        temporary_audit.replace(audit_path)
        data_volume.commit()
        return report
    if sample_shards < 0:
        raise ValueError("sample_shards must be nonnegative")
    if sample_shards and sample_shards < len(source_shards):
        # Spread pilot reads across the month in case shards are chronological.
        indices = [round(i * (len(source_shards) - 1) / (sample_shards - 1))
                   for i in range(sample_shards)] if sample_shards > 1 else [len(source_shards) // 2]
        selected_shards = [source_shards[i] for i in indices]
    else:
        selected_shards = source_shards
    shard_paths = [item.path for item in selected_shards]

    columns = ["id", "author", "subreddit", "created_utc", "score"]
    columns += (["body", "link_id", "parent_id"] if kind == "comments"
                else ["title", "selftext", "url"])
    schema = pa.schema([
        ("id", pa.string()), ("kind", pa.string()), ("subreddit", pa.string()),
        ("created_utc", pa.int64()), ("text", pa.string()), ("score", pa.int64()),
        ("thread_id", pa.string()), ("parent_id", pa.string()),
    ])
    cleaner = Cleaner(year, month, kind)
    target_subreddits = pa.array(sorted(SUBREDDITS))
    tokenizer = tiktoken.get_encoding("gpt2")
    token_counts = {}
    temp_dir = Path("/tmp/zeitgeistlm-clean")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output = temp_dir / "cleaned.parquet"
    writer = pq.ParquetWriter(output, schema, compression="zstd")
    pending = []
    try:
        for shard_path in shard_paths:
            # Raw Arctic shards land only on disposable container storage.
            raw_path = hf_hub_download(
                repo_id, shard_path, repo_type="dataset", cache_dir=str(temp_dir / "cache"),
                token=token,
            )
            reader = pq.ParquetFile(raw_path)
            for batch in reader.iter_batches(batch_size=8192, columns=columns):
                # Filter in Arrow before creating Python dictionaries for rows.
                selected = batch.filter(pc.is_in(
                    pc.utf8_lower(batch.column("subreddit")),
                    value_set=target_subreddits,
                ))
                skipped = batch.num_rows - selected.num_rows
                cleaner.counts["raw"] += skipped
                cleaner.counts["other_subreddit"] += skipped
                for row in selected.to_pylist():
                    cleaned = cleaner.clean(row)
                    if cleaned is None:
                        continue
                    subreddit = cleaned["subreddit"]
                    token_counts[subreddit] = token_counts.get(subreddit, 0) + len(
                        tokenizer.encode_ordinary(cleaned["text"])
                    )
                    pending.append(cleaned)
                    if len(pending) >= 4096:
                        writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                        pending.clear()
            # Release scratch space after each source shard is processed.
            shutil.rmtree(temp_dir / "cache", ignore_errors=True)
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
    finally:
        writer.close()

    report = cleaner.report()
    report["source_shards"] = shard_paths
    report["available_source_shards"] = len(source_shards)
    report["sampled_source_shards"] = len(selected_shards)
    report["sampled_source_bytes"] = sum(item.size or 0 for item in selected_shards)
    report["available_source_bytes"] = sum(item.size or 0 for item in source_shards)
    report["tokens_by_subreddit"] = token_counts
    report["total_text_tokens"] = sum(token_counts.values())
    dataset_dir = "pilots" if sample_shards else "cleaned"
    destination = Path(f"/data/{dataset_dir}/{year:04d}/{month:02d}")
    destination.mkdir(parents=True, exist_ok=True)
    # Publish the data before its audit marker; an interrupted copy is never complete.
    parquet_path = destination / f"{kind}.parquet"
    temporary_parquet = destination / f"{kind}.parquet.tmp"
    shutil.copy2(output, temporary_parquet)
    temporary_parquet.replace(parquet_path)
    audit_path = destination / f"{kind}_audit.json"
    temporary_audit = destination / f"{kind}_audit.json.tmp"
    temporary_audit.write_text(json.dumps(report, indent=2))
    temporary_audit.replace(audit_path)
    data_volume.commit()
    return report


@app.function(volumes={"/data": data_volume}, schedule=modal.Period(minutes=5),
              max_containers=1, timeout=600)
def dispatch_corpus() -> dict:
    """Fan out unfinished month/type partitions to independent CPU workers."""
    import json
    from pathlib import Path

    data_volume.reload()
    root = Path("/data/cleaned")
    # Each month/type is independent, so Modal can process up to 16 at once.
    partitions = [
        (year, month, kind)
        for year in range(2011, 2023)
        for month in range(1, (9 if year == 2022 else 13))
        for kind in ("submissions", "comments")
    ]
    complete = 0
    running = 0
    queued = 0
    failed = []
    for year, month, kind in partitions:
        destination = root / f"{year:04d}" / f"{month:02d}"
        audit_path = destination / f"{kind}_audit.json"
        parquet_path = destination / f"{kind}.parquet"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text())
            if audit.get("status") == "missing_source" or parquet_path.is_file():
                complete += 1
                continue
        key = f"{year:04d}/{month:02d}/{kind}"
        prior = clean_jobs.get(key)
        attempts = prior["attempts"] if prior else 0
        if prior:
            try:
                modal.FunctionCall.from_id(prior["call_id"]).get(timeout=0)
            except TimeoutError:
                running += 1
                continue
            except Exception:
                # Failed inputs are retried at the next scheduled scan.
                pass
            else:
                # Allow one reload cycle, then rerun a lost or incomplete commit.
                checks = prior.get("success_checks", 0) + 1
                if checks < 2:
                    clean_jobs[key] = {**prior, "success_checks": checks}
                    continue
        if attempts >= 3:
            failed.append(key)
            continue
        call = clean_month.spawn(year, month, kind)
        clean_jobs[key] = {"call_id": call.object_id, "attempts": attempts + 1}
        queued += 1
    if complete == len(partitions) and not (root / "COMPLETE.json").is_file():
        (root / "COMPLETE.json").write_text(json.dumps({"partitions": complete}))
        data_volume.commit()
    print(f"clean complete={complete}/{len(partitions)} queued={queued} "
          f"running={running} failed={len(failed)}")
    return {"complete": complete, "total": len(partitions),
            "queued": queued, "running": running, "failed": failed}


@app.local_entrypoint()
def sample(year: int = 2011, month: int = 1, shards: int = 0) -> None:
    """Run both Reddit content types and print their compact audit reports."""
    import json

    for kind in ("submissions", "comments"):
        report = clean_month.remote(year, month, kind, shards)
        print(json.dumps(report, indent=2))


@app.function(image=cleaning_image, volumes={"/data": data_volume})
def inspect_cleaned(year: int, month: int, kind: str, pilot: bool = False) -> list[dict]:
    """Return a small deterministic review sample without downloading the corpus."""
    import hashlib
    import heapq
    from pathlib import Path
    import pyarrow.parquet as pq

    dataset_dir = "pilots" if pilot else "cleaned"
    path = Path(f"/data/{dataset_dir}/{year:04d}/{month:02d}/{kind}.parquet")
    chosen = []
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=4096, columns=["id", "subreddit", "text"]
    ):
        for row in batch.to_pylist():
            # Keep the 12 smallest stable hashes without loading the full file.
            priority = -int.from_bytes(
                hashlib.blake2b(row["id"].encode(), digest_size=8).digest(), "big"
            )
            item = (priority, row["id"], row)
            if len(chosen) < 12:
                heapq.heappush(chosen, item)
            elif item > chosen[0]:
                heapq.heapreplace(chosen, item)
    return [{**row, "text": row["text"][:240]} for _, _, row in sorted(chosen, reverse=True)]


@app.local_entrypoint()
def inspect(year: int = 2011, month: int = 1, pilot: bool = False) -> None:
    """Print review samples from cleaned submissions and comments."""
    import json

    for kind in ("submissions", "comments"):
        print(kind, json.dumps(inspect_cleaned.remote(year, month, kind, pilot), indent=2))

"""Small remote spot check of cleaned Reddit shards on the Modal Volume."""

import json
import random
import re
from pathlib import Path

import modal


app = modal.App("zeitgeistlm-audit")
volume = modal.Volume.from_name("zeitgeistlm-data")
clean_jobs = modal.Dict.from_name("zeitgeistlm-clean-jobs")
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("pyarrow>=20,<24")


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def sample_cleaned() -> dict:
    """Read a few random row groups without copying Parquet files locally."""
    import pyarrow.parquet as pq

    volume.reload()
    rng = random.Random(20260919)
    root = Path("/data/cleaned")
    years = (2013, 2017, 2020, 2022)
    result = []
    for year in years:
        for kind in ("submissions", "comments"):
            candidates = sorted(root.glob(f"{year}/*/{kind}.parquet"))
            if not candidates:
                result.append({"year": year, "kind": kind, "status": "no completed shard"})
                continue
            path = rng.choice(candidates)
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows == 0:
                result.append({"partition": str(path.relative_to(root)), "status": "empty"})
                continue
            # One row group per partition keeps the read small and varied.
            group = rng.randrange(parquet.num_row_groups)
            rows = parquet.read_row_group(group, columns=["id", "created_utc", "subreddit", "text"]).to_pylist()
            rows = rng.sample(rows, min(12, len(rows)))
            issues = []
            for row in rows:
                value = row["text"] or ""
                flags = [
                    name for name, matched in (
                        ("empty_or_removed", value.strip().casefold() in {"", "[deleted]", "[removed]"}),
                        ("image_markup", bool(re.search(r"!\[[^]]*\]\([^)]*\)|<img\b", value, re.I))),
                        ("media_url", bool(re.search(r"https?://\S+\.(?:png|jpe?g|gif|webp)", value, re.I))),
                        ("bot_footer", "i am a bot, and this action was performed automatically" in value.casefold()),
                        ("too_long", len(value) > 12000),
                    ) if matched
                ]
                if flags:
                    issues.append({"id": row["id"], "flags": flags})
            audit = json.loads(path.with_name(f"{kind}_audit.json").read_text())
            result.append({
                "partition": str(path.relative_to(root)),
                "rows_in_parquet": parquet.metadata.num_rows,
                "sampled": len(rows),
                "issues": issues,
                "counts": audit.get("counts", {}),
                "sample": [{"subreddit": row["subreddit"], "text": row["text"][:180]}
                           for row in rows[:4]],
            })
    return {"partitions": result}


@app.function(volumes={"/data": volume}, timeout=300)
def remaining_cleaned() -> list[dict]:
    """List unfinished partitions and their recorded dispatcher attempts."""
    volume.reload()
    root = Path("/data/cleaned")
    remaining = []
    for year in range(2011, 2023):
        for month in range(1, 9 if year == 2022 else 13):
            for kind in ("submissions", "comments"):
                directory = root / f"{year:04d}" / f"{month:02d}"
                audit = directory / f"{kind}_audit.json"
                if audit.is_file():
                    report = json.loads(audit.read_text())
                    if report.get("status") == "missing_source" or (directory / f"{kind}.parquet").is_file():
                        continue
                key = f"{year:04d}/{month:02d}/{kind}"
                job = clean_jobs.get(key) or {}
                remaining.append({"partition": key, "attempts": job.get("attempts", 0),
                                  "call_id": job.get("call_id")})
    return remaining


@app.local_entrypoint()
def main() -> None:
    """Print a compact unfinished-partition summary without local data downloads."""
    print(json.dumps(remaining_cleaned.remote(), indent=2))

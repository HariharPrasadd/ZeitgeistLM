"""Unsupervised, chronological similarity graph over broadly sampled Reddit posts."""

from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np


SAMPLE_PER_YEAR_SUBREDDIT = 150
MAX_TEXT_CHARS = 500
MAX_TOKENS = 128
TIME_PENALTY_PER_YEAR = 0.02
MIN_COSINE = 0.85


def sample_year(root: Path, year: int) -> dict:
    """Keep stable hash samples from every available month without loading a shard fully."""
    import pyarrow.parquet as pq

    buckets = defaultdict(list)
    seen = Counter()
    scanned = []
    for path in sorted((root / str(year)).glob("*/submissions.parquet")):
        scanned.append(str(path.relative_to(root)))
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=8192,
                                          columns=["id", "text", "subreddit", "created_utc"]):
            for row in batch.to_pylist():
                text = (row["text"] or "").strip()
                if not 20 <= len(text) <= MAX_TEXT_CHARS:
                    continue
                if len(text.split()) < 4 or text.lower().startswith(("i am a bot", "your post has been")):
                    continue
                subreddit = row["subreddit"]
                if not subreddit:
                    continue
                seen[subreddit] += 1
                rank = int.from_bytes(hashlib.blake2b(
                    f"{year}:{row['id']}".encode(), digest_size=8).digest(), "big")
                item = {"id": str(row["id"]), "text": text, "subreddit": subreddit,
                        "created_utc": int(row["created_utc"]), "year": year}
                heap = buckets[subreddit]
                candidate = (-rank, row["id"], item)
                if len(heap) < SAMPLE_PER_YEAR_SUBREDDIT:
                    heapq.heappush(heap, candidate)
                elif candidate[0] > heap[0][0]:
                    heapq.heapreplace(heap, candidate)
    rows = [item for heap in buckets.values() for _, _, item in heap]
    return {"rows": rows, "months": scanned, "eligible_by_subreddit": dict(seen)}


def assemble(root: Path, years: range) -> dict:
    """Combine annual samples and remove exact repeats across all years and communities."""
    rows = []
    months = []
    eligible = Counter()
    for year in years:
        path = root / "analysis/broad_genealogy" / f"sample_{year}.json"
        if not path.exists():
            raise FileNotFoundError(f"Annual sample is missing: {path}")
        part = json.loads(path.read_text())
        rows.extend(part["rows"])
        months.extend(part["months"])
        eligible.update(part["eligible_by_subreddit"])
    rows.sort(key=lambda row: (row["created_utc"], row["id"]))
    unique = []
    seen_ids, seen_texts = set(), set()
    for row in rows:
        canonical = " ".join(row["text"].casefold().split())
        if row["id"] in seen_ids or canonical in seen_texts:
            continue
        seen_ids.add(row["id"])
        seen_texts.add(canonical)
        unique.append(row)
    output = root / "analysis/broad_genealogy/sample.json"
    output.write_text(json.dumps(unique, ensure_ascii=False))
    return {"sampled_before_dedup": len(rows), "sampled_after_dedup": len(unique),
            "months_scanned": len(months), "years": dict(Counter(row["year"] for row in unique)),
            "subreddits": dict(Counter(row["subreddit"] for row in unique)),
            "eligible_by_subreddit": dict(eligible)}


def embed(model, manifest, rows):
    """Mean-pool block-10 content states at one reference time for every post."""
    import tiktoken
    import torch
    from zeitgeist_inference import normalized_time, parse_date

    encoder = tiktoken.get_encoding("gpt2")
    encoded = [encoder.encode(row["text"])[:MAX_TOKENS] for row in rows]
    order = sorted(range(len(rows)), key=lambda i: len(encoded[i]))
    vectors = np.empty((len(rows), model.config.n_embd), dtype=np.float32)
    captured = []
    hook = model.transformer.h[9].register_forward_hook(
        lambda module, inputs, output: captured.append(output))
    tau = normalized_time(parse_date("2018-07-01"), manifest)
    try:
        for start in range(0, len(rows), 16):
            ids = order[start:start + 16]
            width = max(len(encoded[i]) for i in ids)
            matrix = torch.full((len(ids), width), encoder.eot_token, device="cuda",
                                dtype=torch.long)
            mask = torch.zeros((len(ids), width), device="cuda", dtype=torch.float32)
            for b, i in enumerate(ids):
                length = len(encoded[i])
                matrix[b, :length] = torch.tensor(encoded[i], device="cuda")
                mask[b, :length] = 1
            captured.clear()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(matrix, time_value=torch.full((len(ids),), tau, device="cuda"))
            # Position zero is the time pseudo-token; padded positions get zero weight.
            states = captured[0][:, 1:, :].float()
            pooled = (states * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
            vectors[ids] = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
    finally:
        hook.remove()
    return vectors


def build_graph(rows, vectors, alpha=TIME_PENALTY_PER_YEAR):
    """Find each node's best earlier neighbor, then reject links below a null threshold."""
    import torch

    device = "cuda"
    matrix = torch.from_numpy(vectors).to(device)
    dates = torch.tensor([row["created_utc"] for row in rows], device=device)
    n = len(rows)
    rng = np.random.default_rng(20260920)
    pair_a = rng.integers(0, n, size=100000)
    pair_b = rng.integers(0, n, size=100000)
    null_cosines = np.sum(vectors[pair_a] * vectors[pair_b], axis=1)
    random_pair_q999 = float(np.quantile(null_cosines[pair_a != pair_b], 0.999))
    # A nearest-neighbor search tests thousands of candidates per post, so a
    # random-pair quantile alone is too permissive after multiple comparisons.
    threshold = max(random_pair_q999, MIN_COSINE)
    edges = []
    # Chunked exact search avoids retaining an N-by-N matrix on the GPU.
    for start in range(0, n, 256):
        stop = min(n, start + 256)
        similarity = matrix[start:stop] @ matrix.T
        gap = (dates[start:stop, None] - dates[None, :]) / 31557600
        score = similarity - alpha * gap
        score.masked_fill_(gap <= 0, -1e9)
        value, parent = score.max(dim=1)
        parents = parent.cpu().tolist()
        best_scores = value.cpu().tolist()
        best_cosines = similarity.gather(1, parent[:, None]).squeeze(1).cpu().tolist()
        best_gaps = gap.gather(1, parent[:, None]).squeeze(1).cpu().tolist()
        for offset, j in enumerate(parents):
            i = start + offset
            cosine = best_cosines[offset]
            if i == 0 or cosine < threshold or best_scores[offset] < -1e8:
                continue
            edges.append({"child": i, "parent": j, "cosine": cosine,
                          "gap_years": best_gaps[offset], "score": best_scores[offset]})
    return edges, {"random_pair_q999_cosine": random_pair_q999,
                   "absolute_cosine_floor": MIN_COSINE,
                   "effective_cosine_threshold": threshold,
                   "time_penalty_per_year": alpha,
                   "null_pairs": 100000}


def summarize(rows, edges, selection):
    """Surface connected groups and high-degree ancestors for human inspection."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(len(rows)))
    graph.add_edges_from((edge["child"], edge["parent"]) for edge in edges)
    children = Counter(edge["parent"] for edge in edges)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    groups = []
    for component in components[:30]:
        if len(component) < 3:
            break
        members = sorted(component, key=lambda i: rows[i]["created_utc"])
        hubs = sorted(component, key=lambda i: children[i], reverse=True)[:3]
        groups.append({"size": len(component),
                       "years": [rows[members[0]]["year"], rows[members[-1]]["year"]],
                       "subreddits": dict(Counter(rows[i]["subreddit"] for i in members)),
                       "earliest": [rows[i] for i in members[:3]],
                       "hubs": [{**rows[i], "children": children[i]} for i in hubs]})
    return {"nodes": len(rows), "edges": len(edges),
            "edge_rate": len(edges) / max(1, len(rows) - 1),
            "components_size_at_least_3": sum(len(c) >= 3 for c in components),
            "largest_component_sizes": [len(c) for c in components[:20]],
            "cross_subreddit_edge_rate": float(np.mean([
                rows[e["child"]]["subreddit"] != rows[e["parent"]]["subreddit"]
                for e in edges])) if edges else None,
            "top_groups": groups, "selection": selection,
            "sample_edges": [{"child": rows[e["child"]], "parent": rows[e["parent"]],
                              "cosine": e["cosine"], "gap_years": e["gap_years"]}
                             for e in sorted(edges, key=lambda e: e["cosine"], reverse=True)[:30]]}


def run(root: Path):
    """Save the full graph remotely and return only its compact inspection summary."""
    from zeitgeist_inference import load_model

    output = root / "analysis/broad_genealogy"
    rows = json.loads((output / "sample.json").read_text())
    if len(rows) < 1000:
        raise ValueError(f"Broad sample too small: {len(rows)} posts")
    model, manifest, checkpoint = load_model(root / "checkpoints/tokens_1000M.pt",
                                              root / "tokenized/manifest.json", "cuda")
    vectors = embed(model, manifest, rows)
    edges, selection = build_graph(rows, vectors)
    np.save(output / "embeddings_block10_fixed2018.npy", vectors)
    (output / "edges.json").write_text(json.dumps(edges))
    summary = summarize(rows, edges, selection)
    summary["checkpoint_tokens_seen"] = checkpoint["tokens_seen"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def refine_saved(root: Path):
    """Apply the stricter edge floor to a completed graph without GPU recompute."""
    output = root / "analysis/broad_genealogy"
    rows = json.loads((output / "sample.json").read_text())
    original = json.loads((output / "summary.json").read_text())
    edges = json.loads((output / "edges.json").read_text())
    if "absolute_cosine_floor" not in original["selection"]:
        (output / "summary_initial.json").write_text(json.dumps(original, indent=2))
        (output / "edges_initial.json").write_text(json.dumps(edges))
    threshold = max(original["selection"]["random_pair_q999_cosine"], MIN_COSINE)
    edges = [edge for edge in edges if edge["cosine"] >= threshold]
    selection = {**original["selection"], "absolute_cosine_floor": MIN_COSINE,
                 "effective_cosine_threshold": threshold}
    summary = summarize(rows, edges, selection)
    summary["checkpoint_tokens_seen"] = original["checkpoint_tokens_seen"]
    (output / "edges.json").write_text(json.dumps(edges))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

"""Fixed-checkpoint generation, forecasting, temporal geometry, and genealogy."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F
import tiktoken

from zeitgeist_inference import load_model, normalized_time, parse_date


FAMILIES = {
    "doge": r"\bdoge\b", "pepe": r"\bpepe\b", "harambe": r"\bharambe\b",
    "stonks": r"\bstonks\b", "big_chungus": r"\bbig chungus\b",
    "among_us": r"\bamong us\b|\bamogus\b", "nobody": r"\bnobody\s*:",
    "this_is_fine": r"\bthis is fine\b", "me_irl": r"\bme[_ ]irl\b",
    "bro_really_thought": r"\bbro really thought\b",
}
PROMPTS = ["bro really thought", "when you realize", "nobody:"]
DATES = ["2016", "2020", "2022", "2023"]
CHECKPOINTS = ["tokens_100M.pt", "tokens_250M.pt", "tokens_500M.pt",
               "tokens_1000M.pt"]


def collect_phrase_examples(root: Path):
    """Use a deterministic hash reservoir per phrase/year, without downloading raw rows."""
    import pyarrow.parquet as pq

    patterns = {name: re.compile(expression, re.I) for name, expression in FAMILIES.items()}
    selected = defaultdict(list)
    counts = Counter()
    scanned = []
    for year in range(2011, 2023):
        for month in (3, 9):
            path = root / f"{year}/{month:02d}/submissions.parquet"
            if not path.exists():
                continue
            scanned.append(str(path.relative_to(root)))
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=8192,
                                              columns=["id", "text", "subreddit", "created_utc"]):
                for row in batch.to_pylist():
                    content = row["text"] or ""
                    if not 10 <= len(content) <= 400:
                        continue
                    for family, pattern in patterns.items():
                        if not pattern.search(content):
                            continue
                        key = (family, year)
                        counts[key] += 1
                        # Choose a stable sample across a full month, not its first rows.
                        rank = int.from_bytes(hashlib.blake2b(
                            f"{year}:{row['id']}:{family}".encode(), digest_size=8).digest(), "big")
                        item = {"family": family, "year": year, "id": row["id"],
                                "created_utc": int(row["created_utc"]),
                                "subreddit": row["subreddit"], "text": content,
                                "sample_rank": rank}
                        bucket = selected[key]
                        bucket.append(item)
                        bucket.sort(key=lambda value: value["sample_rank"])
                        if len(bucket) > 8:
                            bucket.pop()
    examples = [item for bucket in selected.values() for item in bucket]
    examples.sort(key=lambda row: (row["created_utc"], row["id"]))
    audit = {"months_scanned": scanned, "matching_rows_by_family_year": {
        f"{family}:{year}": count for (family, year), count in sorted(counts.items())},
        "sampled_by_family": dict(Counter(row["family"] for row in examples)),
        "sampled_total": len(examples), "sampling": "8 smallest stable hashes per family/year; March and September submissions"}
    return examples, audit


def sample_windows(root, manifest, split, per_stratum=12):
    """Draw the same deterministic rows for every counterfactual time condition."""
    rng = np.random.default_rng(20260920)
    groups = defaultdict(list)
    for spec in manifest["splits"][split]:
        groups[(spec["year"], spec["month"], spec["kind"])].append(spec)
    results = []
    for (year, month, kind), specs in sorted(groups.items()):
        lengths = np.array([item["windows"] for item in specs])
        picks = rng.choice(lengths.sum(), size=min(per_stratum, lengths.sum()), replace=False)
        offsets = np.cumsum(lengths)
        for spec_id in np.unique(np.searchsorted(offsets, picks, side="right")):
            spec = specs[int(spec_id)]
            base = 0 if spec_id == 0 else offsets[spec_id - 1]
            indices = picks[np.searchsorted(offsets, picks, side="right") == spec_id] - base
            tokens = np.load(root / spec["tokens"], mmap_mode="r")
            dates = np.load(root / spec["created_utc"], mmap_mode="r")
            for index in indices:
                results.append((np.array(tokens[index], dtype=np.int64), int(dates[index]),
                                year, month, kind))
    return results


def score_windows(model, manifest, windows, device):
    """Paired true-date and frozen-2020 loss on identical held-out token windows."""
    fixed = normalized_time(parse_date("2020-07-01"), manifest)
    reports = []
    for offset in range(0, len(windows), 8):
        chunk = windows[offset:offset + 8]
        matrix = torch.as_tensor(np.stack([row[0] for row in chunk]), device=device)
        true_tau = torch.tensor([normalized_time(row[1], manifest) for row in chunk],
                                dtype=torch.float32, device=device)
        losses = []
        for tau in (true_tau, torch.full_like(true_tau, fixed)):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(matrix[:, :-1], time_value=tau)
                per_token = F.cross_entropy(logits.float().transpose(1, 2), matrix[:, 1:],
                                            reduction="none")
                losses.append(per_token.mean(dim=1).cpu().tolist())
        for row, actual, frozen in zip(chunk, *losses):
            reports.append({"year": row[2], "month": row[3], "kind": row[4],
                            "created_utc": row[1], "actual_time_loss": actual,
                            "fixed_2020_loss": frozen})
    return reports


def generation_and_sensitivity(model, manifest, encoder, device, prompts=PROMPTS,
                               dates=DATES, new_tokens=48):
    """Generate matched-seed continuations and measure full-vocabulary changes."""
    from generate_zeitgeist import generate

    samples, sensitivities = [], []
    for prompt in prompts:
        ids = encoder.encode(prompt)
        x = torch.tensor(ids, device=device)[None, :]
        probabilities = []
        for date in dates:
            tau = normalized_time(parse_date(date), manifest)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x, time_value=torch.tensor([tau], device=device))
                probs = F.softmax(logits[0, -1, :encoder.n_vocab].float(), dim=-1)
            probabilities.append(probs)
            tokens = generate(model, ids, tau, new_tokens, 50, 0.8, 42, device,
                              encoder.n_vocab)
            samples.append({"prompt": prompt, "date": date, "tau": tau,
                            "text": encoder.decode(tokens),
                            "continuation": encoder.decode(tokens[len(ids):])})
        p, q = probabilities[0], probabilities[-1]
        middle = (p + q) / 2
        js = 0.5 * (F.kl_div(middle.log(), p, reduction="sum") +
                    F.kl_div(middle.log(), q, reduction="sum"))
        sensitivities.append({"prompt": prompt, "dates": [dates[0], dates[-1]],
                              "next_token_js_nats": js.item(),
                              "total_variation": (p - q).abs().sum().item() / 2})
    return samples, sensitivities


def geometry(model, manifest):
    """Measure the affine temporal embedding; PCA must be rank one by design."""
    layer = model.transformer.time_encoder
    values = torch.linspace(0, 1.25, 101, device=layer.weight.device)[:, None]
    with torch.no_grad():
        embeddings = layer(values).float()
        centered = embeddings - embeddings.mean(0)
        singular = torch.linalg.svdvals(centered)
        direction = layer.weight.float()[:, 0]
    return {"sampled_tau": [0, 1.25], "time_weight_norm": direction.norm().item(),
            "time_bias_norm": layer.bias.float().norm().item(),
            "first_pc_variance_fraction": (singular[0] ** 2 / singular.square().sum()).item(),
            "second_to_first_singular_ratio": (singular[1] / singular[0]).item(),
            "normalized_year_2023": normalized_time(parse_date("2023"), manifest)}


def genealogy(model, manifest, encoder, examples, device):
    """Link final-token hidden states to earlier examples, with a temporal control."""
    if not examples:
        return {"error": "No phrase examples available"}
    captured = []
    hook = model.transformer.ln_f.register_forward_hook(
        lambda module, inputs, output: captured.append(output[:, -1].detach().float().cpu()))
    vectors = []
    try:
        for row in examples:
            ids = encoder.encode(row["text"])[:128]
            if not ids:
                continue
            x = torch.tensor(ids, device=device)[None, :]
            tau = normalized_time(row["created_utc"], manifest)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, time_value=torch.tensor([tau], device=device))
            vectors.append(captured.pop()[0])
    finally:
        hook.remove()
    embeddings = F.normalize(torch.stack(vectors), dim=1)
    similarities = (embeddings @ embeddings.T).numpy()
    years = np.array([row["created_utc"] for row in examples]) / (365.25 * 86400)
    edges = []
    for i in range(1, len(examples)):
        earlier = np.flatnonzero(years[:i] < years[i])
        if len(earlier) == 0:
            continue
        distance = years[i] - years[earlier]
        plain = earlier[np.argmax(similarities[i, earlier])]
        penalized = earlier[np.argmax(similarities[i, earlier] - 0.03 * distance)]
        edges.append({"child": i, "parent": int(plain),
                      "penalized_parent": int(penalized),
                      "cosine": float(similarities[i, plain]),
                      "penalized_cosine": float(similarities[i, penalized]),
                      "distance_years": float(years[i] - years[plain])})
    family = [row["family"] for row in examples]
    plain_same = np.mean([family[e["child"]] == family[e["parent"]] for e in edges])
    penalized_same = np.mean([family[e["child"]] == family[e["penalized_parent"]] for e in edges])
    rng = np.random.default_rng(7331)
    null = []
    for _ in range(100):
        shuffled = rng.permutation(family)
        null.append(np.mean([shuffled[e["child"]] == shuffled[e["parent"]] for e in edges]))
    children = Counter(e["penalized_parent"] for e in edges)
    branches = [dict(index=index, family=family[index], year=examples[index]["year"],
                     children=count, text=examples[index]["text"][:140])
                for index, count in children.most_common(15) if count >= 2]
    return {"examples": [{key: value for key, value in row.items() if key != "sample_rank"}
                         for row in examples], "edges": edges, "alpha_per_year": 0.03,
            "same_family_plain": float(plain_same),
            "same_family_penalized": float(penalized_same),
            "shuffled_family_null_mean": float(np.mean(null)),
            "shuffled_family_null_sd": float(np.std(null)),
            "branch_candidates": branches}


def analyze(root: Path):
    """Run all analyses at fixed checkpoints, keeping the 2022 test evaluation final."""
    device = "cuda"
    manifest_path = root / "tokenized/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    encoder = tiktoken.get_encoding("gpt2")
    report = {"settings": {"checkpoint_names": CHECKPOINTS,
                            "generation_dates": DATES, "generation_seed": 42,
                            "sample_windows_per_month_kind": 12,
                            "future_test": "2022; evaluated after fixing 1B checkpoint"}}
    probe = ["bro really thought", "nobody:", "when you realize"]
    emergence = []
    for name in CHECKPOINTS:
        model, _, checkpoint = load_model(root / "checkpoints" / name, manifest_path, device)
        _, sensitivity = generation_and_sensitivity(model, manifest, encoder, device,
                                                     prompts=probe, dates=["2016", "2023"],
                                                     new_tokens=12)
        emergence.append({"checkpoint": name, "tokens_seen": checkpoint["tokens_seen"],
                          "geometry": geometry(model, manifest),
                          "sensitivity": sensitivity})
        del model
        torch.cuda.empty_cache()
    report["emergence"] = emergence

    model, _, checkpoint = load_model(root / "checkpoints/tokens_1000M.pt",
                                      manifest_path, device)
    report["checkpoint_tokens_seen"] = checkpoint["tokens_seen"]
    report["generation"], report["generation_sensitivity"] = (
        generation_and_sensitivity(model, manifest, encoder, device))
    report["forecast"] = {}
    for split in ("val", "test"):
        windows = sample_windows(root / "tokenized", manifest, split)
        report["forecast"][split] = score_windows(model, manifest, windows, device)
    candidates = root / "analysis/phrase_examples.json"
    report["genealogy"] = genealogy(model, manifest, encoder,
                                    json.loads(candidates.read_text()) if candidates.exists() else [],
                                    device)
    return report

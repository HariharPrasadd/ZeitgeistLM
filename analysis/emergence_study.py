"""Retrospective phrase-growth test with candidates and contexts fixed pre-2021."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


WORD = re.compile(r"[A-Za-z][A-Za-z']*")
STOP = {"the", "and", "for", "with", "that", "this", "from", "have", "you",
        "your", "are", "was", "what", "when", "where", "how", "why", "they",
        "them", "our", "not", "but", "all", "out", "can", "one", "two"}
TARGET_CANDIDATES = 1000
CONTEXTS_PER_PHRASE = 3


def candidates(root: Path) -> dict:
    """Select phrases and three natural prefixes without reading any future text."""
    rows = json.loads((root / "analysis/broad_genealogy/sample.json").read_text())
    count = Counter()
    contexts = defaultdict(list)
    for row in rows:
        if row["year"] > 2020 or row["year"] < 2017:
            continue
        text = row["text"].splitlines()[0][:300]
        words = list(WORD.finditer(text))
        seen = set()
        for i in range(4, len(words)):
            for size in (2, 3, 4):
                if i + size > len(words):
                    continue
                terms = [m.group().lower() for m in words[i:i + size]]
                if terms[0] in STOP and terms[-1] in STOP:
                    continue
                if not any(len(w) >= 5 and w not in STOP for w in terms):
                    continue
                phrase = " ".join(terms)
                if phrase in seen:
                    continue
                seen.add(phrase)
                count[phrase] += 1
                if len(contexts[phrase]) < CONTEXTS_PER_PHRASE:
                    prefix = text[:words[i].start()].strip()
                    if len(prefix.split()) >= 4:
                        contexts[phrase].append(" ".join(prefix.split()[-16:]))
    eligible = [phrase for phrase, n in count.items()
                if 3 <= n <= 300 and len(contexts[phrase]) == CONTEXTS_PER_PHRASE]
    # A deterministic hash avoids selecting candidates using future outcomes.
    eligible.sort(key=lambda phrase: hashlib.blake2b(phrase.encode(), digest_size=8).digest())
    chosen = eligible[:TARGET_CANDIDATES]
    result = {"source": "2017-2020 broad stratified submission sample only",
              "eligible_phrases": len(eligible), "candidates": [
                  {"phrase": phrase, "sample_count": count[phrase],
                   "contexts": contexts[phrase]} for phrase in chosen]}
    output = root / "analysis/emergence"
    output.mkdir(exist_ok=True)
    (output / "candidates.json").write_text(json.dumps(result))
    return {"eligible_phrases": len(eligible), "selected": len(chosen)}


def count_year(root: Path, year: int) -> dict:
    """Count selected phrases per month over all cleaned submissions."""
    import ahocorasick
    import pyarrow.parquet as pq

    phrases = json.loads((root / "analysis/emergence/candidates.json").read_text())["candidates"]
    automaton = ahocorasick.Automaton()
    for i, item in enumerate(phrases):
        automaton.add_word(" " + item["phrase"] + " ", i)
    automaton.make_automaton()
    months = {}
    for path in sorted((root / "cleaned" / str(year)).glob("*/submissions.parquet")):
        hits = [0] * len(phrases)
        total = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["text"]):
            for row in batch.to_pylist():
                text = (row["text"] or "").splitlines()[0]
                if not text:
                    continue
                total += 1
                normalized = " " + " ".join(m.group().lower() for m in WORD.finditer(text)) + " "
                for i in {value for _, value in automaton.iter(normalized)}:
                    hits[i] += 1
        months[path.parent.name] = {"documents": total, "hits": hits}
    result = {"year": year, "months": months}
    (root / "analysis/emergence" / f"counts_{year}.json").write_text(json.dumps(result))
    return {"year": year, "documents": sum(x["documents"] for x in months.values()),
            "months": len(months)}


def score(root: Path, checkpoint_name: str = "tokens_1000M.pt",
          score_filename: str = "scores.json") -> dict:
    """Score each phrase in three real pre-cutoff prefixes at three time points."""
    import numpy as np
    import tiktoken
    import torch
    from torch.nn import functional as F
    from zeitgeist_inference import load_model, normalized_time, parse_date

    data = json.loads((root / "analysis/emergence/candidates.json").read_text())
    model, manifest, checkpoint = load_model(root / "checkpoints" / checkpoint_name,
                                              root / "tokenized/manifest.json", "cuda")
    encoder = tiktoken.get_encoding("gpt2")
    examples = []
    for i, item in enumerate(data["candidates"]):
        for context in item["contexts"]:
            prefix = encoder.encode_ordinary(context)[-64:]
            phrase = encoder.encode_ordinary(" " + item["phrase"])
            if prefix and phrase:
                examples.append((i, prefix + phrase, len(prefix), len(phrase)))
    dates = ["2018-07-01", "2019-07-01", "2020-07-01"]
    scores = np.zeros((len(data["candidates"]), len(dates)), dtype=np.float64)
    counts = np.zeros(len(data["candidates"]), dtype=np.int32)
    for i, *_ in examples:
        counts[i] += 1
    for start in range(0, len(examples), 16):
        batch = examples[start:start + 16]
        width = max(len(ids) for _, ids, _, _ in batch)
        x = torch.full((len(batch), width - 1), encoder.eot_token,
                       dtype=torch.long, device="cuda")
        y = torch.full_like(x, encoder.eot_token)
        mask = torch.zeros_like(x, dtype=torch.float32)
        for b, (_, ids, prefix_len, phrase_len) in enumerate(batch):
            x[b, :len(ids) - 1] = torch.tensor(ids[:-1], device="cuda")
            y[b, :len(ids) - 1] = torch.tensor(ids[1:], device="cuda")
            mask[b, prefix_len - 1:prefix_len - 1 + phrase_len] = 1
        for j, date in enumerate(dates):
            tau = normalized_time(parse_date(date), manifest)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x, time_value=torch.full((len(batch),), tau, device="cuda"))
                losses = F.cross_entropy(logits.float().transpose(1, 2), y,
                                         reduction="none")
            values = (losses * mask).sum(1) / mask.sum(1)
            for b, (i, *_rest) in enumerate(batch):
                scores[i, j] -= float(values[b])
    scores /= counts[:, None]
    result = {"checkpoint_name": checkpoint_name,
              "checkpoint_tokens_seen": checkpoint["tokens_seen"], "dates": dates,
              "context_count": counts.tolist(), "mean_logprob_per_token": scores.tolist(),
              "context_policy": "Three natural preceding text spans from 2017-2020 submissions; no subreddit prefix added"}
    (root / "analysis/emergence" / score_filename).write_text(json.dumps(result))
    return {"candidates": len(data["candidates"]), "scored_contexts": len(examples)}


def summarize(root: Path, score_filename: str = "scores.json") -> dict:
    """Compare pre-cutoff model slope with later frequency growth and baseline trend."""
    import numpy as np
    from scipy.stats import spearmanr

    output = root / "analysis/emergence"
    candidates = json.loads((output / "candidates.json").read_text())["candidates"]
    score_data = json.loads((output / score_filename).read_text())
    scores = score_data["mean_logprob_per_token"]
    years = {year: json.loads((output / f"counts_{year}.json").read_text())
             for year in range(2018, 2023)}
    def total(indices, candidate):
        docs = sum(part["documents"] for year in indices for part in years[year]["months"].values())
        hits = sum(part["hits"][candidate] for year in indices
                   for part in years[year]["months"].values())
        return (hits + .5) / docs * 1_000_000, hits
    rows = []
    for i, item in enumerate(candidates):
        r18, h18 = total([2018], i)
        r19, h19 = total([2019], i)
        r20, h20 = total([2020], i)
        future, h_future = total([2021, 2022], i)
        prior, _ = total([2019, 2020], i)
        rows.append({"phrase": item["phrase"], "model_slope": scores[i][2] - scores[i][0],
                     "frequency_slope": float(np.log(r20 / r19)),
                     "future_growth": float(np.log(future / prior)),
                     "train_hits_2019_2020": h19 + h20, "future_hits": h_future,
                     "rates_per_million": {"2018": r18, "2019": r19,
                                           "2020": r20, "2021_2022": future}})
    model = np.array([r["model_slope"] for r in rows])
    baseline = np.array([r["frequency_slope"] for r in rows])
    outcome = np.array([r["future_growth"] for r in rows])
    report = {"n": len(rows), "checkpoint_tokens_seen": score_data["checkpoint_tokens_seen"],
              "model_spearman": float(spearmanr(model, outcome).statistic),
              "frequency_trend_spearman": float(spearmanr(baseline, outcome).statistic),
              "top_model_decile_mean_growth": float(outcome[model >= np.quantile(model, .9)].mean()),
              "overall_mean_growth": float(outcome.mean()),
              "notes": "Exploratory retrospective test; candidates and contexts fixed from pre-2021 sample",
              "examples": sorted(rows, key=lambda r: r["model_slope"], reverse=True)[:30],
              "rows": rows}
    return report

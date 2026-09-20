"""Compare semantic and lexical genealogy links under controlled time inputs."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from zeitgeist_inference import load_model, normalized_time, parse_date


LAYERS = (3, 6, 9, 11)
FIXED_DATE = "2018-07-01"
PROBE_TRIPLETS = [
    ("this goes hard", "this is hard", "this absolutely slaps", "the weather is sunny"),
    ("I am dead from laughing", "I'm dead lol", "this joke killed me", "I have a medical appointment"),
    ("Nobody asked", "literally nobody asked", "who requested this opinion?", "I need to ask a question"),
    ("Everything is fine while the world burns", "this is fine, everything is burning",
     "pretending everything is okay in a disaster", "the kitchen is clean"),
    ("My brain at 3am", "my brain at three in the morning",
     "late night thoughts keeping me awake", "I sleep at noon"),
    ("stonks are going up", "stonks to the moon",
     "my investments are finally making money", "the spaceship reached the moon"),
    ("the joke went over his head", "that joke flew over his head",
     "he missed the punchline entirely", "the bird flew overhead"),
    ("me pretending to work", "me acting like I am working",
     "looking busy without doing anything", "the office opens at nine"),
    ("big chungus is back", "the big chungus returns",
     "the oversized rabbit meme has returned", "a rabbit escaped the yard"),
    ("I can't stop doomscrolling", "doomscrolling again",
     "I keep reading bad news on my phone", "the news anchor stopped speaking"),
]


def extract(model, manifest, examples, encoder):
    """Capture content-only states from four blocks and the final layer norm."""
    snapshots = {}
    hooks = []
    for layer in LAYERS:
        name = f"block_{layer + 1:02d}"
        hooks.append(model.transformer.h[layer].register_forward_hook(
            lambda module, inputs, output, key=name: snapshots.__setitem__(key, output)))
    hooks.append(model.transformer.ln_f.register_forward_hook(
        lambda module, inputs, output: snapshots.__setitem__("final_ln", output)))
    features = {f"{condition}_{name}_{pool}": []
                for condition in ("fixed", "real")
                for name in [f"block_{layer + 1:02d}" for layer in LAYERS] + ["final_ln"]
                for pool in ("mean", "last")}
    losses = {"fixed": [], "real": []}
    fixed_tau = normalized_time(parse_date(FIXED_DATE), manifest)
    try:
        for index, row in enumerate(examples):
            # The input contains no subreddit marker. Position zero is the time pseudo-token;
            # pooling starts at position one, so it never enters the text vector directly.
            ids = encoder.encode(row["text"])[:128]
            if len(ids) < 2:
                raise ValueError(f"Example {index} has fewer than two text tokens")
            tokens = torch.tensor(ids, dtype=torch.long, device="cuda")[None, :]
            real_tau = normalized_time(row["created_utc"], manifest)
            for condition, tau in (("fixed", fixed_tau), ("real", real_tau)):
                snapshots.clear()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(tokens, time_value=torch.tensor([tau], device="cuda"))
                loss = F.cross_entropy(logits[0, :-1].float(), tokens[0, 1:]).item()
                losses[condition].append(loss)
                for name, output in snapshots.items():
                    content = output[0, 1:, :].float()
                    features[f"{condition}_{name}_mean"].append(content.mean(0).cpu().numpy())
                    features[f"{condition}_{name}_last"].append(content[-1].cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()
    vectors = {name: F.normalize(torch.from_numpy(np.stack(rows)), dim=1).numpy()
               for name, rows in features.items()}
    return vectors, losses, fixed_tau


def choose_edges(similarity, lexical, examples, method):
    """Rank only chronologically earlier candidates, using prespecified scores."""
    links = []
    dates = np.array([row["created_utc"] for row in examples], dtype=np.int64)
    for i in range(1, len(examples)):
        earlier = np.flatnonzero(dates[:i] < dates[i])
        if not len(earlier):
            continue
        gaps = (dates[i] - dates[earlier]) / 31557600
        sem = similarity[i, earlier]
        lex = lexical[i, earlier]
        if method == "semantic":
            scores = sem
        elif method == "semantic_time":
            scores = sem - 0.03 * gaps
        elif method == "lexical":
            scores = lex
        elif method == "hybrid":
            # Normalize semantic scores within the earlier candidate set before mixing.
            scaled = (sem - sem.min()) / max(1e-8, sem.max() - sem.min())
            scores = 0.5 * scaled + 0.5 * lex - 0.03 * gaps
        elif method == "lexical_gate20":
            # Lexical retrieval limits the search; the model then reranks its top 20.
            nearest = np.argsort(lex)[-20:]
            earlier, gaps, sem, lex = earlier[nearest], gaps[nearest], sem[nearest], lex[nearest]
            scores = sem - 0.03 * gaps
        else:
            raise ValueError(method)
        best = int(np.argmax(scores))
        links.append({"child": i, "parent": int(earlier[best]),
                      "cosine": float(sem[best]), "lexical": float(lex[best]),
                      "gap_years": float(gaps[best])})
    return links


def metrics(edges, examples):
    """Keep overall and later-year scores separate to expose selection bias."""
    def group(rows):
        if not rows:
            return {"edges": 0}
        return {"edges": len(rows),
                "same_family": float(np.mean([
                    examples[e["child"]]["family"] == examples[e["parent"]]["family"]
                    for e in rows])),
                "mean_gap_years": float(np.mean([e["gap_years"] for e in rows])),
                "cross_subreddit": float(np.mean([
                    examples[e["child"]]["subreddit"] != examples[e["parent"]]["subreddit"]
                    for e in rows]))}
    return {"all": group(edges),
            "through_2020": group([e for e in edges if examples[e["child"]]["year"] <= 2020]),
            "2021_2022": group([e for e in edges if examples[e["child"]]["year"] >= 2021])}


def inspect_links(edges, examples, limit=30):
    """Surface low-overlap cross-family links for human semantic review."""
    chosen = sorted([edge for edge in edges
                     if examples[edge["child"]]["family"] != examples[edge["parent"]]["family"]
                     and edge["lexical"] < 0.12
                     and len(examples[edge["child"]]["text"]) >= 15
                     and len(examples[edge["parent"]]["text"]) >= 15],
                    key=lambda edge: edge["cosine"], reverse=True)[:limit]
    return [{"child": {key: examples[e["child"]][key] for key in
                        ("family", "year", "subreddit", "text")},
             "parent": {key: examples[e["parent"]][key] for key in
                         ("family", "year", "subreddit", "text")},
             "cosine": e["cosine"], "lexical": e["lexical"],
             "gap_years": e["gap_years"]} for e in chosen]


def semantic_probe(model, manifest, encoder):
    """Use a small authored literal/paraphrase/unrelated diagnostic, not lineage ground truth."""
    stamp = parse_date(FIXED_DATE)
    examples = [{"text": text, "created_utc": stamp}
                for group in PROBE_TRIPLETS for text in group]
    vectors, _, _ = extract(model, manifest, examples, encoder)
    lexical = cosine_similarity(TfidfVectorizer(ngram_range=(1, 2)).fit_transform(
        [row["text"] for row in examples])).astype(np.float32)
    report = {"design": "10 hand-authored anchor/literal/paraphrase/unrelated triplets; exploratory, not real genealogy labels",
              "methods": {}}
    for name, similarities in [("tfidf", lexical)] + [
        (key, value @ value.T) for key, value in vectors.items()
        if key in ("fixed_block_07_mean", "fixed_block_10_mean", "fixed_final_ln_mean")]:
        rows = []
        for i, group in enumerate(PROBE_TRIPLETS):
            anchor = i * 4
            rows.append({"anchor": group[0], "literal": group[1],
                         "paraphrase": group[2], "unrelated": group[3],
                         "literal_score": float(similarities[anchor, anchor + 1]),
                         "paraphrase_score": float(similarities[anchor, anchor + 2]),
                         "unrelated_score": float(similarities[anchor, anchor + 3])})
        report["methods"][name] = {"paraphrase_beats_unrelated": sum(
            row["paraphrase_score"] > row["unrelated_score"] for row in rows),
            "literal_beats_unrelated": sum(
                row["literal_score"] > row["unrelated_score"] for row in rows),
            "rows": rows}
    return report


def run(root: Path):
    """Evaluate all representations on the unchanged 485-post phrase sample."""
    examples = json.loads((root / "analysis/phrase_examples.json").read_text())
    model, manifest, checkpoint = load_model(root / "checkpoints/tokens_1000M.pt",
                                              root / "tokenized/manifest.json", "cuda")
    encoder = tiktoken.get_encoding("gpt2")
    vectors, losses, fixed_tau = extract(model, manifest, examples, encoder)
    lexical = cosine_similarity(TfidfVectorizer(ngram_range=(1, 2)).fit_transform(
        [row["text"] for row in examples])).astype(np.float32)
    result = {"checkpoint_tokens_seen": checkpoint["tokens_seen"],
              "examples": len(examples), "fixed_date": FIXED_DATE,
              "fixed_tau": fixed_tau, "layers": [layer + 1 for layer in LAYERS],
              "pooling": "Mean of content token states, excluding time pseudo-token; no subreddit marker in extraction input.",
              "trajectory": {"mean_real_nll": float(np.mean(losses["real"])),
                             "mean_fixed_nll": float(np.mean(losses["fixed"])),
                             "by_family": {}}, "variants": {}}
    for family in sorted(set(row["family"] for row in examples)):
        indices = [i for i, row in enumerate(examples) if row["family"] == family]
        result["trajectory"]["by_family"][family] = {
            "n": len(indices),
            "mean_fixed_minus_real_nll": float(np.mean([
                losses["fixed"][i] - losses["real"][i] for i in indices]))}
    lexical_edges = choose_edges(lexical, lexical, examples, "lexical")
    result["variants"]["tfidf"] = {"lexical": metrics(lexical_edges, examples)}
    result["comparison_edges"] = {"tfidf": lexical_edges}
    for name, vector in vectors.items():
        similarity = (vector @ vector.T).astype(np.float32)
        result["variants"][name] = {}
        for method in ("semantic", "semantic_time", "hybrid", "lexical_gate20"):
            edges = choose_edges(similarity, lexical, examples, method)
            result["variants"][name][method] = metrics(edges, examples)
        if name in ("fixed_block_07_mean", "fixed_block_10_mean", "fixed_final_ln_mean"):
            edges = choose_edges(similarity, lexical, examples, "semantic_time")
            result.setdefault("manual_review", {})[name] = inspect_links(edges, examples)
        if name == "fixed_block_07_mean":
            result["illustrative_edges"] = {
                method: choose_edges(similarity, lexical, examples, method)
                for method in ("semantic_time", "hybrid", "lexical_gate20")}
        if name in ("fixed_final_ln_mean", "fixed_block_07_mean", "fixed_block_10_mean"):
            result["comparison_edges"][name] = choose_edges(
                similarity, lexical, examples, "lexical_gate20")
    result["semantic_probe"] = semantic_probe(model, manifest, encoder)
    return result

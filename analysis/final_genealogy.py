"""Clean and export a portable model-derived genealogy graph."""

from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

import networkx as nx
import numpy as np

from broad_genealogy import build_graph


URL = re.compile(r"https?://|www\.|(?:reddit|imgur)\.com/", re.I)
BOILERPLATE = re.compile(
    r"i am a bot|this action was performed automatically|your (?:post|submission) has been removed|"
    r"please contact the moderators|click here to|subscribe to my|buy (?:cheap|now)|"
    r"search engine optimization|home improvement tips|car buying (?:tips|suggestions)|"
    r"web hosting on the cheap|health benefits of|heart healthy mix|"
    r"(?:visit|check out) my (?:website|store|channel)", re.I)
WORDS = re.compile(r"[\w']+", re.UNICODE)
# Names were assigned after inspecting the model's components; they never
# participate in embedding, neighbor selection, or thresholding.
DISPLAY = {
    0: ("Oof / ouch / owie", "pain_exclamations"),
    2: ("Starter packs", "starter_packs"),
    3: ("Le ... has arrived I", "dogelore_arrivals"),
    7: ("Le ... has arrived II", "dogelore_arrivals"),
    9: ("It do be like that", "it_do_be_like_that"),
    10: ("Invest in this I", "meme_investment"),
    11: ("Invest in this II", "meme_investment"),
    18: ("Le ... has arrived III", "dogelore_arrivals"),
    20: ("Invest in this III", "meme_investment"),
    32: ("BUY BUY BUY", "meme_investment"),
    39: ("Most Interesting Man", "most_interesting_man"),
    45: ("Versatile formats", "meme_investment"),
    60: ("New format, invest", "meme_investment"),
    63: ("Thank you, very cool", "thank_you_very_cool"),
    64: ("My bones", "my_bones"),
    65: ("Le ... has arrived IV", "dogelore_arrivals"),
    79: ("Surgery on a grape", "surgery_on_a_grape"),
}


def rejection_reason(row):
    """Remove obvious non-meme text and encoding artifacts conservatively."""
    text = row["text"]
    if URL.search(text):
        return "url"
    if BOILERPLATE.search(text):
        return "bot_or_spam"
    if sum(unicodedata.category(c) == "Mn" for c in text) / len(text) > 0.03:
        return "combining_mark_artifact"
    words = [word.casefold() for word in WORDS.findall(text)]
    if len(words) >= 8 and Counter(words).most_common(1)[0][1] / len(words) >= 0.5:
        return "single_word_repetition"
    return None


def clean_rows(rows, vectors):
    """Keep first dated copy after punctuation-only normalization."""
    kept_rows, kept_vectors = [], []
    rejected = Counter()
    canonical_seen = set()
    for i, row in enumerate(rows):
        reason = rejection_reason(row)
        if reason:
            rejected[reason] += 1
            continue
        canonical = " ".join(WORDS.findall(row["text"].casefold()))
        if len(canonical) >= 12 and canonical in canonical_seen:
            rejected["punctuation_near_copy"] += 1
            continue
        canonical_seen.add(canonical)
        kept_rows.append(row)
        kept_vectors.append(vectors[i])
    return kept_rows, np.stack(kept_vectors), dict(rejected)


def components(rows, edges):
    """Return dated component memberships and summaries for manual review."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(rows)))
    graph.add_edges_from((edge["child"], edge["parent"]) for edge in edges)
    children = Counter(edge["parent"] for edge in edges)
    groups = []
    for group in sorted(nx.connected_components(graph), key=len, reverse=True):
        if len(group) < 3:
            break
        ordered = sorted(group, key=lambda i: rows[i]["created_utc"])
        hubs = sorted(group, key=lambda i: children[i], reverse=True)[:3]
        groups.append({"index": len(groups), "size": len(group), "members": sorted(group),
                       "years": [rows[ordered[0]]["year"], rows[ordered[-1]]["year"]],
                       "subreddits": dict(Counter(rows[i]["subreddit"] for i in group)),
                       "early": [rows[i]["text"][:160] for i in ordered[:3]],
                       "hubs": [{"text": rows[i]["text"][:160], "children": children[i]}
                                for i in hubs]})
    return groups


def build(root: Path):
    """Recompute neighbors after filtering so removed artifacts cannot be parents."""
    source = root / "analysis/broad_genealogy"
    rows = json.loads((source / "sample.json").read_text())
    source_count = len(rows)
    vectors = np.load(source / "embeddings_block10_fixed2018.npy")
    rows, vectors, rejected = clean_rows(rows, vectors)
    edges, selection = build_graph(rows, vectors)
    groups = components(rows, edges)
    output = source / "final"
    output.mkdir(exist_ok=True)
    (output / "rows.json").write_text(json.dumps(rows, ensure_ascii=False))
    (output / "edges.json").write_text(json.dumps(edges))
    report = {"source_nodes": source_count, "clean_nodes": len(rows), "clean_edges": len(edges),
              "rejected": rejected, "selection": selection,
              "component_count": len(groups), "components": groups}
    (output / "candidates.json").write_text(json.dumps(report, ensure_ascii=False))
    # The local preview has only labels and small samples, never the full corpus.
    preview = {**report, "components": [
        {key: value for key, value in group.items() if key != "members"}
        for group in groups[:100]]}
    return preview


def export(root: Path, selected_indices: list[int]):
    """Export reviewed components in Cytoscape/D3-friendly nodes and links."""
    source = root / "analysis/broad_genealogy/final"
    rows = json.loads((source / "rows.json").read_text())
    edges = json.loads((source / "edges.json").read_text())
    report = json.loads((source / "candidates.json").read_text())
    chosen = [report["components"][i] for i in selected_indices]
    ids = {i for group in chosen for i in group["members"]}
    component_by_node = {i: group["index"] for group in chosen for i in group["members"]}
    nodes = [{"id": rows[i]["id"], "label": rows[i]["text"].splitlines()[0][:120],
              "text": rows[i]["text"], "created_utc": rows[i]["created_utc"],
              "year": rows[i]["year"], "subreddit": rows[i]["subreddit"],
              "component": component_by_node[i],
              "motif": DISPLAY[component_by_node[i]][1]} for i in sorted(ids)]
    links = [{"source": rows[e["parent"]]["id"], "target": rows[e["child"]]["id"],
              "cosine": e["cosine"], "gap_years": e["gap_years"], "score": e["score"]}
             for e in edges if e["parent"] in ids and e["child"] in ids]
    graph = {"format": "zeitgeistlm-genealogy-v1", "directed": True,
             "description": "Earlier-to-later similarity links, not verified causal descent",
             "checkpoint_tokens_seen": 1000079360, "embedding": "block-10 mean at 2018-07-01",
             "selection": report["selection"],
             "curation": "Components selected and named after model-derived graph construction; no labels used in link selection",
             "components": [
                 {**{key: value for key, value in group.items() if key != "members"},
                  "name": DISPLAY[group["index"]][0], "motif": DISPLAY[group["index"]][1]}
                 for group in chosen], "nodes": nodes, "links": links}
    (source / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2))
    return graph

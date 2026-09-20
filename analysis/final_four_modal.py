"""Rerun the four paper experiments on the completed 2.967B checkpoint."""

import json
from pathlib import Path

import modal


CHECKPOINT = "final_2967M.pt"
TAG = "final_2967M"
app = modal.App("zeitgeistlm-final-four")
volume = modal.Volume.from_name("zeitgeistlm-data")
source = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch>=2.5,<3", "numpy>=2,<3", "tiktoken>=0.9,<1",
                         "networkx>=3.3,<4", "scipy>=1.14,<2")
         .add_local_dir(source / "model", remote_path="/workspace/model",
                        ignore=["*.ipynb", "*.txt", "__pycache__"])
         .add_local_dir(source / "analysis", remote_path="/workspace/analysis",
                        ignore=["*.json", "*.csv", "*.png", "__pycache__"]))


def setup():
    """Load analysis modules from the mounted source without copying data locally."""
    import sys
    sys.path[:0] = ["/workspace/model", "/workspace/analysis"]
    volume.reload()
    return Path("/data")


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=7200)
def likelihood_sweep() -> dict:
    """Score all 480 fixed held-out windows on the original 13-year grid."""
    root = setup()
    from temporal_sweep import run
    return run(root, CHECKPOINT)


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=7200)
def emergence() -> dict:
    """Rescore the prespecified phrases and reuse unchanged corpus counts."""
    root = setup()
    from emergence_study import score, summarize
    score(root, CHECKPOINT, f"scores_{TAG}.json")
    volume.commit()
    return summarize(root, f"scores_{TAG}.json")


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=14400)
def genealogy() -> dict:
    """Re-embed the full broad sample, remove artifacts, and rebuild links."""
    import numpy as np
    root = setup()
    from broad_genealogy import embed, build_graph
    from final_genealogy import clean_rows, components
    from zeitgeist_inference import load_model

    rows = json.loads((root / "analysis/broad_genealogy/sample.json").read_text())
    model, manifest, checkpoint = load_model(root / "checkpoints" / CHECKPOINT,
                                              root / "tokenized/manifest.json", "cuda")
    vectors = embed(model, manifest, rows)
    rows, vectors, rejected = clean_rows(rows, vectors)
    edges, selection = build_graph(rows, vectors)
    groups = components(rows, edges)
    output = root / "analysis" / TAG / "genealogy"
    output.mkdir(parents=True, exist_ok=True)
    # The full graph stays remote; local reports contain only small previews.
    np.save(output / "embeddings_block10_fixed2018.npy", vectors)
    nodes_in_graph = {i for edge in edges for i in (edge["child"], edge["parent"])}
    node_data = [{"id": rows[i]["id"], "text": rows[i]["text"],
                  "created_utc": rows[i]["created_utc"], "year": rows[i]["year"],
                  "subreddit": rows[i]["subreddit"]} for i in sorted(nodes_in_graph)]
    links = [{"source": rows[e["parent"]]["id"],
              "target": rows[e["child"]]["id"], "cosine": e["cosine"],
              "gap_years": e["gap_years"], "score": e["score"]} for e in edges]
    graph = {"format": "zeitgeistlm-genealogy-v1", "directed": True,
             "description": "Chronological similarity, not observed causal descent",
             "checkpoint_tokens_seen": checkpoint["tokens_seen"],
             "embedding": "block-10 mean at fixed 2018-07-01",
             "selection": selection, "nodes": node_data, "links": links}
    (output / "graph.json").write_text(json.dumps(graph, ensure_ascii=False))
    report = {"checkpoint_tokens_seen": checkpoint["tokens_seen"],
              "sampled_nodes": len(json.loads((root / "analysis/broad_genealogy/sample.json").read_text())),
              "clean_nodes": len(rows), "linked_nodes": len(nodes_in_graph),
              "edges": len(edges), "rejected": rejected, "selection": selection,
              "components_size_at_least_3": len(groups),
              "largest_component_sizes": [g["size"] for g in groups[:20]],
              "cross_subreddit_edge_rate": sum(
                  rows[e["child"]]["subreddit"] != rows[e["parent"]]["subreddit"]
                  for e in edges) / max(1, len(edges)),
              "top_components": [{key: value for key, value in group.items()
                                  if key != "members"} for group in groups[:30]],
              "graph_volume_path": str(output / "graph.json")}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    volume.commit()
    return report


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=7200)
def mutation() -> dict:
    """Generate fixed-seed continuations for a declared panel of meme prompts."""
    import tiktoken
    root = setup()
    from experiment import generation_and_sensitivity
    from zeitgeist_inference import load_model

    model, manifest, checkpoint = load_model(root / "checkpoints" / CHECKPOINT,
                                              root / "tokenized/manifest.json", "cuda")
    prompts = ["bro really thought", "when you realize", "nobody:",
               "me when", "this is fine", "stonks", "le doge has arrived",
               "it's free real estate", "sir this is a Wendy's",
               "we live in a society", "the meme economy", "that feeling when"]
    dates = ["2016", "2020", "2022", "2023"]
    samples, sensitivity = generation_and_sensitivity(
        model, manifest, tiktoken.get_encoding("gpt2"), "cuda",
        prompts=prompts, dates=dates, new_tokens=64)
    for sample in samples:
        # Separate the first document from any subsequent EOS-delimited post.
        sample["first_document"] = sample["continuation"].split("<|endoftext|>", 1)[0]
    return {"checkpoint_tokens_seen": checkpoint["tokens_seen"],
            "settings": {"seed": 42, "top_k": 50, "temperature": 0.8,
                         "new_tokens": 64, "prompts": prompts, "dates": dates},
            "samples": samples, "sensitivity": sensitivity}


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=7200)
def mutation_constrained() -> dict:
    """Repeat the prompt panel with a minimum first-document length."""
    import tiktoken
    root = setup()
    from generate_zeitgeist import generate
    from zeitgeist_inference import load_model, normalized_time, parse_date

    model, manifest, checkpoint = load_model(root / "checkpoints" / CHECKPOINT,
                                              root / "tokenized/manifest.json", "cuda")
    encoder = tiktoken.get_encoding("gpt2")
    prompts = ["bro really thought", "when you realize", "nobody:",
               "me when", "this is fine", "stonks", "le doge has arrived",
               "it's free real estate", "sir this is a Wendy's",
               "we live in a society", "the meme economy", "that feeling when"]
    dates = ["2016", "2020", "2022", "2023"]
    samples = []
    for prompt in prompts:
        ids = encoder.encode(prompt)
        for date in dates:
            tau = normalized_time(parse_date(date), manifest)
            tokens = generate(model, ids, tau, 64, 50, 0.8, 42, "cuda",
                              encoder.n_vocab, min_document_tokens=16,
                              eot_token=encoder.eot_token)
            continuation = encoder.decode(tokens[len(ids):])
            samples.append({"prompt": prompt, "date": date,
                            "first_document": continuation.split("<|endoftext|>", 1)[0],
                            "continuation": continuation})
    return {"checkpoint_tokens_seen": checkpoint["tokens_seen"],
            "settings": {"seed": 42, "top_k": 50, "temperature": 0.8,
                         "new_tokens": 64, "min_document_tokens": 16,
                         "prompts": prompts, "dates": dates},
            "samples": samples}


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def curated_genealogy() -> dict:
    """Export reviewed motif components without noisy style or spam groups."""
    import networkx as nx

    root = setup()
    source_dir = root / "analysis" / TAG / "genealogy"
    graph = json.loads((source_dir / "graph.json").read_text())
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(node["id"] for node in graph["nodes"])
    nx_graph.add_edges_from((edge["source"], edge["target"]) for edge in graph["links"])
    ordered = sorted(nx.connected_components(nx_graph), key=len, reverse=True)
    summary = json.loads((source_dir / "summary.json").read_text())
    by_id = {node["id"]: node for node in graph["nodes"]}
    # Labels follow manual review of the final checkpoint's component previews.
    names = {0: ("Dogelore arrivals I", "dogelore_arrivals"),
             1: ("Oof / ouch / owie", "pain_exclamations"),
             3: ("Starter packs I", "starter_packs"),
             4: ("Dogelore arrivals II", "dogelore_arrivals"),
             7: ("Starter packs II", "starter_packs"),
             8: ("Invest in this I", "meme_investment"),
             9: ("Starter packs III", "starter_packs"),
             11: ("It do be like that", "it_do_be_like_that"),
             12: ("Invest in this II", "meme_investment"),
             13: ("Starter packs IV", "starter_packs"),
             14: ("Invest in this III", "meme_investment"),
             16: ("Starter packs V", "starter_packs"),
             22: ("Starter packs VI", "starter_packs"),
             23: ("Starter packs VII", "starter_packs"),
             24: ("BUY BUY BUY", "meme_investment"),
             25: ("Dogelore arrivals III", "dogelore_arrivals"),
             28: ("Invest in this IV", "meme_investment"),
             29: ("Most Interesting Man", "most_interesting_man")}
    for index in names:
        earliest = min((by_id[node_id] for node_id in ordered[index]),
                       key=lambda node: node["created_utc"])
        expected = summary["top_components"][index]["early"][0]
        if earliest["text"] != expected:
            raise ValueError(f"Component {index} no longer matches reviewed preview")
    membership = {node_id: index for index in names for node_id in ordered[index]}
    selected_nodes = [{**node, "component": membership[node["id"]],
                       "motif": names[membership[node["id"]]][1]}
                      for node in graph["nodes"] if node["id"] in membership]
    selected_links = [edge for edge in graph["links"]
                      if edge["source"] in membership and edge["target"] in membership]
    result = {**{key: value for key, value in graph.items()
                 if key not in {"nodes", "links"}},
              "curation": "Reviewed motif components, selected after link construction",
              "components": [{"index": index, "name": name, "motif": motif,
                              "size": len(ordered[index])}
                             for index, (name, motif) in names.items()],
              "nodes": selected_nodes, "links": selected_links}
    (source_dir / "curated_graph.json").write_text(json.dumps(result, ensure_ascii=False))
    volume.commit()
    return result


@app.local_entrypoint()
def main(phase: str = "all") -> None:
    """Write compact reports locally while retaining embeddings and graph remotely."""
    stages = {"sweep": likelihood_sweep, "emergence": emergence,
              "genealogy": genealogy, "mutation": mutation,
              "mutation_constrained": mutation_constrained,
              "curated": curated_genealogy}
    if phase != "all" and phase not in stages:
        raise ValueError(f"phase must be all or one of {list(stages)}")
    for name, function in stages.items():
        if phase != "all" and phase != name:
            continue
        result = function.remote()
        target = Path(__file__).with_name(f"{TAG}_{name}.json")
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"Saved {target}", flush=True)

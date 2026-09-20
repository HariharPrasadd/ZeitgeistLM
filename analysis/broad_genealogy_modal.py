"""Modal CPU collection and GPU graph construction for broad meme genealogy."""

import json
from pathlib import Path

import modal


app = modal.App("zeitgeistlm-broad-genealogy")
volume = modal.Volume.from_name("zeitgeistlm-data")
source = Path(__file__).resolve().parents[1]
cpu_image = (modal.Image.debian_slim(python_version="3.11")
             .uv_pip_install("numpy>=2,<3", "pyarrow>=20,<23", "networkx>=3.3,<4")
             .add_local_dir(source / "analysis", remote_path="/workspace/analysis",
                            ignore=["*.json", "*.csv", "*.png", "__pycache__"]))
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .uv_pip_install("torch>=2.5,<3", "numpy>=2,<3", "tiktoken>=0.9,<1",
                             "networkx>=3.3,<4")
             .add_local_dir(source / "model", remote_path="/workspace/model",
                            ignore=["*.ipynb", "*.txt", "__pycache__"])
             .add_local_dir(source / "analysis", remote_path="/workspace/analysis",
                            ignore=["*.json", "*.csv", "*.png", "__pycache__"]))


@app.function(image=cpu_image, cpu=4, memory=8192, volumes={"/data": volume},
              timeout=14400)
def collect_year(year: int) -> dict:
    """Scan one year on CPU and persist only its small stratified sample."""
    import sys
    sys.path.insert(0, "/workspace/analysis")
    from broad_genealogy import sample_year

    volume.reload()
    cached = Path(f"/data/analysis/broad_genealogy/sample_{year}.json")
    if cached.exists():
        # Annual files let an interrupted collection resume without rescanning.
        prior = json.loads(cached.read_text())
        return {"year": year, "rows": len(prior["rows"]),
                "months": len(prior["months"]), "cached": True}
    result = sample_year(Path("/data/cleaned"), year)
    output = Path("/data/analysis/broad_genealogy")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"sample_{year}.json").write_text(json.dumps(result, ensure_ascii=False))
    volume.commit()
    return {"year": year, "rows": len(result["rows"]), "months": len(result["months"])}


@app.function(image=cpu_image, cpu=2, volumes={"/data": volume}, timeout=1800)
def finalize() -> dict:
    """Deduplicate annual samples and keep the complete sample on the Volume."""
    import sys
    sys.path.insert(0, "/workspace/analysis")
    from broad_genealogy import assemble

    volume.reload()
    result = assemble(Path("/data"), range(2011, 2023))
    volume.commit()
    return result


@app.function(image=gpu_image, gpu="A100-80GB", volumes={"/data": volume},
              timeout=14400)
def graph() -> dict:
    """Extract representations and build exact chronological neighbors on GPU."""
    import sys
    sys.path.insert(0, "/workspace/model")
    sys.path.insert(0, "/workspace/analysis")
    from broad_genealogy import run

    volume.reload()
    summary = run(Path("/data"))
    volume.commit()
    return summary


@app.function(image=cpu_image, cpu=2, volumes={"/data": volume}, timeout=1800)
def refine() -> dict:
    """Tighten edge selection using already saved graph files on the Volume."""
    import sys
    sys.path.insert(0, "/workspace/analysis")
    from broad_genealogy import refine_saved

    volume.reload()
    summary = refine_saved(Path("/data"))
    volume.commit()
    return summary


@app.local_entrypoint()
def main(phase: str = "all") -> None:
    """Run CPU sampling and GPU analysis while returning only short reports locally."""
    if phase not in {"all", "collect", "graph", "refine"}:
        raise ValueError("phase must be all, collect, graph, or refine")
    if phase in {"all", "collect"}:
        years = list(collect_year.map(range(2011, 2023)))
        print("Annual samples:", years)
        sample = finalize.remote()
        target = Path(__file__).parent / "broad_genealogy_sample_report.json"
        target.write_text(json.dumps(sample, indent=2) + "\n")
        print(f"Saved {target}: {sample['sampled_after_dedup']} posts")
    if phase in {"all", "graph"}:
        summary = graph.remote()
        target = Path(__file__).parent / "broad_genealogy_summary.json"
        target.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {target}: {summary['nodes']} nodes, {summary['edges']} edges")
    if phase == "refine":
        summary = refine.remote()
        target = Path(__file__).parent / "broad_genealogy_summary.json"
        target.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {target}: {summary['nodes']} nodes, {summary['edges']} edges")

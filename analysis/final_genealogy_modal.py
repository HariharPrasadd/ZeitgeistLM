"""Prepare and export the compact public genealogy graph from saved Modal data."""

import json
from pathlib import Path

import modal


app = modal.App("zeitgeistlm-final-genealogy")
volume = modal.Volume.from_name("zeitgeistlm-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("numpy>=2,<3", "networkx>=3.3,<4", "torch>=2.5,<3")
         .add_local_dir(Path(__file__).resolve().parent,
                        remote_path="/workspace/analysis",
                        ignore=["*.json", "*.csv", "*.png", "__pycache__"]))


@app.function(image=image, gpu="A100", volumes={"/data": volume}, timeout=3600)
def prepare() -> dict:
    """Clean posts, recompute links, and return small component previews."""
    import sys
    sys.path.insert(0, "/workspace/analysis")
    from final_genealogy import build

    volume.reload()
    preview = build(Path("/data"))
    volume.commit()
    return preview


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def export_components(indices: list[int]) -> dict:
    """Return a reviewed, portable subgraph; keep the complete graph remote."""
    import sys
    sys.path.insert(0, "/workspace/analysis")
    from final_genealogy import export

    volume.reload()
    graph = export(Path("/data"), indices)
    volume.commit()
    return graph


@app.local_entrypoint()
def main(phase: str = "prepare", indices: str = ""):
    """Save only compact previews and the selected final graph locally."""
    if phase == "prepare":
        result = prepare.remote()
        target = Path(__file__).parent / "final_genealogy_candidates.json"
    elif phase == "export":
        selected = [int(item) for item in indices.split(",") if item.strip()]
        result = export_components.remote(selected)
        target = Path(__file__).parent / "final_genealogy_graph.json"
    else:
        raise ValueError("phase must be prepare or export")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {target}")

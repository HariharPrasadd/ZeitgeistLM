"""Run a fixed-window temporal likelihood sweep on Modal GPU."""

import json
from pathlib import Path

import modal


app = modal.App("zeitgeistlm-likelihood-sweep")
volume = modal.Volume.from_name("zeitgeistlm-data")
source = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch>=2.5,<3", "numpy>=2,<3", "tiktoken>=0.9,<1")
         .add_local_dir(source / "model", remote_path="/workspace/model",
                        ignore=["*.ipynb", "*.txt", "__pycache__"])
         .add_local_dir(source / "analysis", remote_path="/workspace/analysis",
                        ignore=["*.json", "*.csv", "*.png", "__pycache__"]))


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=7200)
def sweep() -> dict:
    """Keep training and held-out tokens remote and return compact curves."""
    import sys
    sys.path[:0] = ["/workspace/model", "/workspace/analysis"]
    from temporal_sweep import run

    volume.reload()
    return run(Path("/data"))


@app.local_entrypoint()
def main() -> None:
    """Write the small curve table for plotting without rerunning inference."""
    result = sweep.remote()
    path = Path(__file__).parent / "temporal_sweep_results.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {path}: {result['windows']} windows")

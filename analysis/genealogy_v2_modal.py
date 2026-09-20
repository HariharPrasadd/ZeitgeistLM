"""Re-evaluate meme genealogy with pooled layer states and fixed-time controls."""

import json
from pathlib import Path

import modal


app = modal.App("zeitgeistlm-genealogy-v2")
volume = modal.Volume.from_name("zeitgeistlm-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch>=2.5,<3", "numpy>=2,<3", "tiktoken>=0.9,<1",
                    "scikit-learn>=1.5,<2")
    .add_local_dir(Path(__file__).resolve().parents[1] / "model", remote_path="/workspace/model",
                   ignore=["*.ipynb", "*.txt", "__pycache__"])
    .add_local_dir(Path(__file__).resolve().parent, remote_path="/workspace/analysis",
                   ignore=["*.json", "*.csv", "*.png", "*.svg", "__pycache__"])
)


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=14400)
def evaluate() -> dict:
    """Run all embedding variants without moving the checkpoint off Modal."""
    import sys
    sys.path.insert(0, "/workspace/model")
    sys.path.insert(0, "/workspace/analysis")
    from genealogy_v2 import run

    volume.reload()
    return run(Path("/data"))


@app.local_entrypoint()
def main():
    """Keep only compact analysis results in the repository."""
    report = evaluate.remote()
    target = Path(__file__).resolve().parent / "genealogy_v2_results.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {target}")

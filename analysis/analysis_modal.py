"""Run reproducible ZeitgeistLM analyses against data stored on Modal."""

import json
from pathlib import Path

import modal


app = modal.App("zeitgeistlm-analysis")
volume = modal.Volume.from_name("zeitgeistlm-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch>=2.5,<3", "numpy>=2,<3", "tiktoken>=0.9,<1",
                    "pyarrow>=20,<23")
    .add_local_dir(Path(__file__).resolve().parents[1] / "model",
                   remote_path="/workspace/model", ignore=["*.ipynb", "*.txt", "__pycache__"])
    .add_local_dir(Path(__file__).resolve().parent, remote_path="/workspace/analysis",
                   ignore=["*.json", "*.csv", "*.png", "*.svg", "__pycache__"])
)


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=8192,
              timeout=14400)
def collect_phrases() -> dict:
    """Find bounded, dated examples in cleaned submissions for genealogy."""
    import sys
    sys.path.insert(0, "/workspace/model")
    sys.path.insert(0, "/workspace/analysis")
    from experiment import collect_phrase_examples

    volume.reload()
    examples, audit = collect_phrase_examples(Path("/data/cleaned"))
    target = Path("/data/analysis/phrase_examples.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(examples))
    volume.commit()
    return audit


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume},
              timeout=14400)
def run_experiments() -> dict:
    """Evaluate fixed checkpoints on GPU and return only compact results."""
    import sys
    sys.path.insert(0, "/workspace/model")
    sys.path.insert(0, "/workspace/analysis")
    from experiment import analyze

    volume.reload()
    return analyze(Path("/data"))


@app.local_entrypoint()
def main(phase: str = "collect"):
    """Save the report on the laptop while large shards remain on Modal."""
    if phase not in {"collect", "analyze"}:
        raise ValueError("phase must be collect or analyze")
    report = collect_phrases.remote() if phase == "collect" else run_experiments.remote()
    target = Path(__file__).resolve().parent / f"{phase}_results.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {target}")
    if phase == "collect":
        print(json.dumps(report, indent=2))

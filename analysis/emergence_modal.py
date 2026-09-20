"""Run pre-cutoff phrase selection, full-corpus counting, and model scoring."""

import json
from pathlib import Path

import modal


app = modal.App("zeitgeistlm-emergence-study")
volume = modal.Volume.from_name("zeitgeistlm-data")
source = Path(__file__).resolve().parents[1]
cpu_image = (modal.Image.debian_slim(python_version="3.11")
             .uv_pip_install("numpy>=2,<3", "scipy>=1.14,<2", "pyarrow>=20,<23",
                             "pyahocorasick>=2,<3")
             .add_local_dir(source / "analysis", remote_path="/workspace/analysis",
                            ignore=["*.json", "*.csv", "*.png", "__pycache__"]))
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .uv_pip_install("torch>=2.5,<3", "numpy>=2,<3", "tiktoken>=0.9,<1")
             .add_local_dir(source / "analysis", remote_path="/workspace/analysis",
                            ignore=["*.json", "*.csv", "*.png", "__pycache__"])
             .add_local_dir(source / "model", remote_path="/workspace/model",
                            ignore=["*.ipynb", "*.txt", "__pycache__"]))


@app.function(image=cpu_image, volumes={"/data": volume}, timeout=1800)
def select() -> dict:
    """Fix the candidate universe before reading later outcome data."""
    import sys
    sys.path.insert(0, "/workspace/analysis")
    from emergence_study import candidates
    volume.reload()
    result = candidates(Path("/data"))
    volume.commit()
    return result


@app.function(image=cpu_image, volumes={"/data": volume}, cpu=4, memory=8192,
              timeout=14400)
def count(year: int) -> dict:
    """Count one year's complete cleaned submission corpus on CPU."""
    import sys
    sys.path.insert(0, "/workspace/analysis")
    from emergence_study import count_year
    volume.reload()
    result = count_year(Path("/data"), year)
    volume.commit()
    return result


@app.function(image=gpu_image, gpu="A100-80GB", volumes={"/data": volume},
              timeout=7200)
def score() -> dict:
    """Score all four contexts per phrase at pre-cutoff dates."""
    import sys
    sys.path[:0] = ["/workspace/model", "/workspace/analysis"]
    from emergence_study import score as score_all
    volume.reload()
    result = score_all(Path("/data"))
    volume.commit()
    return result


@app.function(image=cpu_image, volumes={"/data": volume}, timeout=1800)
def report() -> dict:
    """Return compact phrase-level comparison after counts and scores exist."""
    import sys
    sys.path.insert(0, "/workspace/analysis")
    from emergence_study import summarize
    volume.reload()
    return summarize(Path("/data"))


@app.local_entrypoint()
def main(phase: str = "all") -> None:
    """Run the full protocol or resume after a completed stage."""
    if phase not in {"all", "select", "count", "score", "report"}:
        raise ValueError("phase must be all, select, count, score, or report")
    if phase in {"all", "select"}:
        print("Candidate selection:", select.remote(), flush=True)
    if phase in {"all", "count"}:
        print("Corpus counts:", list(count.map(range(2018, 2023))), flush=True)
    if phase in {"all", "score"}:
        print("Model scoring:", score.remote(), flush=True)
    if phase in {"all", "report"}:
        result = report.remote()
        target = Path(__file__).parent / "emergence_results.json"
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {target}: {result['n']} phrases", flush=True)

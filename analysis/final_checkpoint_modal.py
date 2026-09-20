"""Run a credit-bounded audit of the completed ZeitgeistLM checkpoint."""

import json
from pathlib import Path

import modal


app = modal.App("zeitgeistlm-final-audit")
volume = modal.Volume.from_name("zeitgeistlm-data")
source = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch>=2.5,<3", "numpy>=2,<3", "tiktoken>=0.9,<1")
         .add_local_dir(source / "model", remote_path="/workspace/model",
                        ignore=["*.ipynb", "*.txt", "__pycache__"])
         .add_local_dir(source / "analysis", remote_path="/workspace/analysis",
                        ignore=["*.json", "*.csv", "*.png", "__pycache__"]))


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=1800)
def audit(checkpoint_name: str = "final_2967M.pt") -> dict:
    """Reuse fixed held-out windows and phrase examples without retraining."""
    import sys
    import tiktoken
    import torch

    sys.path[:0] = ["/workspace/model", "/workspace/analysis"]
    from experiment import (generation_and_sensitivity, genealogy, geometry,
                            sample_windows, score_windows)
    from zeitgeist_inference import load_model

    volume.reload()
    root = Path("/data")
    if checkpoint_name not in {"final_2967M.pt", "tokens_1000M.pt"}:
        raise ValueError("Only the fixed comparison checkpoints are supported")
    model, manifest, checkpoint = load_model(
        root / "checkpoints" / checkpoint_name,
        root / "tokenized/manifest.json", "cuda")
    encoder = tiktoken.get_encoding("gpt2")
    report = {"checkpoint_name": checkpoint_name,
              "checkpoint_tokens_seen": checkpoint["tokens_seen"],
              "next_step": checkpoint["next_step"], "sample_windows_per_month_kind": 2}
    report["generation"], report["generation_sensitivity"] = (
        generation_and_sensitivity(model, manifest, encoder, "cuda"))
    report["geometry"] = geometry(model, manifest)
    report["forecast"] = {}
    for split in ("val", "test"):
        # Reuse deterministic, stratified held-out windows while limiting GPU time.
        windows = sample_windows(root / "tokenized", manifest, split, per_stratum=2)
        report["forecast"][split] = score_windows(model, manifest, windows, "cuda")
    examples = json.loads((root / "analysis/phrase_examples.json").read_text())
    report["genealogy"] = genealogy(model, manifest, encoder, examples, "cuda")
    del model
    torch.cuda.empty_cache()
    return report


@app.local_entrypoint()
def main(checkpoint_name: str = "final_2967M.pt") -> None:
    """Keep the detailed audit locally; leave model weights on the Volume."""
    result = audit.remote(checkpoint_name)
    output_name = ("final_checkpoint_results.json" if checkpoint_name == "final_2967M.pt"
                   else "comparison_1b_results.json")
    target = Path(__file__).with_name(output_name)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {target}: {result['checkpoint_tokens_seen']:,} tokens")

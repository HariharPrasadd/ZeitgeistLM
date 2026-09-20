"""Modal GPU entry point for the temporal GPT training run."""

from pathlib import Path
import subprocess
import sys

import modal


app = modal.App("zeitgeistlm-train")
volume = modal.Volume.from_name("zeitgeistlm-data")
launch_state = modal.Dict.from_name("zeitgeistlm-train-launch", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch>=2.5,<3", "numpy>=2,<3", "tiktoken>=0.9,<1")
    .add_local_dir(Path(__file__).parent / "model", remote_path="/workspace/model",
                   ignore=["*.ipynb", "*.txt", "__pycache__"])
)


@app.function(image=image, gpu="A100", timeout=300)
def gpu_smoke() -> dict:
    """Check CUDA and the temporal forward pass before the costly full run."""
    import os
    import torch

    os.chdir("/workspace/model")
    sys.path.insert(0, "/workspace/model")
    from train_gpt2 import GPT, GPTConfig

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see the Modal GPU")
    model = GPT(GPTConfig(block_size=8, vocab_size=32, n_layer=2,
                          n_head=2, n_embd=16)).cuda()
    tokens = torch.randint(0, 32, (2, 8), device="cuda")
    times = torch.tensor([0.0, 1.2], device="cuda")
    _, loss = model(tokens, tokens, times)
    loss.backward()
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "loss_finite": bool(torch.isfinite(loss).item())}


def _benchmark_training() -> dict:
    """Run the identical benchmark script inside either GPU function."""
    import json

    volume.reload()
    result = subprocess.run([sys.executable, "benchmark_zeitgeist.py"],
                            cwd="/workspace/model", check=True,
                            capture_output=True, text=True)
    return json.loads(result.stdout.splitlines()[-1])


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=3600)
def benchmark_a100() -> dict:
    """Benchmark the current training GPU choice."""
    return _benchmark_training()


@app.function(image=image, gpu="H100", volumes={"/data": volume}, timeout=3600)
def benchmark_h100() -> dict:
    """Benchmark Hopper using the same model and tokenized data."""
    return _benchmark_training()


@app.function(image=image, gpu="H100", volumes={"/data": volume},
              timeout=86400, retries=3)
def train() -> None:
    """Run one GPU process; the script resumes from its latest checkpoint."""
    import os

    if not Path("/data/tokenized/manifest.json").is_file():
        raise FileNotFoundError("Tokenized manifest is not ready on the Modal Volume")
    env = os.environ.copy()
    env["ZEITGEIST_DATA_ROOT"] = "/data/tokenized"
    env["ZEITGEIST_OUTPUT_ROOT"] = "/data/checkpoints"
    env["ZEITGEIST_ON_MODAL"] = "1"
    # Lower the resumed run's update size after sustained gradient-norm spikes.
    env["ZEITGEIST_LR_SCALE"] = "0.25"
    # A nonzero exit causes Modal to retry from the last committed checkpoint.
    subprocess.run([sys.executable, "train_zeitgeist.py"], cwd="/workspace/model",
                   env=env, check=True)
    volume.commit()


@app.function(volumes={"/data": volume}, schedule=modal.Period(minutes=5),
              max_containers=1, timeout=120)
def launch_when_ready() -> dict:
    """Start one training call after the complete corpus manifest is published."""
    import json

    volume.reload()
    manifest_path = Path("/data/tokenized/manifest.json")
    if not manifest_path.is_file():
        return {"status": "waiting_for_manifest"}
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("train_tokens") or any(
        not manifest.get("splits", {}).get(split) for split in ("train", "val", "test")
    ):
        raise ValueError("Final manifest needs nonempty train, val, and test splits")
    prior = launch_state.get("training")
    if prior:
        # The launch record prevents scheduled scans from starting duplicate runs.
        try:
            modal.FunctionCall.from_id(prior["call_id"]).get(timeout=0)
        except TimeoutError:
            return {"status": "running", **prior}
        except Exception as error:
            return {"status": "failed", "error": str(error), **prior}
        return {"status": "completed", **prior}
    call = train.spawn()
    record = {"call_id": call.object_id, "train_tokens": manifest["train_tokens"]}
    launch_state["training"] = record
    print(f"Started H100 training call {call.object_id}")
    return {"status": "started", **record}


@app.local_entrypoint()
def start() -> None:
    """Launch training only after tokenized shards and a manifest exist."""
    call = train.spawn()
    print(f"Training call: {call.object_id}")


@app.function(image=image, gpu="A10", volumes={"/data": volume}, timeout=3600)
def generate_demo(prompt: str, checkpoint_name: str = "latest.pt",
                  dates: tuple[str, ...] = ("2016", "2020", "2022")) -> dict:
    """Generate matched continuations at fixed dates from a saved checkpoint."""
    import json

    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("Checkpoint name must be a filename")
    volume.reload()
    command = [sys.executable, "generate_zeitgeist.py", "--checkpoint",
               f"/data/checkpoints/{checkpoint_name}", "--manifest",
               "/data/tokenized/manifest.json", "--prompt", prompt,
               "--dates", *dates]
    result = subprocess.run(command, cwd="/workspace/model", check=True,
                            capture_output=True, text=True)
    return json.loads(result.stdout)


@app.function(image=image, gpu="A100", volumes={"/data": volume}, timeout=86400)
def evaluate_checkpoint(checkpoint_name: str = "latest.pt", split: str = "val",
                        max_windows: int = 0, allow_test: bool = False) -> dict:
    """Score validation by default; future test requires explicit opt-in."""
    import json

    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("Checkpoint name must be a filename")
    if split == "test" and not allow_test:
        raise ValueError("Set allow_test=True only after model selection")
    if split not in {"val", "test"} or max_windows < 0:
        raise ValueError("Invalid split or max_windows")
    volume.reload()
    command = [sys.executable, "evaluate_zeitgeist.py", "--checkpoint",
               f"/data/checkpoints/{checkpoint_name}", "--manifest",
               "/data/tokenized/manifest.json", "--split", split,
               "--max-windows", str(max_windows)]
    if allow_test:
        command.append("--allow-test")
    result = subprocess.run(command, cwd="/workspace/model", check=True,
                            capture_output=True, text=True)
    return json.loads(result.stdout)

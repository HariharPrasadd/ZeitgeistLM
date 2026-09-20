"""Serve the final ZeitgeistLM checkpoint as a bounded public demo."""

from pathlib import Path

import modal


app = modal.App("zeitgeistlm-demo")
volume = modal.Volume.from_name("zeitgeistlm-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch>=2.5,<3", "numpy>=2,<3", "tiktoken>=0.9,<1", "fastapi>=0.115,<1")
    .add_local_dir(Path(__file__).resolve().parents[1] / "model", remote_path="/workspace/model",
                   ignore=["*.ipynb", "*.txt", "__pycache__"])
)


@app.cls(image=image, gpu="A10", volumes={"/data": volume},
         scaledown_window=120, max_containers=2, timeout=120)
class DemoModel:
    """Load the weights once per GPU container, then reuse them across requests."""

    @modal.enter()
    def load(self):
        import sys
        import tiktoken
        import torch
        sys.path.insert(0, "/workspace/model")
        from train_gpt2 import GPT

        volume.reload()
        checkpoint = torch.load("/data/checkpoints/final_2967M.pt",
                                map_location="cpu", weights_only=False)
        self.model = GPT(checkpoint["config"])
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.cuda().eval()
        self.encoder = tiktoken.get_encoding("gpt2")
        self.tokens_seen = int(checkpoint["tokens_seen"])
        self.bounds = tuple(checkpoint["time_bounds"])

    @modal.fastapi_endpoint(method="POST", label="zeitgeistlm-generate")
    def generate(self, payload: dict) -> dict:
        """Bound output and input sizes so the public demo cannot monopolize a GPU."""
        import sys
        import torch
        from fastapi import HTTPException
        sys.path.insert(0, "/workspace/model")
        from generate_zeitgeist import generate
        from zeitgeist_inference import parse_date

        prompt = str(payload.get("prompt", "")).strip()
        try:
            date_text = str(payload.get("date") or payload.get("year", "2020"))
            date_utc = parse_date(date_text)
            year = int(date_text[:4])
            count = int(payload.get("new_tokens", 60))
            seed = int(payload.get("seed", 42))
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid numeric parameter")
        if not prompt or len(prompt) > 160 or not 2008 <= year <= 2024 or not 1 <= count <= 64:
            raise HTTPException(400, "Prompt, date, or token count out of range")
        tokens = self.encoder.encode(prompt)
        if not 1 <= len(tokens) <= 64:
            raise HTTPException(400, "Prompt must contain 1–64 GPT-2 tokens")
        # Dates outside 2011–2020 extrapolate the learned affine time embedding.
        tau = (date_utc - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = generate(self.model, tokens, tau, count, 50, 0.8, seed,
                              "cuda", self.encoder.n_vocab,
                              min_document_tokens=12, eot_token=self.encoder.eot_token)
        continuation = result[len(tokens):]
        # Keep the live reader inside a single document even when EOS is sampled.
        if self.encoder.eot_token in continuation:
            continuation = continuation[:continuation.index(self.encoder.eot_token)]
        return {"prompt": prompt, "date": date_text, "year": year, "tau": tau,
                "continuation": self.encoder.decode(continuation),
                "checkpoint_tokens_seen": self.tokens_seen}

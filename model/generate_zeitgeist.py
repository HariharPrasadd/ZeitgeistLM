"""Compare one text seed at several continuous time coordinates."""

import argparse
import json
from pathlib import Path

import tiktoken
import torch
from torch.nn import functional as F

from zeitgeist_inference import load_model, normalized_time, parse_date


def generate(model, prompt_tokens: list[int], tau: float, new_tokens: int,
             top_k: int, temperature: float, seed: int, device: str,
             vocab_limit: int, min_document_tokens: int = 0,
             eot_token: int | None = None) -> list[int]:
    """Use the same sampling seed for a fair comparison between dates."""
    if not prompt_tokens or temperature <= 0 or top_k <= 0:
        raise ValueError("Prompt, temperature, and top-k must be positive")
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)[None, :]
    time_value = torch.tensor([tau], dtype=torch.float32, device=device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    with torch.inference_mode():
        for position in range(new_tokens):
            context = tokens[:, -model.config.block_size:]
            logits, _ = model(context, time_value=time_value)
            # The training matrix is padded to 50,304; only GPT-2 IDs decode.
            scores = logits[:, -1, :vocab_limit] / temperature
            # Optional demo constraint prevents a short prompt ending immediately.
            if position < min_document_tokens and eot_token is not None:
                scores[:, eot_token] = -float("inf")
            values, indices = torch.topk(scores, min(top_k, scores.size(-1)))
            probs = F.softmax(values, dim=-1)
            choice = torch.multinomial(probs, 1, generator=rng)
            next_token = indices.gather(1, choice)
            tokens = torch.cat((tokens, next_token), dim=1)
    return tokens[0].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--dates", nargs="+", default=["2016", "2020", "2022"])
    parser.add_argument("--new-tokens", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, manifest, checkpoint = load_model(args.checkpoint, args.manifest, device)
    encoder = tiktoken.get_encoding("gpt2")
    prompt_tokens = encoder.encode(args.prompt)
    results = []
    for date in args.dates:
        tau = normalized_time(parse_date(date), manifest)
        tokens = generate(model, prompt_tokens, tau, args.new_tokens,
                          args.top_k, args.temperature, args.seed, device,
                          encoder.n_vocab)
        results.append({"date": date, "tau": tau, "text": encoder.decode(tokens)})
    print(json.dumps({"checkpoint_tokens_seen": checkpoint.get("tokens_seen"),
                      "prompt": args.prompt, "samples": results}, indent=2))


if __name__ == "__main__":
    main()

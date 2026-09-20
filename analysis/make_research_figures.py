"""Create restrained, code-derived architecture and corpus figures for the paper."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
INK = "#111827"
GRAY = "#596579"
GRID = "#D9DEE7"
TIME = "#E5F3F1"
TOKEN = "#E9EFFA"
PROCESS = "#F4F5F7"
ATTN = "#FFF1DF"
FFN = "#E7F2FB"
OUTPUT = "#E8F3E9"


def label(ax, x, y, value, size=10, weight="normal", color=INK, ha="center"):
    ax.text(x, y, value, fontsize=size, weight=weight, color=color,
            ha=ha, va="center", linespacing=1.25)


def node(ax, x, y, w, h, title, fill=PROCESS, size=10):
    """A compact rectangular stage matching conventional architecture figures."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 linewidth=1.2, edgecolor=INK, facecolor=fill))
    label(ax, x + w / 2, y + h / 2, title, size)


def path(ax, points, dashed=False, head=True):
    """Draw orthogonal signal paths and residual branches."""
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=1.25,
                linestyle="--" if dashed else "-")
    a, b = points[-2:]
    if head:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                     mutation_scale=10, linewidth=1.25, color=INK,
                     linestyle="--" if dashed else "-"))
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=1.25,
                linestyle="--" if dashed else "-")


def architecture():
    """Map the implemented forward pass, including pre-LN residual blocks."""
    fig, ax = plt.subplots(figsize=(10, 15), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15.2)
    ax.axis("off")
    label(ax, 5, 14.92, "ZeitgeistLM: temporal decoder-only transformer", 16, "bold")
    label(ax, 5, 14.58, "124.5M parameters  ·  12 layers  ·  12 heads  ·  width 768", 10, color=GRAY)

    # Two input streams meet only after the date has been projected to width 768.
    node(ax, 0.65, 0.55, 3.55, 0.62, "created_utc  t", TIME)
    node(ax, 5.8, 0.55, 3.55, 0.62, "text  x_1, …, x_T", TOKEN)
    path(ax, [(2.425, 1.17), (2.425, 1.55)])
    path(ax, [(7.575, 1.17), (7.575, 1.55)])
    node(ax, 0.65, 1.55, 3.55, 0.70, "τ = (t − tmin) / (tmax − tmin)", TIME)
    node(ax, 5.8, 1.55, 3.55, 0.70, "GPT-2 BPE token IDs", TOKEN)
    path(ax, [(2.425, 2.25), (2.425, 2.63)])
    path(ax, [(7.575, 2.25), (7.575, 2.63)])
    node(ax, 0.65, 2.63, 3.55, 0.70, "eₜ = Linear(1, 768)(τ)", TIME)
    node(ax, 5.8, 2.63, 3.55, 0.70, "eᵢ = WTE(xᵢ)", TOKEN)
    path(ax, [(2.425, 3.33), (2.425, 3.63), (4.0, 3.63), (4.0, 3.82)])
    path(ax, [(7.575, 3.33), (7.575, 3.63), (6.0, 3.63), (6.0, 3.82)])
    node(ax, 2.62, 3.82, 4.76, 0.62, "prepend:  [ e_time , e_1 , … , e_T ]", PROCESS)
    path(ax, [(5, 4.44), (5, 4.77)])
    node(ax, 2.62, 4.77, 4.76, 0.62, "+ learned positional embeddings  p_0…p_T", PROCESS)
    path(ax, [(5, 5.39), (5, 5.72)])

    # One GPT block is expanded; its input bypasses each sublayer via addition.
    ax.add_patch(FancyBboxPatch((1.30, 5.72), 7.4, 5.12,
                 boxstyle="round,pad=0.02,rounding_size=0.18",
                 linewidth=1.5, edgecolor=INK, facecolor="#FBFCFD"))
    ax.text(0.97, 8.30, "Transformer block × 12", rotation=90,
            fontsize=10, weight="bold", color=INK, ha="center", va="center")
    label(ax, 5, 5.98, "h⁽ℓ⁾", 10)
    path(ax, [(5, 6.08), (5, 6.28)])
    node(ax, 3.26, 6.28, 3.48, 0.52, "LayerNorm", PROCESS)
    path(ax, [(5, 6.80), (5, 7.03)])
    node(ax, 3.26, 7.03, 3.48, 0.70, "12-head masked self-attention", ATTN)
    path(ax, [(5, 7.73), (5, 8.02)])
    node(ax, 4.74, 8.02, 0.52, 0.52, "+", "white", 14)
    path(ax, [(5, 6.08), (7.65, 6.08), (7.65, 8.28), (5.26, 8.28)])
    path(ax, [(5, 8.54), (5, 8.73)])
    node(ax, 3.26, 8.73, 3.48, 0.52, "LayerNorm", PROCESS)
    path(ax, [(5, 9.25), (5, 9.45)])
    node(ax, 3.26, 9.45, 3.48, 0.70, "MLP: Linear → GELU → Linear", FFN)
    path(ax, [(5, 10.15), (5, 10.35)])
    node(ax, 4.74, 10.35, 0.52, 0.52, "+", "white", 14)
    path(ax, [(5, 8.54), (2.30, 8.54), (2.30, 10.61), (4.74, 10.61)])
    path(ax, [(5, 10.87), (5, 11.18)])

    node(ax, 3.26, 11.18, 3.48, 0.58, "final LayerNorm", PROCESS)
    path(ax, [(5, 11.76), (5, 12.02)])
    node(ax, 2.62, 12.02, 4.76, 0.60, "select content positions  h_1…h_T", PROCESS)
    path(ax, [(5, 12.62), (5, 12.88)])
    node(ax, 3.26, 12.88, 3.48, 0.58, "tied LM head  WTEᵀ", OUTPUT)
    path(ax, [(5, 13.46), (5, 13.58)])
    node(ax, 3.26, 13.58, 3.48, 0.39, "softmax", OUTPUT, 9.5)
    path(ax, [(5, 13.97), (5, 14.10)])
    label(ax, 5, 14.23, "P(next token | context, t)", 11, "bold")

    # Intermediate hidden states are reused during analysis, without a new head.
    label(ax, 5, 0.24,
          "Training: next-token cross-entropy on text positions. Analysis: mean-pool block-10 token states for genealogy.",
          8.5, color=GRAY)
    fig.savefig(ROOT / "zeitgeistlm_model_diagram.png", dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(ROOT / "zeitgeistlm_model_diagram.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def corpus(manifest):
    """Plot observed annual token composition without decorative UI elements."""
    counts = manifest["stats"]["tokens_by_year_and_kind"]
    years = np.arange(2011, 2023)
    submissions = np.array([counts[str(y)]["submissions"] for y in years]) / 1e6
    comments = np.array([counts[str(y)]["comments"] for y in years]) / 1e6
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.19, top=0.82)
    ax.bar(years, comments, width=0.72, color="#A7B0BD", label="Comments")
    ax.bar(years, submissions, bottom=comments, width=0.72,
           color="#3C6FA8", label="Submissions")
    ax.set_xlim(2010.42, 2022.6)
    ax.set_ylim(0, 1110)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=8.5)
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_yticklabels(["0", "250", "500", "750", "1,000"], fontsize=9)
    ax.set_ylabel("Tokens (millions)", fontsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0, pad=5, colors=INK)
    ax.legend(ncol=2, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.12),
              fontsize=9)
    ax.axvline(2020.5, color=INK, linewidth=1, linestyle="--")
    ax.axvline(2021.5, color=INK, linewidth=1, linestyle="--")
    ax.text(2015.5, -121, "TRAIN  ·  2011–2020  ·  2.967B", ha="center", fontsize=9)
    ax.text(2021, -121, "VAL", ha="center", fontsize=8.5)
    ax.text(2022, -121, "TEST*", ha="center", fontsize=8.5)
    ax.text(2022, submissions[-1] + comments[-1] + 27, "Jan–Aug", ha="center",
            fontsize=8.5, color=GRAY)
    fig.suptitle("ZeitgeistLM corpus by year and document type", x=0.10,
                 y=0.965, ha="left", fontsize=14, weight="bold", color=INK)
    fig.text(0.10, 0.046,
             "*2022 contains available months only. Counts are tokenized GPT-2 BPE tokens after cleaning and packing.",
             fontsize=8.5, color=GRAY)
    fig.savefig(ROOT / "zeitgeistlm_corpus_timeline.png", dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(ROOT / "zeitgeistlm_corpus_timeline.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/tmp/zeitgeist-manifest.json"))
    arguments = parser.parse_args()
    architecture()
    corpus(json.loads(arguments.manifest.read_text()))
    print("Rendered technical architecture and corpus figures")

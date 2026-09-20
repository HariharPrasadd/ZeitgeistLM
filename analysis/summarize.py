"""Summarize fixed-checkpoint experiments and render reusable static figures."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "analyze_results.json").read_text())


def interval_by_month(rows, seed=7331):
    """Resample months to avoid treating adjacent text windows as independent."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["month"]].append(row["fixed_2020_loss"] - row["actual_time_loss"])
    months = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(10000):
        sample = rng.choice(months, size=len(months), replace=True)
        values = [value for month in sample for value in groups[month]]
        draws.append(np.mean(values))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def forecast_summary(rows):
    """Report paired loss and perplexity for one chronological split."""
    actual = np.array([row["actual_time_loss"] for row in rows])
    frozen = np.array([row["fixed_2020_loss"] for row in rows])
    return {"windows": len(rows), "tokens": len(rows) * 1024,
            "actual_time_loss": float(actual.mean()),
            "fixed_2020_loss": float(frozen.mean()),
            "loss_gain": float((frozen - actual).mean()),
            "month_bootstrap_95pct_gain": interval_by_month(rows),
            "actual_time_perplexity": math.exp(float(actual.mean())),
            "fixed_2020_perplexity": math.exp(float(frozen.mean())),
            "months": sorted(set(row["month"] for row in rows))}


def lexical_baseline(examples):
    """Check whether a simple word overlap graph already recovers family labels."""
    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(
        [row["text"] for row in examples])
    similarities = cosine_similarity(matrix)
    matches = []
    for i in range(1, len(examples)):
        candidates = [j for j in range(i)
                      if examples[j]["created_utc"] < examples[i]["created_utc"]]
        if candidates:
            parent = max(candidates, key=lambda j: similarities[i, j])
            matches.append(examples[i]["family"] == examples[parent]["family"])
    return float(np.mean(matches)), len(matches)


def plot_forecast():
    fig, (ax, delta) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    for split, color in (("val", "#3574b5"), ("test", "#d78442")):
        rows = DATA["forecast"][split]
        year = rows[0]["year"]
        months = sorted(set(row["month"] for row in rows))
        actual = [np.mean([r["actual_time_loss"] for r in rows if r["month"] == m])
                  for m in months]
        frozen = [np.mean([r["fixed_2020_loss"] for r in rows if r["month"] == m])
                  for m in months]
        xs = [year + (m - .5) / 12 for m in months]
        ax.plot(xs, actual, color=color, marker="o", label=f"{year} actual timestamp")
        ax.plot(xs, frozen, color=color, linestyle="--", alpha=.6,
                label=f"{year} fixed 2020 timestamp")
        delta.plot(xs, np.array(frozen) - np.array(actual), color=color, marker="o",
                   label=f"{year}")
    ax.set(ylabel="Held-out cross-entropy (nats/token)",
           title="Future text: actual time versus fixed 2020 time")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    delta.axhline(0, color="#555", linewidth=.8)
    delta.set(xlabel="Document date", ylabel="Loss gain")
    delta.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "forecast.png", dpi=180)
    plt.close(fig)


def plot_emergence():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    records = DATA["emergence"]
    x = [record["tokens_seen"] / 1e6 for record in records]
    for prompt in [item["prompt"] for item in records[0]["sensitivity"]]:
        values = [next(item["next_token_js_nats"] for item in record["sensitivity"]
                       if item["prompt"] == prompt) for record in records]
        ax.plot(x, values, marker="o", label=prompt)
    ax.set(xlabel="Training tokens (millions)", ylabel="Next-token JS divergence (nats)",
           title="Sensitivity to 2016 versus extrapolated 2023")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "temporal_sensitivity.png", dpi=180)
    plt.close(fig)


def plot_genealogy():
    examples = DATA["genealogy"]["examples"]
    edges = DATA["genealogy"]["edges"]
    families = sorted(set(row["family"] for row in examples))
    lookup = {family: index for index, family in enumerate(families)}
    fig, ax = plt.subplots(figsize=(10, 6))
    for edge in edges:
        child, parent = examples[edge["child"]], examples[edge["penalized_parent"]]
        if child["family"] == parent["family"]:
            ax.plot([parent["created_utc"], child["created_utc"]],
                    [lookup[parent["family"]], lookup[child["family"]]],
                    color="#777", alpha=.09, linewidth=.8)
    for family in families:
        rows = [row for row in examples if row["family"] == family]
        xs = [row["created_utc"] for row in rows]
        ax.scatter(xs, [lookup[family]] * len(xs), s=12, alpha=.75)
    from datetime import datetime, timezone
    ticks = range(2012, 2023, 2)
    ax.set_xticks([datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() for y in ticks],
                  labels=[str(y) for y in ticks])
    ax.set_yticks(range(len(families)), labels=families)
    ax.set(xlabel="Date", title="Sampled phrase occurrences and same-family temporal links")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "genealogy.png", dpi=180)
    plt.close(fig)


def main():
    forecast = {split: forecast_summary(rows)
                for split, rows in DATA["forecast"].items()}
    genealogy = DATA["genealogy"]
    lexical, n = lexical_baseline(genealogy["examples"])
    plain_distance = np.mean([edge["distance_years"] for edge in genealogy["edges"]])
    penalized_distance = np.mean([
        (genealogy["examples"][edge["child"]]["created_utc"] -
         genealogy["examples"][edge["penalized_parent"]]["created_utc"]) / 31557600
        for edge in genealogy["edges"]])
    summary = {"forecast": forecast, "genealogy": {
        "examples": len(genealogy["examples"]), "edges": len(genealogy["edges"]),
        "same_family_plain": genealogy["same_family_plain"],
        "same_family_penalized": genealogy["same_family_penalized"],
        "shuffled_label_mean": genealogy["shuffled_family_null_mean"],
        "lexical_tfidf_same_family": lexical, "lexical_edges": n,
        "mean_plain_gap_years": float(plain_distance),
        "mean_penalized_gap_years": float(penalized_distance),
    }, "generation_sensitivity": DATA["generation_sensitivity"],
       "emergence": DATA["emergence"]}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Export flat tables for later interactive charts without re-running the GPU.
    with (ROOT / "forecast_windows.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["split", "year", "month", "kind",
                                                   "created_utc", "actual_time_loss",
                                                   "fixed_2020_loss"])
        writer.writeheader()
        for split, rows in DATA["forecast"].items():
            writer.writerows({"split": split, **row} for row in rows)
    with (ROOT / "genealogy_nodes.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["index", "id", "family", "year",
                                                   "created_utc", "subreddit", "text"])
        writer.writeheader()
        writer.writerows({"index": i, **row} for i, row in enumerate(genealogy["examples"]))
    with (ROOT / "genealogy_edges.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(genealogy["edges"][0]))
        writer.writeheader()
        writer.writerows(genealogy["edges"])
    plot_forecast()
    plot_emergence()
    plot_genealogy()
    print(json.dumps({"forecast": forecast, "genealogy": summary["genealogy"]}, indent=2))


if __name__ == "__main__":
    main()

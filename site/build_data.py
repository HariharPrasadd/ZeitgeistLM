"""Turn reproducible analysis outputs into a compact, static website fixture."""

from collections import defaultdict
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
analysis = json.loads((ROOT / "analysis/analyze_results.json").read_text())
summary = json.loads((ROOT / "analysis/summary.json").read_text())
genealogy_v2 = json.loads((ROOT / "analysis/genealogy_v2_results.json").read_text())

# The site ships chart values and sampled Reddit titles, never token shards or weights.
forecast = []
for split, rows in analysis["forecast"].items():
    groups = defaultdict(list)
    for row in rows:
        groups[(row["year"], row["month"])].append(row)
    for (year, month), items in sorted(groups.items()):
        forecast.append({
            "year": year, "month": month, "split": split,
            "actual": statistics.mean(row["actual_time_loss"] for row in items),
            "fixed": statistics.mean(row["fixed_2020_loss"] for row in items),
        })

graph = analysis["genealogy"]
fixture = {
    "generation": analysis["generation"],
    "generationSensitivity": analysis["generation_sensitivity"],
    "forecast": forecast,
    "forecastSummary": summary["forecast"],
    "genealogy": {
        "nodes": graph["examples"], "edges": graph["edges"],
        "summary": summary["genealogy"],
        # Both comparison graphs share the same 485 sampled posts and chronological indices.
        "comparisonEdges": {
            "lexical": genealogy_v2["comparison_edges"]["tfidf"],
            "reranked": genealogy_v2["comparison_edges"]["fixed_final_ln_mean"],
        },
    },
}
target = Path(__file__).parent / "src/data/analysis.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(fixture, separators=(",", ":")))
print(f"Wrote {target} ({target.stat().st_size:,} bytes)")

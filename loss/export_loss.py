"""Archive a Modal training call's loss series and render its loss curve."""

import argparse
import csv
import json
from pathlib import Path
import re

import modal


TRAIN = re.compile(r"step (\d+) loss ([\d.]+) lr ([\deE+.-]+) norm ([\deE+.-]+) seconds ([\d.]+)")
VALIDATION = re.compile(r"validation step (\d+) loss ([\d.]+)")


def write_csv(path, columns, rows):
    """Write explicit column names so later plots can reuse the raw metrics."""
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("call_id", help="Modal function call to archive")
    parser.add_argument("--name", default="original_spike", help="Output filename prefix")
    args = parser.parse_args()

    train, validation = {}, {}
    call = modal.FunctionCall.from_id(args.call_id)
    for entry in call.logs.fetch(source="stdout"):
        for line in entry.message.splitlines():
            match = TRAIN.fullmatch(line)
            if match:
                step = int(match.group(1))
                train[step] = {
                    "step": step, "tokens_seen": (step + 1) * 262144,
                    "loss": float(match.group(2)), "learning_rate": float(match.group(3)),
                    "gradient_norm_before_clipping": float(match.group(4)),
                    "step_seconds": float(match.group(5)),
                    "timestamp_utc": entry.timestamp.isoformat(),
                }
            match = VALIDATION.fullmatch(line)
            if match:
                step = int(match.group(1))
                validation[step] = {
                    "step": step, "tokens_seen": step * 262144,
                    "loss": float(match.group(2)),
                    "timestamp_utc": entry.timestamp.isoformat(),
                }

    if not train or not validation:
        raise RuntimeError("Expected both training and validation metrics in the call logs")
    directory = Path(__file__).resolve().parent
    training = [train[step] for step in sorted(train)]
    validating = [validation[step] for step in sorted(validation)]
    write_csv(directory / f"{args.name}_train.csv", list(training[0]), training)
    write_csv(directory / f"{args.name}_validation.csv", list(validating[0]), validating)
    (directory / f"{args.name}_metadata.json").write_text(json.dumps({
        "modal_call_id": args.call_id,
        "tokens_per_step": 262144,
        "training_rows": len(training), "validation_rows": len(validating),
        "first_training_step": training[0]["step"],
        "last_training_step": training[-1]["step"],
        "note": "Original run before rollback; gradients are pre-clipping norms.",
    }, indent=2) + "\n")

    # Keep the full warmup curve in the archived image; later charts can restyle the CSVs.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    fig.patch.set_facecolor("#101318")
    ax.set_facecolor("#191f27")
    ax.plot([row["step"] for row in training], [row["loss"] for row in training],
            color="#62d6b0", linewidth=0.8, alpha=0.8, label="Training")
    ax.plot([row["step"] for row in validating], [row["loss"] for row in validating],
            color="#f5bc66", linewidth=2, marker="o", markersize=3, label="Validation")
    ax.set(title="ZeitgeistLM · original run and loss spike", xlabel="Optimizer step",
           ylabel="Cross-entropy loss")
    ax.grid(color="#43505e", alpha=0.35)
    ax.tick_params(colors="#c8d0da")
    for spine in ax.spines.values():
        spine.set_color("#43505e")
    ax.xaxis.label.set_color("#c8d0da")
    ax.yaxis.label.set_color("#c8d0da")
    ax.title.set_color("#eceff3")
    legend = ax.legend(facecolor="#191f27", edgecolor="#43505e")
    for label in legend.get_texts():
        label.set_color("#eceff3")
    fig.tight_layout()
    fig.savefig(directory / f"{args.name}.png", facecolor=fig.get_facecolor())
    fig.savefig(directory / f"{args.name}.svg", facecolor=fig.get_facecolor())
    print(f"Archived {len(training)} training and {len(validating)} validation points in {directory}")


if __name__ == "__main__":
    main()

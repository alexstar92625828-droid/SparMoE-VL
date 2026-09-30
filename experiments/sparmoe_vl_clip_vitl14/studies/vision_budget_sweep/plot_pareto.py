"""Plot the paper's visual performance-efficiency Pareto curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install the project's analysis dependencies to plot") from error

    with args.summary.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("study") != "vision_budget_sweep":
        raise ValueError("summary does not belong to the visual budget sweep")
    rows = summary["pareto"]
    x = [row["ffn_macs_g"]["mean"] for row in rows]
    x_error = [row["ffn_macs_g"]["std"] for row in rows]
    y = [row["average_r1_retention_percent"]["mean"] for row in rows]
    y_error = [row["average_r1_retention_percent"]["std"] for row in rows]

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.errorbar(x, y, xerr=x_error, yerr=y_error, marker="o", capsize=3)
    for row, x_value, y_value in zip(rows, x, y):
        axis.annotate(
            f"p={row['target_ratio']:.1f}",
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_xlabel("Visual FFN MACs (G)")
    axis.set_ylabel("Average R@1 retention (%)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()

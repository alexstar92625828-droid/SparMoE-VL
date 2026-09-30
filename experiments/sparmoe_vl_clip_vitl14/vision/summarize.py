"""Aggregate the three visual-main evaluations without embedding paper results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


SEEDS = (42, 123, 2026)
METRICS = ("I2T_R1", "I2T_R5", "I2T_R10", "T2I_R1", "T2I_R5", "T2I_R10")
MACS = ("sparse_total_g", "sparse_ffn_g", "ffn_reduction_percent")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load_evaluations(input_root: Path) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        path = input_root / f"seed_{seed}" / "evaluation" / "coco.json"
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        if row.get("method") != "sparmoe_vl_two_stage":
            raise ValueError(f"{path}: unexpected method")
        if row.get("modality") != "vision" or row.get("target_ratio") != 0.7:
            raise ValueError(f"{path}: not a visual p=0.7 main evaluation")
        if set(row.get("sparmoe_vl", {})) != set(METRICS):
            raise ValueError(f"{path}: retrieval metric set changed")
        rows.append(row)
    if any(row["dense"] != rows[0]["dense"] for row in rows[1:]):
        raise ValueError("Dense reference metrics differ across seeded evaluations")
    return rows


def aggregate(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "values": list(values),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_evaluations(args.input_root)
    summary = {
        "format_version": 1,
        "method": "sparmoe_vl_two_stage",
        "modality": "vision",
        "target_ratio": 0.7,
        "seeds": list(SEEDS),
        "statistics": "mean and sample standard deviation (ddof=1)",
        "dense": rows[0]["dense"],
        "sparmoe_vl": {
            metric: aggregate([float(row["sparmoe_vl"][metric]) for row in rows])
            for metric in METRICS
        },
        "macs": {
            metric: aggregate([float(row["macs"][metric]) for row in rows]) for metric in MACS
        },
        "mean_metric_retention_percent": aggregate(
            [float(row["mean_metric_retention_percent"]) for row in rows]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()

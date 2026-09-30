"""Aggregate the five visual budgets and three paper seeds into Table 3."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from sparmoe_vl.studies.vision_budget_sweep import (
    BUDGET_POINTS,
    DATASET_SHA256,
    SEEDS,
    STUDY_NAME,
    budget_tag,
)


SUMMARY_FIELDS = (
    "ffn_macs_g",
    "ffn_reduction_percent",
    "active_visual_parameters_m",
    "coco_i2t_r1",
    "coco_i2t_r5",
    "coco_t2i_r1",
    "coco_t2i_r5",
    "flickr_i2t_r1",
    "flickr_i2t_r5",
    "flickr_t2i_r1",
    "flickr_t2i_r5",
    "average_r1_retention_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path, target_ratio: float, seed: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("method") != "SparMoE-VL" or payload.get("study") != STUDY_NAME:
        raise ValueError(f"not a {STUDY_NAME} result: {path}")
    checkpoint = payload.get("checkpoint", {})
    expected = {
        "target_ratio": target_ratio,
        "training_seed": seed,
        "data_seed": 42,
        "dataset_sha256": DATASET_SHA256,
        "reuses_visual_main_experiment": target_ratio == 0.7,
    }
    for key, wanted in expected.items():
        if checkpoint.get(key) != wanted:
            raise ValueError(
                f"{path}: checkpoint.{key}={checkpoint.get(key)!r}; expected {wanted!r}"
            )
    if payload.get("training_data", {}).get("candidate_pool_size") != 500_000:
        raise ValueError(f"{path}: result does not identify the exact 500k pool")

    dense = payload["dense"]
    sparse = payload["sparse"]
    coco = sparse["coco"]
    flickr = sparse["flickr30k"]
    return {
        "target_ratio": target_ratio,
        "training_seed": seed,
        "checkpoint_step": checkpoint["checkpoint_step"],
        "checkpoint_format": checkpoint["format"],
        "ffn_macs_g": float(sparse["ffn_macs_g"]),
        "ffn_reduction_percent": float(sparse["ffn_reduction_percent"]),
        "active_visual_parameters_m": float(sparse["active_visual_parameters_m"]),
        "coco_i2t_r1": float(coco["i2t_r1"]),
        "coco_i2t_r5": float(coco["i2t_r5"]),
        "coco_t2i_r1": float(coco["t2i_r1"]),
        "coco_t2i_r5": float(coco["t2i_r5"]),
        "flickr_i2t_r1": float(flickr["i2t_r1"]),
        "flickr_i2t_r5": float(flickr["i2t_r5"]),
        "flickr_t2i_r1": float(flickr["t2i_r1"]),
        "flickr_t2i_r5": float(flickr["t2i_r5"]),
        "average_r1_retention_percent": float(sparse["average_r1_retention_percent"]),
        "dense": dense,
    }


def stats(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        raise ValueError(f"expected exactly {len(SEEDS)} seeds")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "values": list(values),
    }


def validate_dense_reference(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reference = rows[0]["dense"]
    encoded = json.dumps(reference, sort_keys=True)
    for row in rows[1:]:
        if json.dumps(row["dense"], sort_keys=True) != encoded:
            raise ValueError("Dense retrieval reference changed between sweep runs")
    return reference


def format_cell(row: dict[str, Any], field: str, suffix: str = "") -> str:
    value = row[field]
    return f"{value['mean']:.2f}±{value['std']:.2f}{suffix}"


def render_table(summaries: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| p | FFN MACs (G) | FFN reduction | Active params (M) | "
        "COCO I2T R@1/R@5 | COCO T2I R@1/R@5 | "
        "Flickr I2T R@1/R@5 | Flickr T2I R@1/R@5 | Avg. R@1 retention |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['target_ratio']:.1f} | {format_cell(row, 'ffn_macs_g')} | "
            f"{format_cell(row, 'ffn_reduction_percent', '%')} | "
            f"{format_cell(row, 'active_visual_parameters_m')} | "
            f"{format_cell(row, 'coco_i2t_r1')} / {format_cell(row, 'coco_i2t_r5')} | "
            f"{format_cell(row, 'coco_t2i_r1')} / {format_cell(row, 'coco_t2i_r5')} | "
            f"{format_cell(row, 'flickr_i2t_r1')} / {format_cell(row, 'flickr_i2t_r5')} | "
            f"{format_cell(row, 'flickr_t2i_r1')} / {format_cell(row, 'flickr_t2i_r5')} | "
            f"{format_cell(row, 'average_r1_retention_percent', '%')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    per_seed = []
    summaries = []
    for point in BUDGET_POINTS:
        rows = [
            load_result(
                args.results_root / budget_tag(point) / f"seed_{seed}" / "evaluation.json",
                point,
                seed,
            )
            for seed in SEEDS
        ]
        per_seed.extend(rows)
        summaries.append(
            {
                "target_ratio": point,
                **{
                    field: stats([float(row[field]) for row in rows])
                    for field in SUMMARY_FIELDS
                },
            }
        )

    dense_reference = validate_dense_reference(per_seed)
    serializable_rows = [
        {key: value for key, value in row.items() if key != "dense"} for row in per_seed
    ]
    summary = {
        "format_version": 1,
        "study": STUDY_NAME,
        "budget_points": list(BUDGET_POINTS),
        "training_seeds": list(SEEDS),
        "data_seed": 42,
        "candidate_pool_size": 500_000,
        "dataset_sha256": DATASET_SHA256,
        "p07_source": "visual_main_experiment",
        "standard_deviation": "sample (n-1)",
        "dense_reference": dense_reference,
        "pareto": summaries,
        "per_seed": serializable_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    with (args.output_dir / "per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=serializable_rows[0].keys())
        writer.writeheader()
        writer.writerows(serializable_rows)
    table = render_table(summaries)
    (args.output_dir / "table.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

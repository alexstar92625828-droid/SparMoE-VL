"""Aggregate all six seeded evaluations into the paper's Table 4."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from sparmoe_vl.studies.generalization import (
    DATASET_COUNTS,
    SEEDS,
    STUDY_NAME,
    TEXT_DATASET_SHA256,
    TEXT_TARGET_RATIO,
    VISION_DATASET_SHA256,
    VISION_TARGET_RATIO,
)


FIELDS = (
    "vision_macs_g",
    "text_macs_g",
    "coco_i2t_r1",
    "coco_t2i_r1",
    "flickr30k_i2t_r1",
    "flickr30k_t2i_r1",
    "cifar100_accuracy",
    "imagenet1k_accuracy",
    "food101_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path, modality: str, seed: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    expected_sha = VISION_DATASET_SHA256 if modality == "vision" else TEXT_DATASET_SHA256
    expected_ratio = VISION_TARGET_RATIO if modality == "vision" else TEXT_TARGET_RATIO
    expected = {
        "method": "SparMoE-VL",
        "study": STUDY_NAME,
        "modality": modality,
    }
    for key, wanted in expected.items():
        if result.get(key) != wanted:
            raise ValueError(f"{path}: {key}={result.get(key)!r}; expected {wanted!r}")
    checkpoint = result.get("checkpoint", {})
    checkpoint_expected = {
        "training_seed": seed,
        "data_seed": 42,
        "pool_size": 500_000,
        "dataset_sha256": expected_sha,
        "target_ratio": expected_ratio,
    }
    for key, wanted in checkpoint_expected.items():
        if checkpoint.get(key) != wanted:
            raise ValueError(
                f"{path}: checkpoint.{key}={checkpoint.get(key)!r}; expected {wanted!r}"
            )
    evaluation = result.get("evaluation", {})
    if evaluation.get("counts") != DATASET_COUNTS:
        raise ValueError(f"{path}: evaluation dataset counts changed")
    sparse = result["sparse"]
    if modality == "vision" and float(sparse["macs"]["text_g"]) != 6.64925184:
        raise ValueError(f"{path}: visual conversion must keep the text tower Dense")
    if modality == "text" and float(sparse["macs"]["vision_g"]) != 81.012768768:
        raise ValueError(f"{path}: text conversion must keep the visual tower Dense")
    return {
        "modality": modality,
        "training_seed": seed,
        "checkpoint_step": checkpoint["checkpoint_step"],
        "checkpoint_format": checkpoint["format"],
        "vision_macs_g": float(sparse["macs"]["vision_g"]),
        "text_macs_g": float(sparse["macs"]["text_g"]),
        "coco_i2t_r1": float(sparse["retrieval"]["coco"]["i2t_r1"]),
        "coco_t2i_r1": float(sparse["retrieval"]["coco"]["t2i_r1"]),
        "flickr30k_i2t_r1": float(sparse["retrieval"]["flickr30k"]["i2t_r1"]),
        "flickr30k_t2i_r1": float(sparse["retrieval"]["flickr30k"]["t2i_r1"]),
        "cifar100_accuracy": float(sparse["classification"]["cifar100_accuracy"]),
        "imagenet1k_accuracy": float(sparse["classification"]["imagenet1k_accuracy"]),
        "food101_accuracy": float(sparse["classification"]["food101_accuracy"]),
        "dense": result["dense"],
        "cache": evaluation["cache"],
        "dataset_identities": evaluation["dataset_identities"],
    }


def stats(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        raise ValueError(f"expected exactly {len(SEEDS)} seeded values")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "values": list(values),
    }


def validate_shared_evaluation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dense = rows[0]["dense"]
    cache = rows[0]["cache"]
    identities = rows[0]["dataset_identities"]
    for row in rows[1:]:
        if row["dense"] != dense:
            raise ValueError("Dense reference metrics differ between evaluations")
        if row["cache"] != cache:
            raise ValueError("the six evaluations did not use one shared Dense cache")
        if row["dataset_identities"] != identities:
            raise ValueError("evaluation dataset identities differ between runs")
    return dense


def formatted(row: dict[str, Any], field: str, digits: int) -> str:
    item = row[field]
    return f"{item['mean']:.{digits}f} ± {item['std']:.{digits}f}"


def render_table(
    dense: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
) -> str:
    header = [
        "| Model | MACs-V (G) | MACs-T (G) | COCO I2T@1 | COCO T2I@1 | "
        "Flickr30k I2T@1 | Flickr30k T2I@1 | CIFAR-100 | ImageNet-1K | Food-101 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    dense_retrieval = dense["retrieval"]
    dense_classification = dense["classification"]
    header.append(
        f"| Dense | {dense['macs']['vision_g']:.4f} | {dense['macs']['text_g']:.4f} | "
        f"{dense_retrieval['coco']['i2t_r1']:.2f} | "
        f"{dense_retrieval['coco']['t2i_r1']:.2f} | "
        f"{dense_retrieval['flickr30k']['i2t_r1']:.2f} | "
        f"{dense_retrieval['flickr30k']['t2i_r1']:.2f} | "
        f"{dense_classification['cifar100_accuracy']:.2f} | "
        f"{dense_classification['imagenet1k_accuracy']:.2f} | "
        f"{dense_classification['food101_accuracy']:.2f} |"
    )
    for modality, label in (("vision", "Vision"), ("text", "Text")):
        row = summaries[modality]
        vision_macs = (
            formatted(row, "vision_macs_g", 4)
            if modality == "vision"
            else f"{dense['macs']['vision_g']:.4f}"
        )
        text_macs = (
            formatted(row, "text_macs_g", 4)
            if modality == "text"
            else f"{dense['macs']['text_g']:.4f}"
        )
        header.append(
            f"| SparMoE-VL ({label}) | {vision_macs} | {text_macs} | "
            f"{formatted(row, 'coco_i2t_r1', 2)} | "
            f"{formatted(row, 'coco_t2i_r1', 2)} | "
            f"{formatted(row, 'flickr30k_i2t_r1', 2)} | "
            f"{formatted(row, 'flickr30k_t2i_r1', 2)} | "
            f"{formatted(row, 'cifar100_accuracy', 2)} | "
            f"{formatted(row, 'imagenet1k_accuracy', 2)} | "
            f"{formatted(row, 'food101_accuracy', 2)} |"
        )
    return "\n".join(header) + "\n"


def main() -> None:
    args = parse_args()
    rows = [
        load_result(
            args.results_root / modality / f"seed_{seed}" / "evaluation.json",
            modality,
            seed,
        )
        for modality in ("vision", "text")
        for seed in SEEDS
    ]
    dense = validate_shared_evaluation(rows)
    summaries = {
        modality: {
            field: stats([row[field] for row in rows if row["modality"] == modality])
            for field in FIELDS
        }
        for modality in ("vision", "text")
    }
    portable_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in ("dense", "cache", "dataset_identities")
        }
        for row in rows
    ]
    summary = {
        "format_version": 1,
        "study": STUDY_NAME,
        "training_seeds": list(SEEDS),
        "data_seed": 42,
        "counts": DATASET_COUNTS,
        "standard_deviation": "sample (n-1)",
        "dense": dense,
        "sparmoe_vl": summaries,
        "per_seed": portable_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    with (args.output_dir / "per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=portable_rows[0].keys())
        writer.writeheader()
        writer.writerows(portable_rows)
    table = render_table(dense, summaries)
    (args.output_dir / "table.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

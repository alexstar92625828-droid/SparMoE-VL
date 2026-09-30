"""Aggregate the three Table-7 model seeds with sample standard deviation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional, Sequence

from ...common.two_stage import STAGE2_PROTOCOL
from .protocol import (
    ALLOCATIONS,
    COCO_ANNOTATIONS_SHA256,
    DATA_SEED,
    EVAL_IMAGES,
    LAYER_ALLOCATIONS,
    METRICS,
    MODEL_KEY,
    MODEL_NAME,
    OUTPUT_ROOT,
    SEEDS,
    STUDY_NAME,
    TOKEN_ALLOCATIONS,
    VISION_DATASET_SHA256,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=OUTPUT_ROOT / "evaluation")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "summary" / "summary.json")
    return parser.parse_args(argv)


def result_path(root: Path, seed: int) -> Path:
    return root / f"seed_{seed}" / "result.json"


def validate_result(result: dict[str, Any], seed: int, path: Path) -> None:
    expected = {
        "run_seed": seed,
        "data_seed": DATA_SEED,
        "dataset_sha256": VISION_DATASET_SHA256,
        "images": EVAL_IMAGES,
    }
    for key, wanted in expected.items():
        if result.get(key) != wanted:
            raise ValueError(f"{path}: {key}={result.get(key)!r}; expected {wanted!r}")
    optional = {
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "training_protocol": STAGE2_PROTOCOL,
        "evaluation_annotation_sha256": COCO_ANNOTATIONS_SHA256,
    }
    for key, wanted in optional.items():
        if key in result and result[key] != wanted:
            raise ValueError(f"{path}: {key}={result[key]!r}; expected {wanted!r}")
    completed = result.get("completed")
    if not isinstance(completed, dict) or tuple(completed) != ALLOCATIONS:
        raise ValueError(f"{path}: incomplete or reordered Table-7 allocations")
    for allocation in ALLOCATIONS:
        row = completed[allocation]
        if row.get("Allocation") != allocation:
            raise ValueError(f"{path}: malformed allocation row {allocation}")
        missing = [metric for metric in METRICS if metric not in row]
        if missing:
            raise ValueError(f"{path}: {allocation} is missing metrics {missing}")


def allocation_summary(
    results: Sequence[dict[str, Any]],
    allocation: str,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    source_name = allocation if source is None else source
    return {
        metric: {
            "values": [float(result["completed"][source_name][metric]) for result in results],
            "mean": statistics.mean(
                float(result["completed"][source_name][metric]) for result in results
            ),
            "sample_sd": statistics.stdev(
                float(result["completed"][source_name][metric]) for result in results
            ),
        }
        for metric in METRICS
    }


def summarize(paths: Sequence[Path]) -> dict[str, Any]:
    if len(paths) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} result files, got {len(paths)}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for seed, result, path in zip(SEEDS, results, paths):
        validate_result(result, seed, path)
    token = {
        allocation: allocation_summary(results, allocation) for allocation in TOKEN_ALLOCATIONS
    }
    layer = {
        "Layer-Self": allocation_summary(results, "Layer-Self", source="Self"),
        "Layer-Uniform": allocation_summary(results, "Layer-Uniform"),
        "Layer-Shuffled": allocation_summary(results, "Layer-Shuffled"),
    }
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "paper_scope": "Table 7",
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "run_seeds": list(SEEDS),
        "data_seed": DATA_SEED,
        "dataset_sha256": VISION_DATASET_SHA256,
        "checkpoint_selection": "best.pt",
        "images": EVAL_IMAGES,
        "statistics": "mean and sample standard deviation (ddof=1)",
        "token_allocation": token,
        "layer_allocation": layer,
        "per_seed_checkpoint_steps": {
            str(seed): int(result["checkpoint_step"]) for seed, result in zip(SEEDS, results)
        },
    }


def markdown_table(summary: dict[str, Any], granularity: str) -> str:
    if granularity == "token":
        allocations = TOKEN_ALLOCATIONS
        rows = summary["token_allocation"]
    elif granularity == "layer":
        allocations = LAYER_ALLOCATIONS
        rows = summary["layer_allocation"]
    else:
        raise ValueError("granularity must be 'token' or 'layer'")
    lines = [
        "| Allocation | Activated FFN MACs (G) ↓ | Cosine ↑ | NRE ↓ | Recovery Rate ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for allocation in allocations:
        stats = rows[allocation]

        def cell(metric: str, decimals: int) -> str:
            value = stats[metric]
            return f"{value['mean']:.{decimals}f} ± {value['sample_sd']:.{decimals}f}"

        recovery = stats["Recovery Rate ↑"]
        lines.append(
            f"| {allocation} | {cell('Activated FFN MACs (G)', 4)} | "
            f"{cell('Cosine ↑', 4)} | {cell('NRE ↓', 4)} | "
            f"{100 * recovery['mean']:.2f} ± "
            f"{100 * recovery['sample_sd']:.2f}% |"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    paths = [result_path(args.input_root, seed) for seed in SEEDS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Table-7 seed results: {missing}")
    result = summarize(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(markdown_table(result, "token"))
    print()
    print(markdown_table(result, "layer"))
    print(f"summary={args.output}")

"""Validate and aggregate the result-generating Table-8 evaluations."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .protocol import (
    COMPONENT_FLAGS,
    DATASET_SHA256,
    DATA_SEED,
    EVALUATION_COUNTS,
    EVALUATION_SEED,
    EVALUATION_SHA256,
    METRICS,
    METHOD_LABELS,
    METHODS,
    MODEL_KEY,
    MODEL_NAME,
    OUTPUT_ROOT,
    PAPER_SEEDS,
    REPLACEMENTS,
    STUDY_NAME,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=OUTPUT_ROOT / "evaluation")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "summary" / "summary.json",
    )
    return parser.parse_args(argv)


def result_path(root: Path, method: str, seed: int) -> Path:
    return root / method / f"seed_{seed}" / "result.json"


def validate_result(
    result: Mapping[str, Any],
    method: str,
    seed: int,
    source: str | Path,
) -> None:
    expected = {
        "study": STUDY_NAME,
        "paper_scope": "Table 8",
        "method": method,
        "label": METHOD_LABELS[method],
        "replacement": REPLACEMENTS[method],
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "run_seed": seed,
        "data_seed": DATA_SEED,
        "dataset_sha256": DATASET_SHA256,
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_sha256": EVALUATION_SHA256,
        "counts": EVALUATION_COUNTS,
    }
    for key, wanted in expected.items():
        if result.get(key) != wanted:
            raise ValueError(f"{source}: {key}={result.get(key)!r}; expected {wanted!r}")
    expected_routing = (
        "uniform_random_per_patch_token"
        if method == "without_token_router"
        else "learned_argmax"
    )
    if result.get("routing") != expected_routing:
        raise ValueError(
            f"{source}: routing={result.get('routing')!r}; expected {expected_routing!r}"
        )
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{source}: missing metrics mapping")
    missing = [metric for metric in METRICS if metric not in metrics]
    if missing:
        raise ValueError(f"{source}: missing metrics {missing}")
    dense_reference = result.get("dense_reference")
    if not isinstance(dense_reference, Mapping):
        raise ValueError(f"{source}: missing measured Dense reference")
    missing_dense = [metric for metric in METRICS[:-1] if metric not in dense_reference]
    if missing_dense:
        raise ValueError(f"{source}: Dense reference is missing metrics {missing_dense}")


def aggregate_method(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        metric: {
            "values": [float(result["metrics"][metric]) for result in results],
            "mean": statistics.mean(float(result["metrics"][metric]) for result in results),
            "sample_sd": statistics.stdev(
                float(result["metrics"][metric]) for result in results
            ),
        }
        for metric in METRICS
    }


def summarize_results(
    results_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if tuple(results_by_method) != METHODS:
        raise ValueError(f"results must contain Table-8 methods in canonical order: {METHODS}")
    methods = {}
    dense_reference: dict[str, float] | None = None
    checkpoint_steps: dict[str, dict[str, int]] = {}
    for method in METHODS:
        seeds = PAPER_SEEDS[method]
        results = results_by_method[method]
        if len(results) != len(seeds):
            raise ValueError(f"{method}: expected {len(seeds)} results, got {len(results)}")
        for seed, result in zip(seeds, results):
            validate_result(result, method, seed, f"{method}/seed_{seed}")
        methods[method] = aggregate_method(results)
        for seed, result in zip(seeds, results):
            current = {
                metric: float(result["dense_reference"][metric]) for metric in METRICS[:-1]
            }
            if dense_reference is None:
                dense_reference = current
            elif any(
                not math.isclose(current[key], dense_reference[key], abs_tol=1e-6)
                for key in current
            ):
                raise ValueError(f"{method}/seed_{seed}: measured Dense reference changed")
        checkpoint_steps[method] = {
            str(seed): int(result["checkpoint_step"]) for seed, result in zip(seeds, results)
        }
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "paper_scope": "Table 8",
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "data_seed": DATA_SEED,
        "dataset_sha256": DATASET_SHA256,
        "method_seeds": {method: list(PAPER_SEEDS[method]) for method in METHODS},
        "seed_protocol_note": (
            "The paper-reproducing geometry-preservation row uses historical "
            "seeds 42, 123, and 3407; all other rows use 42, 123, and 2026."
        ),
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_sha256": EVALUATION_SHA256,
        "counts": EVALUATION_COUNTS,
        "statistics": "mean and sample standard deviation (ddof=1)",
        "dense_reference": dense_reference,
        "component_flags": COMPONENT_FLAGS,
        "methods": methods,
        "per_seed_checkpoint_steps": checkpoint_steps,
    }


def summarize_paths(
    paths_by_method: Mapping[str, Sequence[Path]],
) -> dict[str, Any]:
    results: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        paths = paths_by_method.get(method, ())
        results[method] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"missing Table-8 result: {path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"{path}: expected a JSON object")
            results[method].append(value)
    return summarize_results(results)


def _cell(row: Mapping[str, Any], metric: str, decimals: int) -> str:
    value = row[metric]
    return f"{value['mean']:.{decimals}f} ± {value['sample_sd']:.{decimals}f}"


def markdown_table(summary: Mapping[str, Any]) -> str:
    dense = summary["dense_reference"]
    lines = [
        "| Method | SPG | Token Router | Layer-Adaptive Budget | "
        "Geometry Preservation | FFN MACs-V (G) ↓ | COCO I2T@1 ↑ | "
        "COCO T2I@1 ↑ | Flickr30k I2T@1 ↑ | Flickr30k T2I@1 ↑ | Retention ↑ |",
        "|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|",
        f"| Dense | – | – | – | – | {dense['ffn_macs_v_g']:.2f} | "
        f"{dense['coco_i2t_r1']:.2f} | {dense['coco_t2i_r1']:.2f} | "
        f"{dense['flickr_i2t_r1']:.2f} | {dense['flickr_t2i_r1']:.2f} | – |",
    ]
    for method in METHODS:
        row = summary["methods"][method]
        spg, router, layer_budget, geometry = summary["component_flags"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {spg} | {router} | {layer_budget} | "
            f"{geometry} | {_cell(row, 'ffn_macs_v_g', 4)} | "
            f"{_cell(row, 'coco_i2t_r1', 2)} | "
            f"{_cell(row, 'coco_t2i_r1', 2)} | "
            f"{_cell(row, 'flickr_i2t_r1', 2)} | "
            f"{_cell(row, 'flickr_t2i_r1', 2)} | "
            f"{_cell(row, 'retention_percent', 2)}% |"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    paths = {
        method: [result_path(args.input_root, method, seed) for seed in PAPER_SEEDS[method]]
        for method in METHODS
    }
    summary = summarize_paths(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(markdown_table(summary))
    print(f"summary={args.output}")

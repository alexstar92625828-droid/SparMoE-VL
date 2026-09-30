"""Aggregate three seeded SigLIP2 retrieval evaluations."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional, Sequence

from ...common.two_stage import STAGE2_PROTOCOL
from ..siglip.summary import METRICS
from .protocol import DATA_SEED, PROJECT_ROOT, SEEDS, expected_dataset_sha256, get_spec


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    model: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("vision", "text"), required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "architecture_transfer" / "siglip2",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.model = model
    return args


def result_path(root: Path, model: str, modality: str, seed: int) -> Path:
    return root / model / modality / f"seed_{seed}" / "retrieval.json"


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= 1e-8
    return left == right


def summarize(
    paths: Sequence[Path],
    model_name: str,
    modality: str,
) -> dict[str, Any]:
    spec = get_spec(model_name)
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected_sha = expected_dataset_sha256(modality)
    for seed, result, path in zip(SEEDS, results, paths):
        expected = {
            "run_seed": seed,
            "data_seed": DATA_SEED,
            "dataset_sha256": expected_sha,
            "protocol": STAGE2_PROTOCOL,
            "model_key": spec.key,
        }
        for key, wanted in expected.items():
            if result.get(key) != wanted:
                raise ValueError(f"{path}: {key}={result.get(key)!r}; expected {wanted!r}")
        for key, wanted in (("model_name", spec.model_name), ("modality", modality)):
            if key in result and result[key] != wanted:
                raise ValueError(f"{path}: {key}={result[key]!r}; expected {wanted!r}")
    dense = results[0]["dense"]
    for path, result in zip(paths[1:], results[1:]):
        for key, value in dense.items():
            if not _same(result["dense"].get(key), value):
                raise ValueError(f"{path}: dense reference changed at {key}")
    sparse = {}
    for metric in METRICS:
        values = [float(result["sparse"][metric]) for result in results]
        sparse[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "values": values,
        }
    per_seed = [
        {
            "run_seed": seed,
            "checkpoint_step": int(result["checkpoint_step"]),
            **{metric: float(result["sparse"][metric]) for metric in METRICS},
        }
        for seed, result in zip(SEEDS, results)
    ]
    return {
        "run_seeds": list(SEEDS),
        "data_seed": DATA_SEED,
        "dataset_sha256": expected_sha,
        "protocol": STAGE2_PROTOCOL,
        "model_key": spec.key,
        "model_name": spec.model_name,
        "modality": modality,
        "dense": dense,
        "sparse": sparse,
        "per_seed": per_seed,
    }


def markdown_row(summary: dict[str, Any]) -> str:
    sparse = summary["sparse"]

    def value(name: str) -> str:
        return f"{sparse[name]['mean']:.2f} ± {sparse[name]['std']:.2f}"

    return " | ".join(
        (
            summary["model_name"],
            summary["modality"],
            value("ffn_reduction_pct"),
            value("COCO_I2T_R1"),
            value("COCO_T2I_R1"),
            value("Flickr_I2T_R1"),
            value("Flickr_T2I_R1"),
            value("average_retention_pct"),
        )
    )


def main(model: str, argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv, model=model)
    paths = [result_path(args.input_root, model, args.modality, seed) for seed in SEEDS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing seeded evaluations: {missing}")
    result = summarize(paths, model, args.modality)
    output = args.output or (args.input_root / model / args.modality / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(markdown_row(result))
    print(f"summary={output}")

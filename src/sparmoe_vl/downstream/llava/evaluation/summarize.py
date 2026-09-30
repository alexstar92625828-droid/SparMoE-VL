"""Aggregate Dense LLaVA and three seeded SparMoE-LLaVA evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..protocol import (
    BENCHMARK_COUNTS,
    BENCHMARK_IDENTITIES,
    DENSE_VISUAL_FFN_MACS_G,
    SEEDS,
    STUDY_NAME,
    TRAINING_DATASET_SHA256,
)
from .common import save_json


BENCHMARKS = ("pope", "mme_p", "gqa", "vqav2")
DATA_IDENTITIES = {
    "pope": {
        key: BENCHMARK_IDENTITIES[key]
        for key in (
            "pope_random_sha256",
            "pope_popular_sha256",
            "pope_adversarial_sha256",
        )
    },
    "mme_p": {"mme_p_annotations_sha256": BENCHMARK_IDENTITIES["mme_p_annotations_sha256"]},
    "gqa": {"gqa_questions_sha256": BENCHMARK_IDENTITIES["gqa_questions_sha256"]},
    "vqav2": {
        "vqav2_questions_sha256": BENCHMARK_IDENTITIES["vqav2_questions_sha256"],
        "vqav2_annotations_sha256": BENCHMARK_IDENTITIES["vqav2_annotations_sha256"],
    },
}
DATA_IDENTITIES["visual_ffn_macs"] = DATA_IDENTITIES["pope"]
GENERATION_PROTOCOL = {
    "pope": (8, "Answer with yes or no only."),
    "mme_p": (8, ""),
    "gqa": (12, "Answer with a single word or short phrase."),
    "vqav2": (12, "Answer with a single word or short phrase."),
}


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty evaluation result: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation result must be a mapping: {path}")
    return payload


def _validate_result(
    result: Mapping[str, Any],
    path: Path,
    benchmark: str,
    mode: str,
    seed: int | None,
) -> None:
    expected = {"study": STUDY_NAME, "benchmark": benchmark, "mode": mode}
    for key, wanted in expected.items():
        if result.get(key) != wanted:
            raise ValueError(f"{path}: {key}={result.get(key)!r}; expected {wanted!r}")
    if result.get("data_identity") != DATA_IDENTITIES[benchmark]:
        raise ValueError(f"{path}: evaluation data identity differs from the paper")
    checkpoint = result.get("checkpoint")
    if mode == "dense":
        if checkpoint is not None:
            raise ValueError(f"{path}: Dense result unexpectedly contains a checkpoint")
    else:
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{path}: Sparse result has no checkpoint identity")
        checkpoint_expected = {
            "stage": 2,
            "training_seed": seed,
            "data_seed": 42,
            "pool_size": 500_000,
            "dataset_sha256": TRAINING_DATASET_SHA256,
            "target_ratio": 0.7,
        }
        for key, wanted in checkpoint_expected.items():
            if checkpoint.get(key) != wanted:
                raise ValueError(
                    f"{path}: checkpoint.{key}={checkpoint.get(key)!r}; expected {wanted!r}"
                )
    if benchmark in GENERATION_PROTOCOL:
        max_new_tokens, question_suffix = GENERATION_PROTOCOL[benchmark]
        generation = result.get("generation", {})
        if generation.get("do_sample") is not False:
            raise ValueError(f"{path}: decoding must be deterministic")
        if int(generation.get("max_new_tokens", -1)) != max_new_tokens:
            raise ValueError(f"{path}: generation length differs from the paper")
        if generation.get("question_suffix") != question_suffix:
            raise ValueError(f"{path}: question prompt differs from the paper")
        model = result.get("model", {})
        if (
            int(model.get("vision_select_layer", 0)) != -2
            or model.get("vision_select_feature") != "patch"
        ):
            raise ValueError(f"{path}: LLaVA visual feature selection changed")
    counts = result.get("counts", {})
    metrics = result.get("metrics", {})
    if benchmark == "pope" and counts != BENCHMARK_COUNTS["pope"]:
        raise ValueError(f"{path}: incomplete POPE result")
    if benchmark == "mme_p" and (
        counts != BENCHMARK_COUNTS["mme_p"] or int(metrics.get("num_categories", -1)) != 10
    ):
        raise ValueError(f"{path}: incomplete MME-P result")
    if benchmark == "gqa" and (
        counts != BENCHMARK_COUNTS["gqa"] or int(metrics.get("num_questions", -1)) != 12_578
    ):
        raise ValueError(f"{path}: incomplete GQA result")
    if benchmark == "vqav2" and (
        int(counts.get("start_index", -1)) != 0
        or int(counts.get("end_index", -1)) != 214_354
        or int(metrics.get("num_questions", -1)) != 214_354
    ):
        raise ValueError(f"{path}: incomplete VQAv2 result")


def _load_run(root: Path, mode: str, seed: int | None) -> dict[str, Any]:
    results = {}
    for benchmark in BENCHMARKS:
        path = root / benchmark / "evaluation.json"
        result = _load(path)
        _validate_result(result, path, benchmark, mode, seed)
        results[benchmark] = result
    if mode == "sparse":
        path = root / "macs" / "evaluation.json"
        result = _load(path)
        _validate_result(result, path, "visual_ffn_macs", mode, seed)
        if int(result.get("metrics", {}).get("num_images", -1)) != 500:
            raise ValueError(f"{path}: incomplete POPE MAC measurement")
        results["visual_ffn_macs"] = result
    return results


def _dense_metrics(results: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    return {
        "visual_ffn_macs_g": DENSE_VISUAL_FFN_MACS_G,
        "pope_avg_f1": float(results["pope"]["metrics"]["pope_avg_f1"]),
        "pope_yes_ratio": float(results["pope"]["metrics"]["pope_yes_ratio"]),
        "mme_p_score": float(results["mme_p"]["metrics"]["mme_p_score"]),
        "gqa_acc_pct": 100.0 * float(results["gqa"]["metrics"]["gqa_acc"]),
        "vqav2_acc_pct": 100.0 * float(results["vqav2"]["metrics"]["vqav2_acc"]),
    }


def _sparse_metrics(
    results: Mapping[str, Mapping[str, Any]],
    dense: Mapping[str, float],
) -> dict[str, float]:
    sparse_macs = float(results["visual_ffn_macs"]["metrics"]["sparse_visual_ffn_macs_g"])
    pope_f1 = float(results["pope"]["metrics"]["pope_avg_f1"])
    pope_yes = float(results["pope"]["metrics"]["pope_yes_ratio"])
    mme = float(results["mme_p"]["metrics"]["mme_p_score"])
    gqa = 100.0 * float(results["gqa"]["metrics"]["gqa_acc"])
    vqav2 = 100.0 * float(results["vqav2"]["metrics"]["vqav2_acc"])
    return {
        "visual_ffn_macs_g": sparse_macs,
        "visual_ffn_delta_pct": 100.0 * (sparse_macs / dense["visual_ffn_macs_g"] - 1.0),
        "pope_avg_f1": pope_f1,
        "pope_f1_retention_pct": 100.0 * pope_f1 / dense["pope_avg_f1"],
        "pope_yes_ratio": pope_yes,
        "pope_yes_ratio_delta": pope_yes - dense["pope_yes_ratio"],
        "mme_p_score": mme,
        "mme_p_retention_pct": 100.0 * mme / dense["mme_p_score"],
        "gqa_acc_pct": gqa,
        "gqa_retention_pct": 100.0 * gqa / dense["gqa_acc_pct"],
        "vqav2_acc_pct": vqav2,
        "vqav2_retention_pct": 100.0 * vqav2 / dense["vqav2_acc_pct"],
    }


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        raise ValueError(f"expected exactly {len(SEEDS)} seeded values")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "values": list(values),
    }


def summarize(dense_root: Path, sparse_root: Path) -> dict[str, Any]:
    dense_results = _load_run(dense_root, "dense", None)
    dense = _dense_metrics(dense_results)
    per_seed = []
    dense_identities = {
        benchmark: dense_results[benchmark]["data_identity"] for benchmark in BENCHMARKS
    }
    for seed in SEEDS:
        results = _load_run(sparse_root / f"seed_{seed}", "sparse", seed)
        for benchmark in BENCHMARKS:
            if results[benchmark]["data_identity"] != dense_identities[benchmark]:
                raise ValueError(
                    f"seed={seed}: Dense and Sparse {benchmark} data identities differ"
                )
        checkpoint_steps = {
            int(result["checkpoint"]["checkpoint_step"]) for result in results.values()
        }
        if len(checkpoint_steps) != 1:
            raise ValueError(f"seed={seed}: evaluations use different checkpoints")
        per_seed.append(
            {
                "run_seed": seed,
                "checkpoint_step": checkpoint_steps.pop(),
                **_sparse_metrics(results, dense),
            }
        )
    fields = [key for key in per_seed[0] if key not in ("run_seed", "checkpoint_step")]
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "run_seeds": list(SEEDS),
        "data_seed": 42,
        "training_pool_size": 500_000,
        "training_dataset_sha256": TRAINING_DATASET_SHA256,
        "dense": dense,
        "sparse": {field: _stats([float(row[field]) for row in per_seed]) for field in fields},
        "per_seed": per_seed,
    }


def render_table(summary: Mapping[str, Any]) -> str:
    dense = summary["dense"]
    sparse = summary["sparse"]

    def formatted(field: str, digits: int = 2) -> str:
        item = sparse[field]
        return f"{item['mean']:.{digits}f} ± {item['std']:.{digits}f}"

    return (
        "\n".join(
            [
                "| Benchmark | Metric | Dense-LLaVA | SparMoE-LLaVA | Retention / Delta |",
                "|---|---|---:|---:|---:|",
                f"| Visual FFN | MACs-V (G) | {dense['visual_ffn_macs_g']:.2f} | "
                f"{formatted('visual_ffn_macs_g')} | {formatted('visual_ffn_delta_pct')}% |",
                f"| POPE | Avg F1 | {dense['pope_avg_f1']:.4f} | "
                f"{formatted('pope_avg_f1', 4)} | {formatted('pope_f1_retention_pct')}% |",
                f"| POPE | Yes Ratio | {dense['pope_yes_ratio']:.4f} | "
                f"{formatted('pope_yes_ratio', 4)} | {formatted('pope_yes_ratio_delta', 4)} |",
                f"| MME-P | Score | {dense['mme_p_score']:.2f} | "
                f"{formatted('mme_p_score')} | {formatted('mme_p_retention_pct')}% |",
                f"| GQA | Acc | {dense['gqa_acc_pct']:.2f} | "
                f"{formatted('gqa_acc_pct')} | {formatted('gqa_retention_pct')}% |",
                f"| VQAv2 | Acc | {dense['vqav2_acc_pct']:.2f} | "
                f"{formatted('vqav2_acc_pct')} | {formatted('vqav2_retention_pct')}% |",
            ]
        )
        + "\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = summarize(args.dense_root, args.sparse_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, args.output_dir / "summary.json")
    with (args.output_dir / "per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary["per_seed"][0].keys())
        writer.writeheader()
        writer.writerows(summary["per_seed"])
    table = render_table(summary)
    (args.output_dir / "paper_table.md").write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

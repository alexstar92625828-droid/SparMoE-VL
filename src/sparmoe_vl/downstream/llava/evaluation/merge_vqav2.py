"""Merge complete, non-overlapping VQAv2 slices for one model run."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from ..protocol import BENCHMARK_COUNTS, STUDY_NAME, inspect_checkpoint
from .common import save_json


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dense", "sparse"), required=True)
    parser.add_argument("--chunk-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-total",
        type=int,
        default=BENCHMARK_COUNTS["vqav2"]["questions"],
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty VQAv2 artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _checkpoint_for_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.mode == "dense":
        if args.checkpoint is not None:
            raise ValueError("Dense VQAv2 merging must not receive a sparse checkpoint")
        return None
    if args.checkpoint is None:
        raise ValueError("Sparse VQAv2 merging requires --checkpoint")
    return inspect_checkpoint(args.checkpoint, expected_stage=2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.expected_total != BENCHMARK_COUNTS["vqav2"]["questions"]:
        raise ValueError("the paper protocol requires all 214,354 VQAv2 questions")
    checkpoint = _checkpoint_for_args(args)
    chunks = []
    first_evaluation = None
    for directory in args.chunk_dirs:
        evaluation_path = directory / "evaluation.json"
        predictions_path = directory / "predictions.jsonl"
        official_path = directory / "predictions_official.json"
        evaluation = _load_json(evaluation_path)
        official = _load_json(official_path)
        if not predictions_path.is_file() or predictions_path.stat().st_size == 0:
            raise FileNotFoundError(f"missing VQAv2 predictions: {predictions_path}")
        expected_fields = {
            "study": STUDY_NAME,
            "benchmark": "vqav2",
            "mode": args.mode,
            "checkpoint": checkpoint,
        }
        for key, wanted in expected_fields.items():
            if evaluation.get(key) != wanted:
                raise ValueError(
                    f"{evaluation_path}: {key}={evaluation.get(key)!r}; expected {wanted!r}"
                )
        counts = evaluation.get("counts", {})
        metrics = evaluation.get("metrics", {})
        start = int(counts.get("start_index", -1))
        end = int(counts.get("end_index", -1))
        count = int(metrics.get("num_questions", -1))
        if start < 0 or end <= start or count != end - start:
            raise ValueError(f"invalid VQAv2 slice in {evaluation_path}")
        if len(official) != count:
            raise ValueError(f"official prediction count mismatch in {official_path}")
        if first_evaluation is None:
            first_evaluation = evaluation
        else:
            for key in ("data_identity", "generation", "model"):
                if evaluation.get(key) != first_evaluation.get(key):
                    raise ValueError(f"VQAv2 chunks differ in {key}")
        chunks.append((start, end, directory, evaluation, official))
    chunks.sort(key=lambda item: item[0])
    cursor = 0
    for start, end, _, _, _ in chunks:
        if start != cursor:
            raise ValueError(f"VQAv2 slices have a gap or overlap at index {cursor}")
        cursor = end
    if cursor != args.expected_total:
        raise ValueError(f"VQAv2 slices end at {cursor}; expected {args.expected_total}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "predictions.jsonl"
    temporary_path = merged_path.with_suffix(".jsonl.tmp")
    official_rows: list[dict[str, Any]] = []
    question_ids: set[int] = set()
    score_sum = 0.0
    count = 0
    with temporary_path.open("w", encoding="utf-8") as writer:
        for _, _, directory, _, official in chunks:
            with (directory / "predictions.jsonl").open(encoding="utf-8") as reader:
                for line in reader:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    question_id = int(row["question_id"])
                    if question_id in question_ids:
                        raise ValueError(f"duplicate VQAv2 question_id={question_id}")
                    question_ids.add(question_id)
                    score_sum += float(row["score"])
                    count += 1
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            official_rows.extend(official)
    if count != args.expected_total or len(question_ids) != args.expected_total:
        raise ValueError("merged VQAv2 predictions are incomplete or duplicated")
    if len(official_rows) != args.expected_total:
        raise ValueError("merged official VQAv2 predictions are incomplete")
    official_ids = [int(row["question_id"]) for row in official_rows]
    if len(set(official_ids)) != args.expected_total:
        raise ValueError("merged official VQAv2 predictions contain duplicates")
    if set(official_ids) != question_ids:
        raise ValueError("official and diagnostic VQAv2 question IDs differ")
    os.replace(temporary_path, merged_path)
    save_json(official_rows, args.output_dir / "predictions_official.json")
    assert first_evaluation is not None
    counts = dict(first_evaluation["counts"])
    counts.update(start_index=0, end_index=count, evaluated_questions=count)
    result = {
        **{key: value for key, value in first_evaluation.items() if key != "metrics"},
        "counts": counts,
        "chunks": [{"start_index": start, "end_index": end} for start, end, _, _, _ in chunks],
        "metrics": {
            "vqav2_acc": score_sum / count,
            "score_sum": score_sum,
            "num_questions": count,
        },
    }
    save_json(result, args.output_dir / "evaluation.json")
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "num_questions", "vqav2_acc"])
        writer.writerow([args.mode, count, score_sum / count])
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=True))
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

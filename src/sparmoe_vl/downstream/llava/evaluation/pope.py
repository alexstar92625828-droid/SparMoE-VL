"""POPE hallucination evaluation used by the paper's LLaVA table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from tqdm import tqdm

from ..data import DEFAULT_POPE_ROOT, POPE_SPLITS, batched, load_pope_split
from ..metrics import parse_yes_no, pope_metrics
from ..protocol import BENCHMARK_COUNTS, BENCHMARK_IDENTITIES
from ..runtime import generate_answers, load_runtime
from .common import (
    evaluation_parser,
    result_header,
    save_json,
    validate_evaluation_args,
)


QUESTION_SUFFIX = "Answer with yes or no only."
MAX_NEW_TOKENS = 8


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = evaluation_parser("pope", __doc__ or "Evaluate POPE")
    parser.add_argument("--pope-root", type=Path, default=DEFAULT_POPE_ROOT)
    parser.add_argument("--max-samples-per-split", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    checkpoint = validate_evaluation_args(args)
    examples = {
        split: load_pope_split(
            args.pope_root,
            split,
            max_samples=args.max_samples_per_split,
        )
        for split in POPE_SPLITS
    }
    counts = {split: len(items) for split, items in examples.items()}
    counts["total_questions"] = sum(counts.values())
    counts["unique_images"] = len(
        {item.image_name for items in examples.values() for item in items}
    )
    identities = {
        key: BENCHMARK_IDENTITIES[key]
        for key in (
            "pope_random_sha256",
            "pope_popular_sha256",
            "pope_adversarial_sha256",
        )
    }
    header = result_header(
        args,
        checkpoint,
        identities,
        counts,
        max_new_tokens=MAX_NEW_TOKENS,
        question_suffix=QUESTION_SUFFIX,
    )
    if args.check_only:
        print(json.dumps(header, indent=2, ensure_ascii=True))
        return
    model, tokenizer, processor, runtime_checkpoint = load_runtime(args, args.mode)
    if runtime_checkpoint != checkpoint:
        raise RuntimeError("checkpoint changed between validation and model loading")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_split: dict[str, dict[str, float]] = {}
    for split, split_examples in examples.items():
        rows: list[dict[str, Any]] = []
        output_path = args.output_dir / f"predictions_{split}.jsonl"
        progress = tqdm(
            batched(split_examples, args.batch_size),
            total=(len(split_examples) + args.batch_size - 1) // args.batch_size,
            desc=f"{args.mode}:POPE-{split}",
        )
        with output_path.open("w", encoding="utf-8") as writer:
            for batch in progress:
                answers = generate_answers(
                    model,
                    tokenizer,
                    processor,
                    [item.image_path for item in batch],
                    [f"{item.question} {QUESTION_SUFFIX}" for item in batch],
                    device=args.device,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                for item, answer in zip(batch, answers):
                    row = {
                        "question_id": item.question_id,
                        "image": item.image_name,
                        "question": item.question,
                        "label": item.label,
                        "answer": answer,
                        "prediction": parse_yes_no(answer),
                    }
                    rows.append(row)
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_by_split[split] = pope_metrics(rows)
    average_f1 = sum(item["f1"] for item in metrics_by_split.values()) / len(POPE_SPLITS)
    average_yes = sum(item["yes_ratio"] for item in metrics_by_split.values()) / len(
        POPE_SPLITS
    )
    average_accuracy = sum(item["accuracy"] for item in metrics_by_split.values()) / len(
        POPE_SPLITS
    )
    result = {
        **header,
        "metrics": {
            "splits": metrics_by_split,
            "pope_avg_f1": average_f1,
            "pope_yes_ratio": average_yes,
            "pope_avg_accuracy": average_accuracy,
        },
    }
    if args.max_samples_per_split is None and counts != BENCHMARK_COUNTS["pope"]:
        raise RuntimeError("POPE evaluation is incomplete")
    save_json(result, args.output_dir / "evaluation.json")
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "pope_avg_f1", "pope_yes_ratio", "pope_avg_accuracy"])
        writer.writerow([args.mode, average_f1, average_yes, average_accuracy])
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=True))
    print(f"saved={args.output_dir.resolve()}")

"""MME-Perception evaluation used by the paper's LLaVA table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from tqdm import tqdm

from ..data import DEFAULT_MME_ROOT, batched, load_mme_category, validate_mme_identity
from ..metrics import mme_category_metrics, parse_yes_no
from ..protocol import (
    BENCHMARK_COUNTS,
    BENCHMARK_IDENTITIES,
    MME_CATEGORY_COUNTS,
    MME_PERCEPTION_CATEGORIES,
)
from ..runtime import generate_answers, load_runtime
from .common import (
    evaluation_parser,
    result_header,
    save_json,
    validate_evaluation_args,
)


QUESTION_SUFFIX = ""
MAX_NEW_TOKENS = 8


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = evaluation_parser("mme_p", __doc__ or "Evaluate MME-P")
    parser.add_argument("--mme-root", type=Path, default=DEFAULT_MME_ROOT)
    parser.add_argument("--max-groups-per-category", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    checkpoint = validate_evaluation_args(args)
    validate_mme_identity(args.mme_root)
    examples = {
        category: load_mme_category(
            args.mme_root,
            category,
            max_groups=args.max_groups_per_category,
        )
        for category in MME_PERCEPTION_CATEGORIES
    }
    category_counts = {
        category: {
            "groups": len({item.group_id for item in items}),
            "questions": len(items),
        }
        for category, items in examples.items()
    }
    counts = {
        "categories": len(category_counts),
        "groups": sum(item["groups"] for item in category_counts.values()),
        "questions": sum(item["questions"] for item in category_counts.values()),
    }
    header = result_header(
        args,
        checkpoint,
        {"mme_p_annotations_sha256": BENCHMARK_IDENTITIES["mme_p_annotations_sha256"]},
        counts,
        max_new_tokens=MAX_NEW_TOKENS,
        question_suffix=QUESTION_SUFFIX,
    )
    header["category_counts"] = category_counts
    if args.check_only:
        print(json.dumps(header, indent=2, ensure_ascii=True))
        return
    model, tokenizer, processor, runtime_checkpoint = load_runtime(args, args.mode)
    if runtime_checkpoint != checkpoint:
        raise RuntimeError("checkpoint changed between validation and model loading")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_category: dict[str, dict[str, float]] = {}
    for category, category_examples in examples.items():
        rows: list[dict[str, Any]] = []
        output_path = args.output_dir / f"predictions_{category}.jsonl"
        progress = tqdm(
            batched(category_examples, args.batch_size),
            total=(len(category_examples) + args.batch_size - 1) // args.batch_size,
            desc=f"{args.mode}:MME-{category}",
        )
        with output_path.open("w", encoding="utf-8") as writer:
            for batch in progress:
                answers = generate_answers(
                    model,
                    tokenizer,
                    processor,
                    [item.image_path for item in batch],
                    [item.question for item in batch],
                    device=args.device,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                for item, answer in zip(batch, answers):
                    row = {
                        "category": item.category,
                        "group_id": item.group_id,
                        "question_id": item.question_id,
                        "image": str(item.image_path),
                        "question": item.question,
                        "label": item.label,
                        "answer": answer,
                        "prediction": parse_yes_no(answer),
                    }
                    rows.append(row)
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_by_category[category] = mme_category_metrics(rows)
    score = sum(item["score"] for item in metrics_by_category.values())
    question_count = sum(item["n_questions"] for item in metrics_by_category.values())
    unknown_ratio = sum(
        item["unknown_ratio"] * item["n_questions"] for item in metrics_by_category.values()
    ) / max(question_count, 1)
    result = {
        **header,
        "metrics": {
            "categories": metrics_by_category,
            "mme_p_score": score,
            "unknown_ratio": unknown_ratio,
            "num_categories": len(metrics_by_category),
        },
    }
    if args.max_groups_per_category is None:
        if category_counts != MME_CATEGORY_COUNTS or counts != BENCHMARK_COUNTS["mme_p"]:
            raise RuntimeError("MME-P evaluation is incomplete")
    save_json(result, args.output_dir / "evaluation.json")
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "mme_p_score", "unknown_ratio", "num_categories"])
        writer.writerow([args.mode, score, unknown_ratio, len(metrics_by_category)])
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=True))
    print(f"saved={args.output_dir.resolve()}")

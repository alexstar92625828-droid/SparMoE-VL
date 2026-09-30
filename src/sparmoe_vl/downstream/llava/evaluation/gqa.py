"""GQA testdev-balanced evaluation used by the paper's LLaVA study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from tqdm import tqdm

from ..data import DEFAULT_GQA_ROOT, GQA_QUESTIONS, batched, load_gqa
from ..metrics import normalize_gqa_answer
from ..protocol import BENCHMARK_COUNTS, BENCHMARK_IDENTITIES
from ..runtime import generate_answers, load_runtime
from .common import evaluation_parser, result_header, save_json, validate_evaluation_args


QUESTION_SUFFIX = "Answer with a single word or short phrase."
MAX_NEW_TOKENS = 12


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = evaluation_parser("gqa", __doc__ or "Evaluate GQA")
    parser.add_argument("--gqa-root", type=Path, default=DEFAULT_GQA_ROOT)
    parser.add_argument("--question-file", default=GQA_QUESTIONS)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    checkpoint = validate_evaluation_args(args)
    examples = load_gqa(
        args.gqa_root,
        question_file=args.question_file,
        max_samples=args.max_samples,
    )
    counts = {"questions": len(examples)}
    header = result_header(
        args,
        checkpoint,
        {"gqa_questions_sha256": BENCHMARK_IDENTITIES["gqa_questions_sha256"]},
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
    prediction_path = args.output_dir / "predictions.jsonl"
    official_predictions: list[dict[str, str]] = []
    correct = 0
    rows = 0
    progress = tqdm(
        batched(examples, args.batch_size),
        total=(len(examples) + args.batch_size - 1) // args.batch_size,
        desc=f"{args.mode}:GQA",
    )
    with prediction_path.open("w", encoding="utf-8") as writer:
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
                prediction = normalize_gqa_answer(answer)
                reference = normalize_gqa_answer(item.answer)
                is_correct = prediction == reference
                row: dict[str, Any] = {
                    "question_id": item.question_id,
                    "image_id": item.image_id,
                    "question": item.question,
                    "answer": reference,
                    "raw_answer": answer,
                    "prediction": prediction,
                    "correct": is_correct,
                }
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                official_predictions.append(
                    {"questionId": item.question_id, "prediction": prediction}
                )
                rows += 1
                correct += int(is_correct)
            progress.set_postfix(acc=correct / max(rows, 1))
    accuracy = correct / max(rows, 1)
    result = {
        **header,
        "metrics": {
            "gqa_acc": accuracy,
            "correct": correct,
            "num_questions": rows,
        },
    }
    if args.max_samples is None and counts != BENCHMARK_COUNTS["gqa"]:
        raise RuntimeError("GQA evaluation is incomplete")
    save_json(result, args.output_dir / "evaluation.json")
    save_json(official_predictions, args.output_dir / "predictions_official.json")
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "num_questions", "gqa_acc"])
        writer.writerow([args.mode, rows, accuracy])
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=True))
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

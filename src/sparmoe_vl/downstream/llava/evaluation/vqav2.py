"""VQAv2 validation evaluation used by the paper's LLaVA study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from tqdm import tqdm

from ..data import (
    DEFAULT_COCO_VAL2014,
    DEFAULT_VQAV2_ROOT,
    VQAV2_ANNOTATIONS,
    VQAV2_QUESTIONS,
    batched,
    load_vqav2,
)
from ..metrics import normalize_vqa_answer, vqav2_score
from ..protocol import BENCHMARK_COUNTS, BENCHMARK_IDENTITIES
from ..runtime import generate_answers, load_runtime
from .common import evaluation_parser, result_header, save_json, validate_evaluation_args


QUESTION_SUFFIX = "Answer with a single word or short phrase."
MAX_NEW_TOKENS = 12


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = evaluation_parser("vqav2", __doc__ or "Evaluate VQAv2")
    parser.add_argument("--vqav2-root", type=Path, default=DEFAULT_VQAV2_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_COCO_VAL2014)
    parser.add_argument("--question-file", default=VQAV2_QUESTIONS)
    parser.add_argument("--annotation-file", default=VQAV2_ANNOTATIONS)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    checkpoint = validate_evaluation_args(args)
    examples = load_vqav2(
        args.vqav2_root,
        args.image_root,
        question_file=args.question_file,
        annotation_file=args.annotation_file,
        start_index=args.start_index,
        end_index=args.end_index,
        max_samples=args.max_samples,
    )
    actual_end = args.start_index + len(examples)
    counts = {
        "dataset_questions": BENCHMARK_COUNTS["vqav2"]["questions"],
        "dataset_annotations": BENCHMARK_COUNTS["vqav2"]["annotations"],
        "start_index": args.start_index,
        "end_index": actual_end,
        "evaluated_questions": len(examples),
    }
    identities = {
        "vqav2_questions_sha256": BENCHMARK_IDENTITIES["vqav2_questions_sha256"],
        "vqav2_annotations_sha256": BENCHMARK_IDENTITIES["vqav2_annotations_sha256"],
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
    prediction_path = args.output_dir / "predictions.jsonl"
    official_predictions: list[dict[str, Any]] = []
    score_sum = 0.0
    rows = 0
    progress = tqdm(
        batched(examples, args.batch_size),
        total=(len(examples) + args.batch_size - 1) // args.batch_size,
        desc=f"{args.mode}:VQAv2[{args.start_index}:{actual_end}]",
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
                prediction = normalize_vqa_answer(answer)
                score = vqav2_score(answer, item.answers)
                row = {
                    "question_id": item.question_id,
                    "image_id": item.image_id,
                    "question": item.question,
                    "multiple_choice_answer": normalize_vqa_answer(item.multiple_choice_answer),
                    "answers": list(item.answers),
                    "raw_answer": answer,
                    "prediction": prediction,
                    "score": score,
                }
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                official_predictions.append(
                    {"question_id": int(item.question_id), "answer": prediction}
                )
                rows += 1
                score_sum += score
            progress.set_postfix(acc=score_sum / max(rows, 1))
    accuracy = score_sum / max(rows, 1)
    result = {
        **header,
        "metrics": {
            "vqav2_acc": accuracy,
            "score_sum": score_sum,
            "num_questions": rows,
        },
    }
    full_run = args.start_index == 0 and args.end_index is None and args.max_samples is None
    if full_run and rows != BENCHMARK_COUNTS["vqav2"]["questions"]:
        raise RuntimeError("VQAv2 evaluation is incomplete")
    save_json(result, args.output_dir / "evaluation.json")
    save_json(official_predictions, args.output_dir / "predictions_official.json")
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "start_index", "end_index", "num_questions", "vqav2_acc"])
        writer.writerow([args.mode, args.start_index, actual_end, rows, accuracy])
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=True))
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

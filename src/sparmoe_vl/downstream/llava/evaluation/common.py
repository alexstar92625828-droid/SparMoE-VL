"""Shared validation and output helpers for LLaVA benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ..protocol import STUDY_NAME, inspect_checkpoint
from ..runtime import add_runtime_arguments, validate_runtime_paths


def evaluation_parser(benchmark: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_runtime_arguments(parser, sparse_only=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.set_defaults(benchmark=benchmark)
    return parser


def validate_evaluation_args(args: argparse.Namespace) -> dict[str, Any] | None:
    validate_runtime_paths(args, args.mode)
    if args.mode == "sparse":
        return inspect_checkpoint(args.checkpoint, expected_stage=2)
    return None


def result_header(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
    data_identity: dict[str, str],
    counts: dict[str, Any],
    *,
    max_new_tokens: int,
    question_suffix: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "benchmark": args.benchmark,
        "mode": args.mode,
        "checkpoint": checkpoint,
        "data_identity": data_identity,
        "counts": counts,
        "generation": {
            "do_sample": False,
            "batch_size": args.batch_size,
            "max_new_tokens": max_new_tokens,
            "question_suffix": question_suffix,
        },
        "model": {
            "llava": str(args.llava_model.resolve()),
            "tokenizer": str(args.tokenizer_path.resolve()),
            "vision_tower": str(args.clip_path.resolve()),
            "vision_select_layer": -2,
            "vision_select_feature": "patch",
        },
    }


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)

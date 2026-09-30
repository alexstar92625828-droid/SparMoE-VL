#!/usr/bin/env python3
"""Score one visual OPTIN candidate shard on all 500k main-pool pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.vision.common import (
    ANNOTATIONS,
    EXPECTED_IMAGE_POOL_SHA256,
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_VISION_CAPTION_POOL_SHA256,
    IMAGE_ROOT,
    POOL_SIZE,
    PRETRAINED,
    build_main_image_text_pool,
    create_clip,
    file_sha256,
    save_json,
    seed_everything,
)
from sparmoe_vl.baselines.vision.optin import (
    MANIFOLD_SAMPLE_K,
    METHOD,
    TOTAL_CANDIDATES,
    candidate_partition,
    full_calibration_manifest,
    save_score_shard,
    search_full_pool_score_shard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 123, 2026))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-candidate-evaluations",
        type=int,
        help="diagnostic stop only; incomplete state cannot emit a score shard",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 32:
        parser.error("32 is the per-batch size; the released run still consumes all 500,000")
    if args.candidate_batch <= 0 or args.workers < 0:
        parser.error("candidate batch must be positive and workers non-negative")
    if args.max_candidate_evaluations is not None and args.max_candidate_evaluations <= 0:
        parser.error("--max-candidate-evaluations must be positive")
    try:
        args.candidate_start, args.candidate_end = candidate_partition(
            args.shard_index, args.num_shards
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    configuration = {
        "method": METHOD,
        "seed": args.seed,
        "data_seed": 42,
        "samples": POOL_SIZE,
        "batch_size": args.batch_size,
        "batches": POOL_SIZE // args.batch_size,
        "image_pool_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "paired_caption_sha256": EXPECTED_VISION_CAPTION_POOL_SHA256,
        "total_candidates": TOTAL_CANDIDATES,
        "candidate_start": args.candidate_start,
        "candidate_end": args.candidate_end,
        "candidate_batch": args.candidate_batch,
        "manifold_sample_k": MANIFOLD_SAMPLE_K,
        "require_complete_main_pool": True,
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.check_only:
        print(json.dumps(configuration, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing full-pool visual OPTIN on CPU")
    if file_sha256(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")

    seed_everything(args.seed)
    paths, captions, pool = build_main_image_text_pool(args.annotations, args.image_root)
    order, manifest = full_calibration_manifest(args.seed, args.batch_size)
    for key in (
        "data_seed",
        "pool_size",
        "uses_complete_main_pool",
        "dataset_sha256",
        "paired_caption_sha256",
    ):
        if pool[key] != manifest[key]:
            raise RuntimeError(f"OPTIN paired pool metadata mismatch: {key}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, args.output_dir / "data_manifest.json")

    model, preprocess, tokenizer = create_clip(args.device, args.pretrained)
    raw_mmd, raw_kl, report = search_full_pool_score_shard(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        paths=paths,
        captions=captions,
        order=order,
        manifest=manifest,
        candidate_start=args.candidate_start,
        candidate_end=args.candidate_end,
        device=args.device,
        output_dir=args.output_dir,
        workers=args.workers,
        candidate_batch=args.candidate_batch,
        resume=args.resume,
        save_every=args.save_every,
        log_every=args.log_every,
        max_candidate_evaluations=args.max_candidate_evaluations,
    )
    save_json(report, args.output_dir / "search_report.json")
    if not report["complete"]:
        print(json.dumps(report, indent=2))
        print("incomplete diagnostic/search run; no OPTIN score shard emitted", flush=True)
        return
    if raw_mmd is None or raw_kl is None:
        raise AssertionError("complete visual OPTIN search did not return scores")
    shard_path = args.output_dir / "optin_score_shard.pt"
    save_score_shard(
        shard_path,
        raw_mmd,
        raw_kl,
        manifest,
        args.candidate_start,
        args.candidate_end,
    )
    print(json.dumps({**report, "score_shard": str(shard_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()

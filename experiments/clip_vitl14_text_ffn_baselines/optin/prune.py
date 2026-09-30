#!/usr/bin/env python3
"""Score and structurally prune text FFNs with OPTIN on all 500k samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.text.common import (
    ANNOTATIONS,
    EXPECTED_PAIRED_PATH_POOL_SHA256,
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_TEXT_POOL_SHA256,
    IMAGE_FEATURE_CACHE,
    POOL_SIZE,
    PRETRAINED,
    TOKEN_CACHE,
    create_clip,
    file_sha256,
    prepare_token_cache,
    save_json,
    seed_everything,
    validate_image_feature_cache,
)
from sparmoe_vl.baselines.text.optin import (
    MANIFOLD_SAMPLE_K,
    METHOD,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    PAPER,
    full_calibration_manifest,
    save_final_checkpoint,
    search_full_pool_scores,
)


TARGETS = {42: 0.441, 123: 0.450, 2026: 0.462}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=tuple(TARGETS))
    parser.add_argument("--target-ffn-reduction", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--token-cache", type=Path, default=TOKEN_CACHE)
    parser.add_argument("--image-feature-cache", type=Path, default=IMAGE_FEATURE_CACHE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-candidate-evaluations",
        type=int,
        help="diagnostic stop only; an incomplete run cannot emit a checkpoint",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    expected = TARGETS[args.seed]
    if args.target_ffn_reduction is None:
        args.target_ffn_reduction = expected
    if abs(args.target_ffn_reduction - expected) > 1e-10:
        parser.error(f"seed {args.seed} fixes target reduction to {expected}")
    if args.batch_size != 32:
        parser.error("the released OPTIN protocol fixes batch size to 32")
    if args.max_candidate_evaluations is not None and args.max_candidate_evaluations <= 0:
        parser.error("--max-candidate-evaluations must be positive")
    return args


def main() -> None:
    args = parse_args()
    configuration = {
        "method": METHOD,
        "seed": args.seed,
        "data_seed": 42,
        "samples": POOL_SIZE,
        "batches": POOL_SIZE // args.batch_size,
        "batch_size": args.batch_size,
        "text_pool_sha256": EXPECTED_TEXT_POOL_SHA256,
        "paired_path_pool_sha256": EXPECTED_PAIRED_PATH_POOL_SHA256,
        "target_ffn_reduction": args.target_ffn_reduction,
        "output_dir": str(args.output_dir.resolve()),
        "require_complete_main_pool": True,
    }
    if args.check_only:
        print(json.dumps(configuration, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing full-pool OPTIN on CPU")
    if file_sha256(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")
    if not args.image_feature_cache.is_file():
        raise FileNotFoundError(
            f"missing {args.image_feature_cache}; run optin/prepare_data.py first"
        )

    seed_everything(args.seed)
    token_cache = prepare_token_cache(args.token_cache, args.annotations)
    image_cache = torch.load(
        args.image_feature_cache,
        map_location="cpu",
        weights_only=False,
    )
    validate_image_feature_cache(image_cache)
    order, manifest = full_calibration_manifest(args.seed, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, args.output_dir / "data_manifest.json")

    model, _, _ = create_clip(args.device, args.pretrained)
    raw_mmd, raw_kl, report = search_full_pool_scores(
        model=model,
        tokens=token_cache["tokens"],
        image_features=image_cache["features"],
        order=order,
        manifest=manifest,
        device=args.device,
        output_dir=args.output_dir,
        resume=args.resume,
        save_every=args.save_every,
        log_every=args.log_every,
        max_candidate_evaluations=args.max_candidate_evaluations,
    )
    save_json(report, args.output_dir / "search_report.json")
    if not report["complete"]:
        print(json.dumps(report, indent=2))
        print("incomplete diagnostic/search run; no OPTIN checkpoint emitted", flush=True)
        return
    if raw_mmd is None or raw_kl is None:
        raise AssertionError("complete search did not return scores")

    checkpoint_path, score_path, checkpoint = save_final_checkpoint(
        model=model,
        raw_mmd=raw_mmd,
        raw_kl=raw_kl,
        manifest=manifest,
        target_reduction=args.target_ffn_reduction,
        pretrained_sha256=EXPECTED_PRETRAINED_SHA256,
        output_dir=args.output_dir,
    )
    with torch.inference_mode():
        feature = model.encode_text(token_cache["tokens"][:2].to(args.device), normalize=True)
    if tuple(feature.shape) != (2, 768) or not torch.isfinite(feature).all():
        raise RuntimeError("physically pruned OPTIN text tower failed its forward check")

    protocol = {
        "method": METHOD,
        "paper": PAPER,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "initialization": "Dense OpenCLIP ViT-L/14",
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "compressed_modality": "text only",
        "compressed_modules": "12 text FFN hidden dimensions",
        "attention_pruned": False,
        "tokens_pruned": False,
        "visual_tower_pruned": False,
        "weight_updates": 0,
        "score": "downstream c_proj trajectory MMD plus symmetric CLIP-output KL",
        "manifold_sample_k": MANIFOLD_SAMPLE_K,
        "allocation": "global importance ranking over all text FFN neurons",
        "physical_pruning": "slice c_fc rows/bias and c_proj columns",
        "release_data_policy": "all 500,000 main-experiment paired samples",
        "data_manifest": manifest,
        "target_ffn_reduction": args.target_ffn_reduction,
        "hidden_sizes": checkpoint["hidden_sizes"],
        "statistics": checkpoint["statistics"],
        "checkpoint": str(checkpoint_path.resolve()),
        "scores": str(score_path.resolve()),
    }
    save_json(protocol, args.output_dir / "protocol.json")
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()

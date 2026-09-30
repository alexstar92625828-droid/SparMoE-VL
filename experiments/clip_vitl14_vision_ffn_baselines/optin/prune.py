#!/usr/bin/env python3
"""Merge complete visual OPTIN score shards and structurally prune the FFNs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.vision.common import (
    EXPECTED_IMAGE_POOL_SHA256,
    EXPECTED_PRETRAINED_SHA256,
    POOL_SIZE,
    PRETRAINED,
    create_clip,
    file_sha256,
    save_json,
    seed_everything,
)
from sparmoe_vl.baselines.vision.optin import (
    MANIFOLD_SAMPLE_K,
    METHOD,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    PAPER,
    merge_score_shards,
    save_final_checkpoint,
)


TARGETS = {42: 0.348, 123: 0.354, 2026: 0.361}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=tuple(TARGETS))
    parser.add_argument("--target-ffn-reduction", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    target = TARGETS[args.seed]
    if args.target_ffn_reduction is None:
        args.target_ffn_reduction = target
    if abs(args.target_ffn_reduction - target) > 1e-10:
        parser.error(f"seed {args.seed} fixes target reduction to {target}")
    return args


def main() -> None:
    args = parse_args()
    configuration = {
        "method": METHOD,
        "seed": args.seed,
        "samples": POOL_SIZE,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "target_ffn_reduction": args.target_ffn_reduction,
        "score_root": str(args.score_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "require_complete_candidate_coverage": True,
        "require_complete_main_pool_per_shard": True,
    }
    if args.check_only:
        print(json.dumps(configuration, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing visual OPTIN pruning")
    pretrained_sha256 = file_sha256(args.pretrained)
    if pretrained_sha256 != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")

    shard_paths = sorted(args.score_root.rglob("optin_score_shard.pt"))
    raw_mmd, raw_kl, manifest = merge_score_shards(shard_paths)
    if int(manifest["processing_seed"]) != args.seed:
        raise ValueError("OPTIN score shards belong to another seed")
    seed_everything(args.seed)
    model, _, _ = create_clip(args.device, args.pretrained)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, score_path, checkpoint = save_final_checkpoint(
        model=model,
        raw_mmd=raw_mmd,
        raw_kl=raw_kl,
        manifest=manifest,
        target_reduction=args.target_ffn_reduction,
        pretrained_sha256=pretrained_sha256,
        output_dir=args.output_dir,
    )
    with torch.inference_mode():
        feature = model.encode_image(torch.zeros(1, 3, 224, 224, device=args.device))
    if tuple(feature.shape) != (1, 768) or not torch.isfinite(feature).all():
        raise RuntimeError("physically pruned OPTIN visual tower failed its forward check")

    protocol = {
        "method": METHOD,
        "paper": PAPER,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "initialization": "Dense OpenCLIP ViT-L/14",
        "pretrained_sha256": pretrained_sha256,
        "compressed_modality": "vision only",
        "compressed_modules": "24 visual FFN hidden dimensions",
        "attention_pruned": False,
        "tokens_pruned": False,
        "text_tower_pruned": False,
        "weight_updates": 0,
        "score": "downstream ln_1 trajectory MMD plus symmetric CLIP-output KL",
        "manifold_sample_k": MANIFOLD_SAMPLE_K,
        "allocation": "global importance ranking over all visual FFN neurons",
        "physical_pruning": "slice c_fc rows/bias and c_proj columns",
        "release_data_policy": "all 500,000 visual main-experiment samples",
        "data_manifest": manifest,
        "target_ffn_reduction": args.target_ffn_reduction,
        "hidden_sizes": checkpoint["hidden_sizes"],
        "statistics": checkpoint["statistics"],
        "checkpoint": str(checkpoint_path.resolve()),
        "scores": str(score_path.resolve()),
    }
    save_json(protocol, args.output_dir / "protocol.json")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()

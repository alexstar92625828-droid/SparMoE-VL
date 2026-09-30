#!/usr/bin/env python3
"""Prepare the exact 500k paired main-experiment pool for OPTIN."""

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
    IMAGE_ROOT,
    POOL_SIZE,
    PRETRAINED,
    TOKEN_CACHE,
    create_clip,
    file_sha256,
    prepare_image_feature_cache,
    prepare_token_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--token-cache", type=Path, default=TOKEN_CACHE)
    parser.add_argument("--image-feature-cache", type=Path, default=IMAGE_FEATURE_CACHE)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configuration = {
        "samples": POOL_SIZE,
        "data_seed": 42,
        "text_pool_sha256": EXPECTED_TEXT_POOL_SHA256,
        "paired_path_pool_sha256": EXPECTED_PAIRED_PATH_POOL_SHA256,
        "token_cache": str(args.token_cache.resolve()),
        "image_feature_cache": str(args.image_feature_cache.resolve()),
        "require_complete_main_pool": True,
    }
    if args.check_only:
        print(json.dumps(configuration, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing 500k image encoding on CPU")
    if file_sha256(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")

    tokens = prepare_token_cache(args.token_cache, args.annotations)
    model, preprocess, _ = create_clip(args.device, args.pretrained)
    images = prepare_image_feature_cache(
        model=model,
        preprocess=preprocess,
        device=args.device,
        cache_path=args.image_feature_cache,
        annotations=args.annotations,
        image_root=args.image_root,
        batch_size=args.image_batch_size,
        workers=args.workers,
    )
    configuration["token_shape"] = list(tokens["tokens"].shape)
    configuration["image_feature_shape"] = list(images["features"].shape)
    configuration["complete"] = True
    print(json.dumps(configuration, indent=2))


if __name__ == "__main__":
    main()

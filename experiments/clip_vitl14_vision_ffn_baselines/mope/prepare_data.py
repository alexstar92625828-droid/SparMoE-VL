#!/usr/bin/env python3
"""Build the exact 500k visual-main pool and MoPE text-feature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.vision.common import (
    ANNOTATIONS,
    EXPECTED_PRETRAINED_SHA256,
    IMAGE_ROOT,
    PRETRAINED,
    build_main_image_text_pool,
    create_clip,
    file_sha256,
    save_json,
)
from sparmoe_vl.baselines.vision.mope import (
    METHOD,
    TEXT_FEATURE_CACHE,
    prepare_dense_text_cache,
    structure_data_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--text-feature-cache", type=Path, default=TEXT_FEATURE_CACHE)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.text_batch_size <= 0:
        parser.error("--text-batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    manifest = {
        "method": METHOD,
        "data_manifest": structure_data_manifest(),
        "text_feature_cache": str(args.text_feature_cache.resolve()),
    }
    if args.check_only:
        print(json.dumps(manifest, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing full MoPE cache construction")
    pretrained_sha256 = file_sha256(args.pretrained)
    if pretrained_sha256 != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")
    _paths, captions, pool_manifest = build_main_image_text_pool(
        args.annotations,
        args.image_root,
    )
    model, _preprocess, tokenizer = create_clip(args.device, args.pretrained)
    prepare_dense_text_cache(
        model,
        tokenizer,
        captions,
        args.device,
        args.text_feature_cache,
        args.text_batch_size,
    )
    manifest["pretrained_sha256"] = pretrained_sha256
    manifest["scanned_pool"] = pool_manifest
    save_json(manifest, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

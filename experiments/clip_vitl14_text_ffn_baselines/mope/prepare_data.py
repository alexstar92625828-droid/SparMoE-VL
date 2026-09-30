#!/usr/bin/env python3
"""Prepare full-main-pool feature caches required by MoPE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.text.common import (
    ANNOTATIONS,
    EXPECTED_PRETRAINED_SHA256,
    IMAGE_FEATURE_CACHE,
    IMAGE_ROOT,
    PRETRAINED,
    TOKEN_CACHE,
    create_clip,
    file_sha256,
    prepare_image_feature_cache,
    prepare_token_cache,
    save_json,
)
from sparmoe_vl.baselines.text.mope import (
    MOPE_DENSE_TEXT_CACHE,
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
    parser.add_argument("--token-cache", type=Path, default=TOKEN_CACHE)
    parser.add_argument("--image-feature-cache", type=Path, default=IMAGE_FEATURE_CACHE)
    parser.add_argument("--text-feature-cache", type=Path, default=MOPE_DENSE_TEXT_CACHE)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "method": "MoPE-CLIP-FFN",
        **structure_data_manifest(),
        "token_cache": str(args.token_cache.resolve()),
        "image_feature_cache": str(args.image_feature_cache.resolve()),
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

    token_cache = prepare_token_cache(args.token_cache, args.annotations)
    model, preprocess, _ = create_clip(args.device, args.pretrained)
    prepare_image_feature_cache(
        model,
        preprocess,
        args.device,
        args.image_feature_cache,
        args.annotations,
        args.image_root,
        args.image_batch_size,
        args.workers,
    )
    prepare_dense_text_cache(
        model,
        token_cache["tokens"],
        args.device,
        pretrained_sha256,
        args.text_feature_cache,
        args.text_batch_size,
    )
    manifest["pretrained_sha256"] = pretrained_sha256
    save_json(manifest, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate one complete-pool OPTIN text checkpoint on COCO retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.text.common import (
    COCO_ANNOTATIONS,
    COCO_IMAGES,
    EXPECTED_PRETRAINED_SHA256,
    PRETRAINED,
    create_clip,
    encode_images,
    encode_texts,
    file_sha256,
    prepare_coco,
    retrieval_metrics,
    save_json,
    seed_everything,
)
from sparmoe_vl.baselines.text.optin import (
    METHOD,
    apply_structural_pruning,
    ffn_statistics,
    load_text_state_dict,
    validate_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--coco-annotations", type=Path, default=COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=COCO_IMAGES)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing accidental CPU evaluation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint)
    if checkpoint.get("pretrained_sha256") != EXPECTED_PRETRAINED_SHA256:
        raise ValueError("OPTIN checkpoint records the wrong Dense CLIP")
    if file_sha256(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise ValueError("current Dense CLIP differs from OPTIN initialization")

    seed = int(checkpoint["seed"])
    seed_everything(seed)
    model, preprocess, tokenizer = create_clip(args.device, args.pretrained)
    coco = prepare_coco(args.coco_annotations, args.coco_images)
    image_features = encode_images(
        model,
        preprocess,
        coco.image_paths,
        args.device,
        args.image_batch_size,
        args.workers,
        f"COCO dense images for OPTIN seed {seed}",
    )
    apply_structural_pruning(model, checkpoint["kept_indices"])
    load_text_state_dict(model, checkpoint["text_state_dict"])
    text_features = encode_texts(
        model,
        tokenizer,
        coco.captions,
        args.device,
        args.text_batch_size,
        f"COCO OPTIN texts seed {seed}",
    )
    metrics = retrieval_metrics(
        image_features,
        text_features,
        coco.caption_image_indices,
    )
    statistics = ffn_statistics(checkpoint["hidden_sizes"])
    result = {
        "method": METHOD,
        "seed": seed,
        "data_seed": checkpoint["data_manifest"]["data_seed"],
        "samples": checkpoint["data_manifest"]["selected_samples"],
        "text_pool_sha256": checkpoint["data_manifest"]["text_pool_sha256"],
        "paired_path_pool_sha256": checkpoint["data_manifest"]["paired_path_pool_sha256"],
        "checkpoint": str(args.checkpoint.resolve()),
        "hidden_sizes": checkpoint["hidden_sizes"],
        **statistics,
        "metrics": {"coco": metrics},
        "counts": {"coco": {"images": len(coco.image_paths), "texts": len(coco.captions)}},
    }
    save_json(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

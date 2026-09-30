#!/usr/bin/env python3
"""Evaluate one complete-pool visual FLAP checkpoint on retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.retrieval import (
    COCO_ANNOTATIONS,
    COCO_IMAGES,
    FLICKR_ANNOTATIONS,
    FLICKR_IMAGES,
    encode_images,
    encode_texts,
    prepare_coco,
    prepare_flickr30k,
    retrieval_metrics,
)
from sparmoe_vl.baselines.vision.common import (
    EXPECTED_PRETRAINED_SHA256,
    PRETRAINED,
    create_clip,
    file_sha256,
    save_json,
    seed_everything,
)
from sparmoe_vl.baselines.vision.flap import (
    DENSE_FFN_MACS_G,
    DENSE_TOTAL_MACS_G,
    DENSE_VISUAL_PARAMETERS,
    METHOD,
    NON_FFN_MACS_G,
    apply_checkpoint_structure,
    ffn_statistics,
    load_visual_state_dict,
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
    parser.add_argument("--flickr-annotations", type=Path, default=FLICKR_ANNOTATIONS)
    parser.add_argument("--flickr-images", type=Path, default=FLICKR_IMAGES)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing accidental CPU evaluation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint)
    if checkpoint.get("pretrained_sha256") != EXPECTED_PRETRAINED_SHA256:
        raise ValueError("FLAP checkpoint records the wrong Dense CLIP")
    if file_sha256(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise ValueError("current Dense CLIP differs from FLAP initialization")

    seed = int(checkpoint["seed"])
    seed_everything(seed)
    model, preprocess, tokenizer = create_clip(args.device, args.pretrained)
    apply_checkpoint_structure(model, checkpoint["kept_indices"])
    load_visual_state_dict(model, checkpoint["visual_state_dict"])
    datasets = {
        "coco": prepare_coco(args.coco_annotations, args.coco_images),
        "flickr30k": prepare_flickr30k(args.flickr_annotations, args.flickr_images),
    }
    metrics, counts = {}, {}
    for name, dataset in datasets.items():
        counts[name] = {
            "images": len(dataset.image_paths),
            "texts": len(dataset.captions),
        }
        text_features = encode_texts(
            model,
            tokenizer,
            dataset.captions,
            args.device,
            args.text_batch_size,
            f"{name} dense texts",
        )
        image_features = encode_images(
            model,
            preprocess,
            dataset.image_paths,
            args.device,
            args.image_batch_size,
            args.workers,
            f"{name} FLAP images seed {seed}",
        )
        metrics[name] = retrieval_metrics(
            image_features,
            text_features,
            dataset.caption_image_indices,
        )

    manifest = checkpoint["data_manifest"]
    statistics = ffn_statistics(checkpoint["hidden_sizes"])
    result = {
        "method": METHOD,
        "seed": seed,
        "data_seed": manifest["data_seed"],
        "unique_samples": manifest["unique_samples"],
        "calibration_exposures": manifest["calibration_exposures"],
        "dataset_sha256": manifest["dataset_sha256"],
        "weight_updates": 0,
        "checkpoint": str(args.checkpoint.resolve()),
        "hidden_sizes": checkpoint["hidden_sizes"],
        "active_visual_parameters_m": statistics["active_visual_parameters_m"],
        "dense_visual_parameters_m": DENSE_VISUAL_PARAMETERS / 1e6,
        "macs_vision_g": statistics["visual_total_macs_g"],
        "ffn_macs_vision_g": statistics["visual_ffn_macs_g"],
        "dense_total_macs_vision_g": DENSE_TOTAL_MACS_G,
        "dense_ffn_macs_vision_g": DENSE_FFN_MACS_G,
        "non_ffn_macs_vision_g": NON_FFN_MACS_G,
        "ffn_macs_reduction_percent": statistics["ffn_reduction_percent"],
        "metrics": metrics,
        "counts": counts,
    }
    save_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

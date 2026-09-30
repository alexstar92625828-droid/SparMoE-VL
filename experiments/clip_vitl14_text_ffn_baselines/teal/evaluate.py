#!/usr/bin/env python3
"""Evaluate one complete-pool TEAL calibration on COCO and Flickr30k."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.text.common import (
    COCO_ANNOTATIONS,
    COCO_IMAGES,
    DENSE_FFN_MACS_G,
    DENSE_TOTAL_MACS_G,
    DENSE_TRANSFORMER_PARAMETERS,
    EXPECTED_PRETRAINED_SHA256,
    FLICKR_ANNOTATIONS,
    FLICKR_IMAGES,
    NON_FFN_MACS_G,
    PRETRAINED,
    create_clip,
    encode_images,
    encode_texts,
    file_sha256,
    prepare_coco,
    prepare_flickr30k,
    retrieval_metrics,
    save_json,
    seed_everything,
)
from sparmoe_vl.baselines.text.teal import (
    METHOD,
    activity_statistics,
    install_teal_text,
    reset_activity,
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
        raise ValueError("unexpected Dense CLIP SHA-256 in TEAL checkpoint")
    if file_sha256(args.pretrained) != checkpoint["pretrained_sha256"]:
        raise ValueError("current Dense CLIP checkpoint differs from calibration")

    seed = int(checkpoint["calibration_manifest"]["processing_seed"])
    seed_everything(seed)
    model, preprocess, tokenizer = create_clip(args.device, args.pretrained)
    wrappers = install_teal_text(model, checkpoint)
    model.eval()
    coco = prepare_coco(args.coco_annotations, args.coco_images)
    flickr = prepare_flickr30k(args.flickr_annotations, args.flickr_images)
    coco_images = encode_images(
        model,
        preprocess,
        coco.image_paths,
        args.device,
        args.image_batch_size,
        args.workers,
        "COCO dense images",
    )
    flickr_images = encode_images(
        model,
        preprocess,
        flickr.image_paths,
        args.device,
        args.image_batch_size,
        args.workers,
        "Flickr30k dense images",
    )

    metrics = {}
    activity = {}
    for name, dataset, image_features in (
        ("coco", coco, coco_images),
        ("flickr30k", flickr, flickr_images),
    ):
        reset_activity(wrappers)
        text_features = encode_texts(
            model,
            tokenizer,
            dataset.captions,
            args.device,
            args.text_batch_size,
            f"{name} TEAL texts seed {seed}",
        )
        metrics[name] = retrieval_metrics(
            image_features,
            text_features,
            dataset.caption_image_indices,
        )
        activity[name] = activity_statistics(wrappers).as_dict()

    primary = activity["coco"]
    result = {
        "method": METHOD,
        "checkpoint": str(args.checkpoint.resolve()),
        "data_seed": checkpoint["calibration_manifest"]["data_seed"],
        "dataset_sha256": checkpoint["calibration_manifest"]["pool_sha256"],
        "samples": checkpoint["calibration_manifest"]["selected_samples"],
        "processing_seed": seed,
        "weight_updates": 0,
        "counts": {
            "coco": {"images": len(coco.image_paths), "texts": len(coco.captions)},
            "flickr30k": {
                "images": len(flickr.image_paths),
                "texts": len(flickr.captions),
            },
        },
        "active_text_parameters_m": primary["active_text_parameters_m"],
        "dense_text_parameters_m": DENSE_TRANSFORMER_PARAMETERS / 1e6,
        "macs_text_g": primary["macs_text_g"],
        "ffn_macs_text_g": primary["ffn_macs_text_g"],
        "dense_total_macs_text_g": DENSE_TOTAL_MACS_G,
        "dense_ffn_macs_text_g": DENSE_FFN_MACS_G,
        "non_ffn_macs_text_g": NON_FFN_MACS_G,
        "ffn_macs_reduction_percent": primary["ffn_macs_reduction_percent"],
        "metrics": metrics,
        "activity_by_dataset": activity,
    }
    save_json(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

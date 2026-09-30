#!/usr/bin/env python3
"""Evaluate one completed MoPE text checkpoint on COCO and Flickr30k."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.text.common import (
    COCO_ANNOTATIONS,
    COCO_IMAGES,
    EXPECTED_PRETRAINED_SHA256,
    FLICKR_ANNOTATIONS,
    FLICKR_IMAGES,
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
from sparmoe_vl.baselines.text.mope import (
    METHOD,
    kept_channel_indices,
    load_text_state_dict,
    selected_groups,
    statistics,
    structurally_prune_text_ffn,
    validate_final_checkpoint,
    validate_pruned_text,
    validate_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
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
    if file_sha256(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("current Dense CLIP differs from the MoPE initialization")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    selection = torch.load(args.selection, map_location="cpu", weights_only=False)
    validate_final_checkpoint(checkpoint)
    validate_selection(selection)
    if file_sha256(args.selection) != checkpoint.get("selection_sha256"):
        raise ValueError("MoPE checkpoint and structure-selection artifact differ")

    seed = int(checkpoint["seed"])
    seed_everything(seed)
    model, preprocess, tokenizer = create_clip(args.device, args.pretrained)
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

    expected_groups = selected_groups(selection, seed)
    if checkpoint["kept_group_indices"] != expected_groups:
        raise ValueError("MoPE checkpoint retained groups differ from structure selection")
    kept_indices = kept_channel_indices(selection["groups"], expected_groups)
    structurally_prune_text_ffn(model, kept_indices)
    validate_pruned_text(model, int(checkpoint["retained_width"]))
    load_text_state_dict(model, checkpoint["text_state_dict"])
    model.eval()

    summary = statistics(int(checkpoint["retained_width"]))
    active_parameters_m = (
        sum(parameter.numel() for parameter in model.transformer.parameters()) / 1e6
    )
    if abs(active_parameters_m - float(summary["active_text_parameters_m"])) > 1e-9:
        raise RuntimeError("physical MoPE parameter count differs from its checkpoint")

    metrics = {}
    for name, dataset, dense_images in (
        ("coco", coco, coco_images),
        ("flickr30k", flickr, flickr_images),
    ):
        text_features = encode_texts(
            model,
            tokenizer,
            dataset.captions,
            args.device,
            args.text_batch_size,
            f"{name} MoPE texts seed {seed}",
        )
        metrics[name] = retrieval_metrics(
            dense_images,
            text_features,
            dataset.caption_image_indices,
        )

    manifest = checkpoint["data_manifest"]
    result = {
        "method": METHOD,
        "seed": seed,
        "data_seed": manifest["data_seed"],
        "unique_samples": manifest["unique_samples"],
        "stage1_exposures": manifest["stage1_exposures"],
        "stage2_exposures": manifest["stage2_exposures"],
        "stage_sequences_identical": manifest["stage_sequences_identical"],
        "total_exposures": manifest["total_exposures"],
        "dataset_sha256": manifest["dataset_sha256"],
        "checkpoint": str(args.checkpoint.resolve()),
        "selection": str(args.selection.resolve()),
        "kept_group_indices": checkpoint["kept_group_indices"],
        "target_ffn_reduction": checkpoint["target_ffn_reduction"],
        **summary,
        "metrics": metrics,
        "counts": {
            "coco": {"images": len(coco.image_paths), "texts": len(coco.captions)},
            "flickr30k": {
                "images": len(flickr.image_paths),
                "texts": len(flickr.captions),
            },
        },
    }
    save_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

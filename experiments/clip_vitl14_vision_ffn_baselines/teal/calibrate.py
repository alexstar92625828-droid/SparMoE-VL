#!/usr/bin/env python3
"""Calibrate visual TEAL thresholds on the complete 500k main pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.vision.common import (
    ANNOTATIONS,
    EXPECTED_IMAGE_POOL_SHA256,
    EXPECTED_PRETRAINED_SHA256,
    IMAGE_ROOT,
    N_LAYERS,
    POOL_SIZE,
    PRETRAINED,
    atomic_torch_save,
    create_clip,
    file_sha256,
    save_json,
    seed_everything,
)
from sparmoe_vl.baselines.vision.teal import (
    METHOD,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    calibrate_layer_streaming,
    full_calibration_pool,
    validate_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 123, 2026))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-ffn-reduction", type=float, default=0.3563)
    parser.add_argument("--base-step-size", type=float, default=0.05)
    parser.add_argument("--histogram-bins", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.histogram_bins <= 1:
        parser.error("batch size/bins must be positive and workers non-negative")
    if abs(args.target_ffn_reduction - 0.3563) > 1e-10:
        parser.error("the visual TEAL comparison fixes target reduction=0.3563")
    if abs(args.base_step_size - 0.05) > 1e-10:
        parser.error("the visual TEAL comparison fixes step size=0.05")
    return args


def main() -> None:
    args = parse_args()
    configuration = {
        "method": METHOD,
        "processing_seed": args.seed,
        "data_seed": 42,
        "selected_images": POOL_SIZE,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "target_ffn_reduction": args.target_ffn_reduction,
        "base_step_size": args.base_step_size,
        "histogram_bins": args.histogram_bins,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "pretrained": str(args.pretrained.resolve()),
        "annotations": str(args.annotations.resolve()),
        "image_root": str(args.image_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.check_only:
        print(json.dumps(configuration, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing full-pool TEAL calibration")
    if not args.annotations.is_file() or not args.image_root.is_dir():
        raise FileNotFoundError("ShareGPT4V annotations or image root is unavailable")
    pretrained_sha256 = file_sha256(args.pretrained)
    if pretrained_sha256 != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")

    seed_everything(args.seed)
    paths, indices, manifest = full_calibration_pool(
        args.seed,
        args.annotations,
        args.image_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, args.output_dir / "data_manifest.json")
    progress_path = args.output_dir / "calibration_progress.pt"
    schedules = []
    if progress_path.is_file():
        if not args.resume:
            raise RuntimeError(f"progress exists at {progress_path}; pass --resume")
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if (
            progress.get("configuration") != configuration
            or progress.get("manifest") != manifest
        ):
            raise RuntimeError("TEAL calibration progress belongs to another protocol")
        schedules = progress.get("layers", [])
        if not isinstance(schedules, list) or len(schedules) > N_LAYERS:
            raise RuntimeError("invalid TEAL visual calibration progress")

    model, preprocess, _ = create_clip(args.device, args.pretrained)
    for layer_index in range(len(schedules), N_LAYERS):
        print(
            f"calibrating visual layer {layer_index + 1}/{N_LAYERS} "
            f"with all {POOL_SIZE:,} images",
            flush=True,
        )
        schedule = calibrate_layer_streaming(
            model=model,
            paths=paths,
            indices=indices,
            preprocess=preprocess,
            layer_index=layer_index,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            target=args.target_ffn_reduction,
            base_step=args.base_step_size,
            histogram_bins=args.histogram_bins,
        )
        schedules.append(schedule)
        atomic_torch_save(
            {
                "configuration": configuration,
                "manifest": manifest,
                "layers": schedules,
            },
            progress_path,
        )
        print(
            f"layer={layer_index:02d} effective={schedule['effective_sparsity']:.4f} "
            f"fc={schedule['fc_sparsity']:.2f} proj={schedule['proj_sparsity']:.2f}",
            flush=True,
        )

    checkpoint = {
        "format_version": 1,
        "method": METHOD,
        "stage": "calibrated",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "adaptation": {
            "scope": "visual FFN c_fc and c_proj only",
            "token_policy": "patch tokens magnitude-sparsified; CLS token dense",
            "allocation": "TEAL layer-wise greedy L2 allocation",
            "weight_updates": 0,
        },
        "model": "OpenCLIP ViT-L-14",
        "pretrained": str(args.pretrained.resolve()),
        "pretrained_sha256": pretrained_sha256,
        "calibration_manifest": manifest,
        "target_ffn_reduction": args.target_ffn_reduction,
        "base_step_size": args.base_step_size,
        "histogram_bins": args.histogram_bins,
        "layers": schedules,
        "complete": len(schedules) == N_LAYERS,
    }
    validate_checkpoint(checkpoint)
    checkpoint_path = args.output_dir / "teal_thresholds.pt"
    atomic_torch_save(checkpoint, checkpoint_path)
    save_json(
        checkpoint | {"checkpoint": str(checkpoint_path.resolve())},
        args.output_dir / "protocol.json",
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), "complete": True}, indent=2))


if __name__ == "__main__":
    main()

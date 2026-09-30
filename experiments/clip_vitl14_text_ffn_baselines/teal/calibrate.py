#!/usr/bin/env python3
"""Calibrate TEAL on the complete SparMoE-VL text training pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sparmoe_vl.baselines.text.common import (
    ANNOTATIONS,
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_TEXT_POOL_SHA256,
    N_LAYERS,
    POOL_SIZE,
    PRETRAINED,
    TOKEN_CACHE,
    create_clip,
    file_sha256,
    save_json,
    seed_everything,
)
from sparmoe_vl.baselines.text.teal import (
    METHOD,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    calibrate_layer_streaming,
    full_calibration_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--token-cache", type=Path, default=TOKEN_CACHE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-ffn-reduction", type=float, default=0.425)
    parser.add_argument("--base-step-size", type=float, default=0.05)
    parser.add_argument("--histogram-bins", type=int, default=10_000)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.histogram_bins <= 1:
        parser.error("batch size and histogram bins must be positive")
    if abs(args.target_ffn_reduction - 0.425) > 1e-10:
        parser.error("the released text TEAL comparison fixes reduction=0.425")
    if abs(args.base_step_size - 0.05) > 1e-10:
        parser.error("the released text TEAL comparison fixes step size=0.05")
    return args


def main() -> None:
    args = parse_args()
    configuration = {
        "method": METHOD,
        "seed": args.seed,
        "data_seed": 42,
        "samples": POOL_SIZE,
        "dataset_sha256": EXPECTED_TEXT_POOL_SHA256,
        "target_ffn_reduction": args.target_ffn_reduction,
        "base_step_size": args.base_step_size,
        "histogram_bins": args.histogram_bins,
        "batch_size": args.batch_size,
        "pretrained": str(args.pretrained.resolve()),
        "annotations": str(args.annotations.resolve()),
        "token_cache": str(args.token_cache.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.check_only:
        print(json.dumps(configuration, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing full-pool CPU calibration")
    if not args.pretrained.is_file():
        raise FileNotFoundError(args.pretrained)
    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    pretrained_sha256 = file_sha256(args.pretrained)
    if pretrained_sha256 != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")

    seed_everything(args.seed)
    tokens, indices, manifest = full_calibration_pool(
        args.seed,
        args.token_cache,
        args.annotations,
    )
    if not manifest["uses_complete_pool"] or manifest["selected_samples"] != POOL_SIZE:
        raise RuntimeError("TEAL calibration must consume the complete text pool")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, args.output_dir / "data_manifest.json")
    model, _, _ = create_clip(args.device, args.pretrained)

    schedules = []
    for layer_index in range(N_LAYERS):
        print(
            f"calibrating text layer {layer_index + 1}/{N_LAYERS} "
            f"with all {POOL_SIZE:,} samples",
            flush=True,
        )
        schedule = calibrate_layer_streaming(
            model=model,
            tokens=tokens,
            indices=indices,
            layer_index=layer_index,
            device=args.device,
            batch_size=args.batch_size,
            target=args.target_ffn_reduction,
            base_step=args.base_step_size,
            histogram_bins=args.histogram_bins,
        )
        schedules.append(schedule)
        print(
            f"layer={layer_index:02d} effective={schedule['effective_sparsity']:.4f} "
            f"fc={schedule['fc_sparsity']:.2f} "
            f"proj={schedule['proj_sparsity']:.2f}",
            flush=True,
        )

    checkpoint = {
        "format_version": 1,
        "method": METHOD,
        "stage": "calibrated",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "adaptation": {
            "scope": "text FFN c_fc and c_proj only",
            "token_policy": "all 77 CLIP text positions",
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
    checkpoint_path = args.output_dir / "teal_thresholds.pt"
    torch.save(checkpoint, checkpoint_path)
    save_json(
        checkpoint | {"checkpoint": str(checkpoint_path.resolve())},
        args.output_dir / "protocol.json",
    )
    print(json.dumps({"checkpoint": str(checkpoint_path), "complete": True}, indent=2))


if __name__ == "__main__":
    main()

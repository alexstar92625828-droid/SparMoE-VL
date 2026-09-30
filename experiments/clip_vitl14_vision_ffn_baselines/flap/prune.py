#!/usr/bin/env python3
"""Collect full-pool FLAP statistics and prune CLIP visual FFNs."""

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
    POOL_SIZE,
    PRETRAINED,
    create_clip,
    file_sha256,
    save_json,
    seed_everything,
)
from sparmoe_vl.baselines.vision.flap import (
    CALIBRATION_BATCH_SIZE,
    METHOD,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    PAPER,
    TARGET_FFN_REDUCTION,
    collect_wifv,
    full_calibration_pool,
    save_final_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42, 123, 2026))
    parser.add_argument("--target-ffn-reduction", type=float, default=TARGET_FFN_REDUCTION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--batch-size", type=int, default=CALIBRATION_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-batches",
        type=int,
        help="diagnostic stop only; incomplete calibration cannot emit a checkpoint",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size != CALIBRATION_BATCH_SIZE:
        parser.error(f"the released visual FLAP batch size is {CALIBRATION_BATCH_SIZE}")
    if abs(args.target_ffn_reduction - TARGET_FFN_REDUCTION) > 1e-12:
        parser.error(f"the visual FLAP comparison fixes reduction={TARGET_FFN_REDUCTION}")
    if args.workers < 0 or args.log_every <= 0 or args.save_every <= 0:
        parser.error("workers must be non-negative and progress intervals positive")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    return args


def main() -> None:
    args = parse_args()
    batches = (POOL_SIZE + args.batch_size - 1) // args.batch_size
    configuration = {
        "method": METHOD,
        "seed": args.seed,
        "data_seed": 42,
        "unique_samples": POOL_SIZE,
        "calibration_exposures": POOL_SIZE,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "batch_size": args.batch_size,
        "batches": batches,
        "final_batch_size": POOL_SIZE - (batches - 1) * args.batch_size,
        "target_ffn_reduction": args.target_ffn_reduction,
        "require_complete_main_pool": True,
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.check_only:
        print(json.dumps(configuration, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing full-pool FLAP on CPU")
    pretrained_sha256 = file_sha256(args.pretrained)
    if pretrained_sha256 != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")

    seed_everything(args.seed)
    paths, order, manifest = full_calibration_pool(
        args.seed,
        args.annotations,
        args.image_root,
        args.batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, args.output_dir / "data_manifest.json")
    model, preprocess, _ = create_clip(args.device, args.pretrained)
    moment_pack, report = collect_wifv(
        model=model,
        preprocess=preprocess,
        paths=paths,
        order=order,
        manifest=manifest,
        device=args.device,
        output_dir=args.output_dir,
        workers=args.workers,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
        max_batches=args.max_batches,
    )
    save_json(report, args.output_dir / "calibration_report.json")
    if not report["complete"]:
        print(json.dumps(report, indent=2))
        print("incomplete diagnostic/calibration run; no FLAP checkpoint emitted", flush=True)
        return
    if moment_pack is None:
        raise AssertionError("complete FLAP calibration did not return statistics")

    checkpoint_path, statistics_path, checkpoint = save_final_checkpoint(
        model=model,
        moment_pack=moment_pack,
        manifest=manifest,
        pretrained_sha256=pretrained_sha256,
        output_dir=args.output_dir,
        target_reduction=args.target_ffn_reduction,
    )
    torch_device = torch.device(args.device)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=torch_device.type,
            dtype=torch.float16,
            enabled=torch_device.type == "cuda",
        ),
    ):
        feature = model.encode_image(torch.zeros(1, 3, 224, 224, device=args.device))
    if tuple(feature.shape) != (1, 768) or not torch.isfinite(feature).all():
        raise RuntimeError("physically pruned FLAP visual tower failed validation")

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
        "text_tower_pruned": False,
        "weight_updates": 0,
        "gradients_used": False,
        "captions_used": False,
        "metric": "WIFV (FLAP Eq. 5)",
        "metric_formula": moment_pack["formula"],
        "allocation": "per-layer z-score followed by global channel ranking",
        "bias_compensation": "mean-input compensation (FLAP Eqs. 3-4)",
        "release_data_policy": "all 500,000 visual main-experiment images exactly once",
        "data_manifest": manifest,
        "target_ffn_reduction": args.target_ffn_reduction,
        "hidden_sizes": checkpoint["hidden_sizes"],
        "statistics": checkpoint["statistics"],
        "checkpoint": str(checkpoint_path.resolve()),
        "statistics_file": str(statistics_path.resolve()),
    }
    save_json(protocol, args.output_dir / "protocol.json")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()

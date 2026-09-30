#!/usr/bin/env python3
"""Collect full-pool FLAP statistics and prune CLIP text FFNs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from sparmoe_vl.baselines.text.common import (
    ANNOTATIONS,
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_TEXT_POOL_SHA256,
    POOL_SIZE,
    PRETRAINED,
    TOKEN_CACHE,
    create_clip,
    file_sha256,
    prepare_token_cache,
    save_json,
    seed_everything,
)
from sparmoe_vl.baselines.text.flap import (
    BUDGET_REFERENCE_SHA256,
    METHOD,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    PAPER,
    STAGE_EXPOSURES,
    TARGET_REDUCTIONS,
    TOTAL_EXPOSURES,
    collect_wifv,
    full_exposure_manifest,
    save_final_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=tuple(TARGET_REDUCTIONS))
    parser.add_argument("--target-ffn-reduction", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--token-cache", type=Path, default=TOKEN_CACHE)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    expected = TARGET_REDUCTIONS[args.seed]
    if args.target_ffn_reduction is None:
        args.target_ffn_reduction = expected
    if abs(args.target_ffn_reduction - expected) > 1e-12:
        parser.error(f"seed {args.seed} fixes target reduction to {expected}")
    if args.batch_size != 256:
        parser.error("FLAP batch size must match the text main experiment: 256")
    if args.log_every <= 0 or args.save_every <= 0:
        parser.error("progress intervals must be positive")
    return args


def main() -> None:
    args = parse_args()
    configuration = {
        "method": METHOD,
        "seed": args.seed,
        "data_seed": 42,
        "unique_samples": POOL_SIZE,
        "dataset_sha256": EXPECTED_TEXT_POOL_SHA256,
        "stage1_exposures": STAGE_EXPOSURES,
        "stage2_exposures": STAGE_EXPOSURES,
        "total_exposures": TOTAL_EXPOSURES,
        "batch_size": args.batch_size,
        "target_ffn_reduction": args.target_ffn_reduction,
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.check_only:
        print(json.dumps(configuration, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing full FLAP calibration on CPU")
    if file_sha256(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")

    seed_everything(args.seed)
    cache = prepare_token_cache(args.token_cache, args.annotations)
    indices, manifest = full_exposure_manifest(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(manifest, args.output_dir / "data_manifest.json")

    model, _, _ = create_clip(args.device, args.pretrained)
    model.visual = nn.Identity()
    moment_pack = collect_wifv(
        model=model,
        tokens=cache["tokens"],
        indices=indices,
        manifest=manifest,
        device=args.device,
        output_dir=args.output_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        resume=args.resume,
    )
    checkpoint_path, statistics_path, checkpoint = save_final_checkpoint(
        model=model,
        moment_pack=moment_pack,
        manifest=manifest,
        pretrained_sha256=EXPECTED_PRETRAINED_SHA256,
        output_dir=args.output_dir,
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
        feature = model.encode_text(cache["tokens"][:2].to(args.device))
    if tuple(feature.shape) != (2, 768) or not torch.isfinite(feature).all():
        raise RuntimeError("physically pruned FLAP text tower failed validation")

    protocol = {
        "method": METHOD,
        "paper": PAPER,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "compressed_modality": "text only",
        "compressed_modules": "12 text FFN hidden dimensions",
        "attention_pruned": False,
        "visual_tower_pruned": False,
        "weight_updates": 0,
        "metric": "WIFV (FLAP Eq. 5)",
        "allocation": "per-layer z-score followed by global channel ranking",
        "bias_compensation": "mean-input compensation (FLAP Eqs. 3-4)",
        "budget_reference_sha256": BUDGET_REFERENCE_SHA256,
        "paired_budget_target": args.target_ffn_reduction,
        "data_manifest": manifest,
        "hidden_sizes": checkpoint["hidden_sizes"],
        "statistics": checkpoint["statistics"],
        "checkpoint": str(checkpoint_path.resolve()),
        "statistics_file": str(statistics_path.resolve()),
    }
    save_json(protocol, args.output_dir / "protocol.json")
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select MoPE channel groups using the complete 500k main-experiment pool."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sparmoe_vl.baselines.text.common import (
    EXPECTED_PRETRAINED_SHA256,
    IMAGE_FEATURE_CACHE,
    POOL_SIZE,
    PRETRAINED,
    TOKEN_CACHE,
    create_clip,
    file_sha256,
    prepare_token_cache,
    save_json,
    seed_everything,
    text_blocks,
    validate_image_feature_cache,
)
from sparmoe_vl.baselines.text.mope import (
    CHECKPOINT_METHOD,
    GROUP_SIZE,
    METHOD,
    MOPE_DENSE_TEXT_CACHE,
    NUM_GROUPS,
    PAPER,
    STRUCTURE_SEED,
    contrastive_loss,
    ffn_taylor_scores,
    rankings_to_groups,
    recall_counts,
    recall_percentages,
    register_group_ablation,
    structure_data_manifest,
    validate_dense_text_cache,
    validate_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--token-cache", type=Path, default=TOKEN_CACHE)
    parser.add_argument("--image-feature-cache", type=Path, default=IMAGE_FEATURE_CACHE)
    parser.add_argument("--text-feature-cache", type=Path, default=MOPE_DENSE_TEXT_CACHE)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 256:
        parser.error("MoPE structure selection uses the main text batch size: 256")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    return args


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def collect_taylor_scores(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    image_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.requires_grad_(False)
    scored_parameters = []
    for block in text_blocks(model):
        for parameter in (block.mlp.c_fc.weight, block.mlp.c_fc.bias, block.mlp.c_proj.weight):
            parameter.requires_grad_(True)
            scored_parameters.append(parameter)
    accumulated = torch.zeros((12, 3072), dtype=torch.float64)
    samples = 0
    scale = model.logit_scale.exp().detach().clamp(max=100)
    for start in range(0, POOL_SIZE, batch_size):
        stop = min(start + batch_size, POOL_SIZE)
        for parameter in scored_parameters:
            parameter.grad = None
        batch_tokens = tokens[start:stop].to(device, non_blocking=device.type == "cuda")
        batch_images = image_features[start:stop].to(
            device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            text_features = F.normalize(model.encode_text(batch_tokens).float(), dim=-1)
            loss = contrastive_loss(batch_images, text_features, scale)
        loss.backward()
        count = stop - start
        accumulated += torch.stack(ffn_taylor_scores(model)).double() * count
        samples += count
        if (start // batch_size + 1) % 100 == 0:
            print(f"taylor_samples={samples}/{POOL_SIZE}", flush=True)
    model.requires_grad_(False)
    return (accumulated / samples).float()


@torch.inference_mode()
def dense_recall(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    counts = torch.zeros(3, dtype=torch.int64)
    for start in range(0, POOL_SIZE, batch_size):
        stop = min(start + batch_size, POOL_SIZE)
        counts += recall_counts(image_features[start:stop], text_features[start:stop])
    return recall_percentages(counts, POOL_SIZE)


@torch.inference_mode()
def ablated_recall(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    image_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
    groups: torch.Tensor,
    group_index: int,
) -> dict[str, float]:
    handles = register_group_ablation(model, groups, group_index)
    counts = torch.zeros(3, dtype=torch.int64)
    try:
        for start in range(0, POOL_SIZE, batch_size):
            stop = min(start + batch_size, POOL_SIZE)
            batch_tokens = tokens[start:stop].to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                text_features = F.normalize(model.encode_text(batch_tokens), dim=-1).cpu()
            counts += recall_counts(image_features[start:stop], text_features)
    finally:
        for handle in handles:
            handle.remove()
    return recall_percentages(counts, POOL_SIZE)


def main() -> None:
    args = parse_args()
    contract = {
        "method": METHOD,
        "structure_seed": STRUCTURE_SEED,
        "group_size": GROUP_SIZE,
        "groups_per_layer": NUM_GROUPS,
        "batch_size": args.batch_size,
        **structure_data_manifest(),
    }
    if args.check_only:
        print(json.dumps(contract, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing full-pool MoPE selection")
    pretrained_sha256 = file_sha256(args.pretrained)
    if pretrained_sha256 != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")
    seed_everything(STRUCTURE_SEED)

    cache = prepare_token_cache(args.token_cache)
    image_cache = torch.load(args.image_feature_cache, map_location="cpu", weights_only=False)
    text_cache = torch.load(args.text_feature_cache, map_location="cpu", weights_only=False)
    validate_image_feature_cache(image_cache)
    validate_dense_text_cache(text_cache)
    tokens = cache["tokens"]
    image_features = image_cache["features"]
    dense_text_features = text_cache["features"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "selection_progress.pt"
    if progress_path.exists():
        if not args.resume:
            raise RuntimeError(f"progress exists at {progress_path}; pass --resume")
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("contract") != contract:
            raise RuntimeError("MoPE selection progress belongs to a different protocol")
        taylor_scores = progress["taylor_scores"]
        rankings, groups = rankings_to_groups(taylor_scores)
        dense = progress["dense_recall"]
        records = progress["group_records"]
        elapsed_before = float(progress.get("elapsed_seconds", 0.0))
    else:
        model, _, _ = create_clip(args.device, args.pretrained)
        taylor_scores = collect_taylor_scores(
            model,
            tokens,
            image_features,
            torch.device(args.device),
            args.batch_size,
        )
        rankings, groups = rankings_to_groups(taylor_scores)
        dense = dense_recall(image_features, dense_text_features, args.batch_size)
        records = []
        elapsed_before = 0.0
        atomic_torch_save(
            {
                "contract": contract,
                "taylor_scores": taylor_scores,
                "dense_recall": dense,
                "group_records": records,
                "elapsed_seconds": elapsed_before,
            },
            progress_path,
        )
        del model
        torch.cuda.empty_cache()

    model, _, _ = create_clip(args.device, args.pretrained)
    started = time.time()
    for group_index in range(len(records), NUM_GROUPS):
        recall = ablated_recall(
            model,
            tokens,
            image_features,
            torch.device(args.device),
            args.batch_size,
            groups,
            group_index,
        )
        records.append(
            {
                "group": group_index,
                "mope": dense["mean"] - recall["mean"],
                "ablated_recall": recall,
            }
        )
        elapsed = elapsed_before + time.time() - started
        print(
            f"group={group_index + 1}/{NUM_GROUPS} "
            f"samples={POOL_SIZE} mope={records[-1]['mope']:.6f}",
            flush=True,
        )
        if len(records) % args.save_every == 0 or len(records) == NUM_GROUPS:
            atomic_torch_save(
                {
                    "contract": contract,
                    "taylor_scores": taylor_scores,
                    "dense_recall": dense,
                    "group_records": records,
                    "elapsed_seconds": elapsed,
                },
                progress_path,
            )

    priority_records = sorted(records, key=lambda item: (-item["mope"], item["group"]))
    selection = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "paper": PAPER,
        "stage": "structure_selection",
        "complete": True,
        "structure_seed": STRUCTURE_SEED,
        "pretrained_sha256": pretrained_sha256,
        "data_manifest": structure_data_manifest(),
        "group_size": GROUP_SIZE,
        "num_groups": NUM_GROUPS,
        "taylor_scores": taylor_scores,
        "rankings": rankings,
        "groups": groups,
        "dense_recall": dense,
        "group_records": records,
        "group_priority": [int(item["group"]) for item in priority_records],
    }
    validate_selection(selection)
    selection_path = args.output_dir / "selection.pt"
    atomic_torch_save(selection, selection_path)
    report = {
        key: value
        for key, value in selection.items()
        if not isinstance(value, torch.Tensor) and key != "group_records"
    }
    report["selection"] = str(selection_path.resolve())
    report["groups_scored"] = len(records)
    report["elapsed_seconds"] = elapsed_before + time.time() - started
    save_json(report, args.output_dir / "protocol.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

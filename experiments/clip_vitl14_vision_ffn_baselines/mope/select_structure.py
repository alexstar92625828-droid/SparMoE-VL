#!/usr/bin/env python3
"""Select visual MoPE groups from the complete 500k main-experiment pool."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sparmoe_vl.baselines.vision.common import (
    ANNOTATIONS,
    EXPECTED_PRETRAINED_SHA256,
    IMAGE_ROOT,
    PRETRAINED,
    atomic_torch_save,
    build_main_image_text_pool,
    create_clip,
    file_sha256,
    full_pool_permutation,
    save_json,
    seed_everything,
)
from sparmoe_vl.baselines.vision.mope import (
    CHECKPOINT_METHOD,
    GROUP_SIZE,
    METHOD,
    NUM_GROUPS,
    PAPER,
    SELECTION_BATCH_SIZE,
    STRUCTURE_SEED,
    TEXT_FEATURE_CACHE,
    collect_taylor_scores,
    full_pool_paired_recall,
    rankings_to_groups,
    structure_data_manifest,
    validate_dense_text_cache,
    validate_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--text-feature-cache", type=Path, default=TEXT_FEATURE_CACHE)
    parser.add_argument("--batch-size", type=int, default=SELECTION_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--taylor-save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size != SELECTION_BATCH_SIZE:
        parser.error("formal visual MoPE selection fixes batch size 32")
    if args.workers < 0 or args.taylor_save_every <= 0 or args.log_every <= 0:
        parser.error("worker and progress settings are invalid")
    return args


def main() -> None:
    args = parse_args()
    data_manifest = structure_data_manifest()
    contract = {
        "method": METHOD,
        "structure_seed": STRUCTURE_SEED,
        "group_size": GROUP_SIZE,
        "groups_per_layer": NUM_GROUPS,
        "batch_size": SELECTION_BATCH_SIZE,
        "data_manifest": data_manifest,
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
    paths, _captions, pool_manifest = build_main_image_text_pool(
        args.annotations,
        args.image_root,
    )
    if (
        pool_manifest["dataset_sha256"] != data_manifest["dataset_sha256"]
        or pool_manifest["paired_caption_sha256"] != data_manifest["paired_caption_sha256"]
    ):
        raise RuntimeError("current visual pairs differ from the MoPE selection manifest")
    cache = torch.load(args.text_feature_cache, map_location="cpu", weights_only=False)
    validate_dense_text_cache(cache)
    text_features = cache["features"]
    order = full_pool_permutation(STRUCTURE_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess, _tokenizer = create_clip(args.device, args.pretrained)
    taylor_scores, taylor_report = collect_taylor_scores(
        model,
        preprocess,
        paths,
        text_features,
        order,
        args.device,
        args.output_dir,
        contract,
        args.workers,
        args.taylor_save_every,
        args.log_every,
        args.resume,
    )
    if taylor_scores is None:
        print(json.dumps({"method": METHOD, "taylor": taylor_report}, indent=2))
        return
    rankings, groups = rankings_to_groups(taylor_scores)

    progress_path = args.output_dir / "selection_progress.pt"
    if progress_path.is_file():
        if not args.resume:
            raise RuntimeError(f"selection progress exists at {progress_path}; pass --resume")
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("contract") != contract:
            raise RuntimeError("MoPE selection progress belongs to another protocol")
        progress_scores = progress.get("taylor_scores")
        if not isinstance(progress_scores, torch.Tensor) or not torch.equal(
            progress_scores,
            taylor_scores,
        ):
            raise RuntimeError("MoPE Taylor scores changed since group scoring began")
        dense_recall = progress["dense_recall"]
        records = progress["group_records"]
        elapsed_before = float(progress.get("elapsed_seconds", 0.0))
        if not isinstance(records, list) or not 0 <= len(records) <= NUM_GROUPS:
            raise RuntimeError("invalid MoPE group-scoring progress")
        if [record.get("group") for record in records] != list(range(len(records))):
            raise RuntimeError("MoPE group-scoring progress is not an ordered prefix")
    else:
        dense_recall = full_pool_paired_recall(
            model,
            preprocess,
            paths,
            text_features,
            order,
            args.device,
            args.workers,
        )
        records = []
        elapsed_before = 0.0
        atomic_torch_save(
            {
                "contract": contract,
                "taylor_scores": taylor_scores,
                "dense_recall": dense_recall,
                "group_records": records,
                "elapsed_seconds": elapsed_before,
            },
            progress_path,
        )

    started = time.time()
    for group_index in range(len(records), NUM_GROUPS):
        recall = full_pool_paired_recall(
            model,
            preprocess,
            paths,
            text_features,
            order,
            args.device,
            args.workers,
            groups,
            group_index,
        )
        records.append(
            {
                "group": group_index,
                "mope": dense_recall["mean"] - recall["mean"],
                "ablated_recall": recall,
            }
        )
        elapsed = elapsed_before + time.time() - started
        atomic_torch_save(
            {
                "contract": contract,
                "taylor_scores": taylor_scores,
                "dense_recall": dense_recall,
                "group_records": records,
                "elapsed_seconds": elapsed,
            },
            progress_path,
        )
        print(
            f"group={group_index + 1}/{NUM_GROUPS} samples=500000 "
            f"mope={records[-1]['mope']:.6f}",
            flush=True,
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
        "data_manifest": data_manifest,
        "group_size": GROUP_SIZE,
        "num_groups": NUM_GROUPS,
        "taylor_scores": taylor_scores,
        "rankings": rankings,
        "groups": groups,
        "dense_recall": dense_recall,
        "group_records": records,
        "group_priority": [int(record["group"]) for record in priority_records],
    }
    validate_selection(selection)
    selection_path = args.output_dir / "selection.pt"
    atomic_torch_save(selection, selection_path)
    report = {
        "method": METHOD,
        "paper": PAPER,
        "complete": True,
        "selection": str(selection_path.resolve()),
        "pretrained_sha256": pretrained_sha256,
        "data_manifest": data_manifest,
        "group_size": GROUP_SIZE,
        "groups_per_layer": NUM_GROUPS,
        "groups_scored": len(records),
        "group_priority": selection["group_priority"],
        "dense_recall": dense_recall,
        "taylor": taylor_report,
        "elapsed_seconds": elapsed_before + time.time() - started,
    }
    save_json(report, args.output_dir / "protocol.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Train the strict two-stage N=8 model used by the capacity study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...common.data import ShareGPT4VImageTrainingDataset
from ...common.training import (
    RunLogger,
    dataset_metadata,
    plain_args,
    save_checkpoint_atomic,
    seed_worker,
    set_reproducible_seed,
)
from ...common.two_stage import protocol_for_stage
from .checkpoints import checkpoint_metadata, torch_load
from .model import CLIPSparMoE, build_model, controller_state, initialize_stage2
from .protocol import (
    CAPACITY_FACTORS,
    DATA_SEED,
    MODEL_KEY,
    MODEL_NAME,
    NUM_WORKERS,
    OUTPUT_ROOT,
    POOL_SIZE,
    PRETRAINED,
    SEEDS,
    STUDY_NAME,
    TARGET_RATIO,
    TRAIN_ANNOTATIONS,
    TRAIN_BATCH_SIZE,
    TRAIN_IMAGES,
    TRAIN_STEPS,
    VISION_DATASET_SHA256,
    training_manifest,
)


def parse_args(argv: Optional[Sequence[str]] = None, *, stage: int = 2) -> argparse.Namespace:
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=TRAIN_ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=TRAIN_IMAGES)
    if stage == 2:
        parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    args.stage = stage
    args.stage1_checkpoint = getattr(args, "stage1_checkpoint", None)
    run_root = OUTPUT_ROOT / "training" / f"seed_{args.seed}"
    if stage == 2 and args.stage1_checkpoint is None:
        args.stage1_checkpoint = run_root / "stage1" / "best.pt"
    args.output_dir = args.output_dir or run_root / f"stage{stage}"
    args.model_name = MODEL_NAME
    args.data_seed = DATA_SEED
    args.max_samples = POOL_SIZE
    args.steps = TRAIN_STEPS
    args.batch_size = TRAIN_BATCH_SIZE
    args.num_workers = NUM_WORKERS
    args.target_ratio = TARGET_RATIO
    args.capacity_factors = CAPACITY_FACTORS
    args.learning_rate = 1e-3 if stage == 1 else 3e-4
    args.weight_decay = 0.05
    args.temperature = 0.4
    if stage == 1:
        args.budget_weight = 50.0
        args.separation_weight = 1.0
        args.best_budget_loss = 0.01
    else:
        args.router_warmup = 1_000
        args.random_evals = 3
    args.log_every = 100
    args.save_every = 0
    return args


def validate_paths(args: argparse.Namespace) -> None:
    for path, label in (
        (args.pretrained, "Dense CLIP ViT-L/14 weights"),
        (args.annotations, "ShareGPT4V annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"missing ShareGPT4V image root: {args.image_root}")
    if args.stage == 2 and not args.check_only and not args.stage1_checkpoint.is_file():
        raise FileNotFoundError(f"missing Stage-1 checkpoint: {args.stage1_checkpoint}")


def manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **training_manifest(args.seed, args.stage),
        "pretrained": str(args.pretrained.resolve()),
        "annotations": str(args.annotations.resolve()),
        "image_root": str(args.image_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "stage1_checkpoint": (
            str(args.stage1_checkpoint.resolve())
            if args.stage1_checkpoint is not None
            else None
        ),
    }


def build_dataset(
    args: argparse.Namespace,
    preprocess: Any,
) -> ShareGPT4VImageTrainingDataset:
    dataset = ShareGPT4VImageTrainingDataset(
        args.annotations,
        args.image_root,
        preprocess,
        max_samples=POOL_SIZE,
        data_seed=DATA_SEED,
    )
    if len(dataset) != POOL_SIZE or dataset.ordered_sha256 != VISION_DATASET_SHA256:
        raise RuntimeError(
            "training pool differs from the visual main experiment: "
            f"samples={len(dataset)}, sha256={dataset.ordered_sha256}; "
            f"expected samples={POOL_SIZE}, sha256={VISION_DATASET_SHA256}"
        )
    return dataset


def build_model_and_dataset(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CLIPSparMoE, ShareGPT4VImageTrainingDataset]:
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("open_clip_torch is required for this study") from error
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(args.pretrained),
        force_quick_gelu=True,
    )
    clip_model = clip_model.to(device).eval()
    model = build_model(
        clip_model,
        tau=args.temperature,
        training_stage=args.stage,
    ).to(device)
    return model, build_dataset(args, preprocess)


def make_loader(
    dataset: ShareGPT4VImageTrainingDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader[Tensor]:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def checkpoint_payload(
    model: CLIPSparMoE,
    args: argparse.Namespace,
    step: int,
    data_identity: dict[str, Any],
    metrics: dict[str, Any],
    stage1_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "format_version": 3,
        "method": "sparmoe_vl_capacity_intervention",
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "training_protocol": protocol_for_stage(args.stage),
        "stage": args.stage,
        "modality": "vision",
        "training_seed": args.seed,
        "target_ratio": TARGET_RATIO,
        "capacity_factors": list(CAPACITY_FACTORS),
        "sparse_layers": list(range(24)),
        "step": step,
        "dataset": data_identity,
        "controller": controller_state(model),
        "metrics": metrics,
        "train_args": plain_args(args),
    }
    if stage1_metadata is not None:
        payload["stage1"] = stage1_metadata
    return payload


def save_checkpoint(
    path: Path,
    model: CLIPSparMoE,
    args: argparse.Namespace,
    step: int,
    data_identity: dict[str, Any],
    metrics: dict[str, Any],
    stage1_metadata: dict[str, Any] | None,
) -> None:
    save_checkpoint_atomic(
        path,
        checkpoint_payload(model, args, step, data_identity, metrics, stage1_metadata),
    )


@torch.inference_mode()
def routing_cosines(
    model: CLIPSparMoE,
    images: Tensor,
    dense: Tensor,
    random_evals: int,
) -> tuple[float, float]:
    model.eval()
    model.set_routing_mode("learned")
    learned, _ = model.encode_sparse(images)
    learned_cosine = float(F.cosine_similarity(learned, dense, dim=-1).mean())
    random_cosine = 0.0
    for _ in range(random_evals):
        model.set_routing_mode("random")
        random_features, _ = model.encode_sparse(images)
        random_cosine += (
            float(F.cosine_similarity(random_features, dense, dim=-1).mean()) / random_evals
        )
    return learned_cosine, random_cosine


def run(args: argparse.Namespace) -> None:
    validate_paths(args)
    if args.check_only:
        dataset = build_dataset(args, lambda image: image)
        result = {
            **manifest(args),
            "validated_samples": len(dataset),
            "validated_ordered_sha256": dataset.ordered_sha256,
        }
        print(json.dumps(result, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_reproducible_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output_dir / "train.log")
    model, dataset = build_model_and_dataset(args, device)
    data_identity = dataset_metadata(
        args.annotations,
        DATA_SEED,
        len(dataset),
        dataset.ordered_sha256,
        args.image_root,
    )
    stage1_metadata = None
    if args.stage == 2:
        stage1_payload = torch_load(args.stage1_checkpoint)
        stage1_metadata = checkpoint_metadata(stage1_payload, expected_stage=1)
        if stage1_metadata["dataset_sha256"] != data_identity["ordered_sha256"]:
            raise ValueError("Stage 1 and Stage 2 must use the identical ordered pool")
        initialize_stage2(model, stage1_payload)
        stage1_metadata = {
            **stage1_metadata,
            "checkpoint": str(args.stage1_checkpoint.resolve()),
        }
        del stage1_payload

    loader = make_loader(dataset, args, device)
    iterator = iter(loader)
    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    protocol = protocol_for_stage(args.stage)
    logger(
        f"study={STUDY_NAME} stage={args.stage} protocol={protocol} "
        f"model={MODEL_NAME} modality=vision seed={args.seed} "
        f"data_seed={DATA_SEED} samples={len(dataset)} "
        f"ordered_sha256={dataset.ordered_sha256} batch={TRAIN_BATCH_SIZE} "
        f"levels={list(CAPACITY_FACTORS)} "
        f"trainable={sum(parameter.numel() for parameter in parameters):,}"
    )
    best_cosine = -1.0
    best_step: Optional[int] = None
    stats: dict[str, Any] = {}
    for step in tqdm(range(1, TRAIN_STEPS + 1), desc=f"capacity study Stage {args.stage}"):
        try:
            images = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images = next(iterator)
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            dense = model.dense_features(images)
        model.train()
        if args.stage == 2:
            model.set_routing_mode("learned")
        sparse, auxiliary = model.encode_sparse(images)
        if args.stage == 1:
            loss, stats = model.stage1_loss(sparse, dense, auxiliary)
        else:
            router_weight = min(1.0, step / max(args.router_warmup, 1))
            loss, stats = model.stage2_loss(sparse, dense, auxiliary, router_weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            if args.stage == 1:
                learned_cosine = float(
                    F.cosine_similarity(sparse.detach(), dense, dim=-1).mean()
                )
                random_cosine = None
                eligible = stats["Rp"] < args.best_budget_loss
            else:
                learned_cosine, random_cosine = routing_cosines(
                    model, images, dense, args.random_evals
                )
                eligible = True
            if eligible and learned_cosine > best_cosine:
                best_cosine = learned_cosine
                best_step = step
                best_metrics = dict(stats)
                best_metrics.update(
                    learned_cosine=learned_cosine,
                    random_cosine=random_cosine,
                )
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    args,
                    step,
                    data_identity,
                    best_metrics,
                    stage1_metadata,
                )
            structure_budget = (
                stats["Rp"] if args.stage == 1 else stats["inherited_budget_error"]
            )
            logger(
                f"step={step} loss={stats['total']:.5f} "
                f"learned_cos={learned_cosine:.6f} random_cos={random_cosine} "
                f"structure_budget={structure_budget:.6f} "
                f"best={best_cosine:.6f} best_step={best_step}"
            )
            if args.stage == 2:
                model.train()

    save_checkpoint(
        args.output_dir / "final.pt",
        model,
        args,
        TRAIN_STEPS,
        data_identity,
        stats,
        stage1_metadata,
    )
    if best_step is None:
        raise RuntimeError(f"Stage {args.stage} produced no selected checkpoint")
    logger(f"finished best_step={best_step} best_cosine={best_cosine:.6f}")


def main(stage: int, argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv, stage=stage))

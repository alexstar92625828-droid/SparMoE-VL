"""Exact two-stage trainer for the CLIP ViT-B architecture-transfer rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ...common.data import ShareGPT4VImageTrainingDataset, ShareGPT4VTextTrainingDataset
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
    POOL_SIZE,
    PROJECT_ROOT,
    SEEDS,
    TRAIN_ANNOTATIONS,
    TRAIN_IMAGES,
    expected_dataset_sha256,
    get_spec,
)


def default_output_dir(model: str, modality: str, seed: int, stage: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "architecture_transfer"
        / "clip"
        / model
        / modality
        / f"seed_{seed}"
        / f"stage{stage}"
    )


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    model: str,
    stage: int,
) -> argparse.Namespace:
    spec = get_spec(model)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("vision", "text"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", default=spec.default_pretrained())
    parser.add_argument("--annotations", type=Path, default=TRAIN_ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=TRAIN_IMAGES)
    parser.add_argument("--output-dir", type=Path)
    if stage == 2:
        parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    args.model = model
    args.stage = stage
    args.data_seed = DATA_SEED
    args.max_samples = POOL_SIZE
    args.steps = 5_000
    args.batch_size = spec.batch_size(args.modality, stage)
    args.grad_accumulation = spec.text_grad_accumulation if args.modality == "text" else 1
    args.learning_rate = 1e-3 if stage == 1 else 3e-4
    args.weight_decay = 0.05
    args.temperature = 0.4
    args.router_warmup = 1_000
    args.log_every = 100
    args.save_every = 500
    args.random_evals = 3
    args.num_workers = 8 if args.modality == "vision" else 4
    args.target_ratio = spec.target_ratio(args.modality)
    args.output_dir = args.output_dir or default_output_dir(
        model, args.modality, args.seed, stage
    )
    if stage == 2 and args.stage1_checkpoint is None:
        args.stage1_checkpoint = (
            default_output_dir(model, args.modality, args.seed, 1) / "best.pt"
        )
    return args


def _pretrained_display(value: str) -> str:
    path = Path(value)
    return str(path.resolve()) if path.is_file() else value


def protocol_manifest(args: argparse.Namespace) -> dict[str, Any]:
    spec = get_spec(args.model)
    return {
        "study": "architecture_transfer",
        "family": "CLIP",
        "version": args.model,
        "model_key": spec.key,
        "open_clip_name": spec.open_clip_name,
        "modality": args.modality,
        "stage": args.stage,
        "training_seed": args.seed,
        "data_seed": DATA_SEED,
        "samples": POOL_SIZE,
        "expected_ordered_sha256": expected_dataset_sha256(args.modality),
        "steps": args.steps,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accumulation,
        "effective_batch_size": args.batch_size * args.grad_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "target_ratio": args.target_ratio,
        "capacity_factors": list(CAPACITY_FACTORS),
        "router_temperature": args.temperature,
        "router_warmup_steps": args.router_warmup,
        "annotations": str(args.annotations.resolve()),
        "image_root": str(args.image_root.resolve()),
        "pretrained": _pretrained_display(args.pretrained),
        "output_dir": str(args.output_dir.resolve()),
        "stage1_checkpoint": (
            str(args.stage1_checkpoint.resolve()) if args.stage == 2 else None
        ),
        "main_experiment_excluded": "CLIP ViT-L/14",
    }


def _looks_like_path(value: str) -> bool:
    return "/" in value or Path(value).suffix.lower() in {".pt", ".pth", ".bin", ".safetensors"}


def validate_paths(args: argparse.Namespace, *, require_stage1: bool) -> None:
    if not args.annotations.is_file():
        raise FileNotFoundError(f"missing ShareGPT4V annotations: {args.annotations}")
    if args.modality == "vision" and not args.image_root.is_dir():
        raise FileNotFoundError(f"missing ShareGPT4V image root: {args.image_root}")
    if _looks_like_path(args.pretrained) and not Path(args.pretrained).is_file():
        raise FileNotFoundError(f"missing CLIP pretrained weights: {args.pretrained}")
    if require_stage1 and not args.stage1_checkpoint.is_file():
        raise FileNotFoundError(f"missing Stage-1 checkpoint: {args.stage1_checkpoint}")


def build_backbone(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, Any, Any]:
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("open_clip_torch is required for CLIP training") from error
    spec = get_spec(args.model)
    backbone, _, preprocess = open_clip.create_model_and_transforms(
        spec.open_clip_name,
        pretrained=args.pretrained,
        force_quick_gelu=True,
    )
    tokenizer = open_clip.get_tokenizer(spec.open_clip_name)
    return backbone.to(device).eval(), preprocess, tokenizer


def build_dataset(
    args: argparse.Namespace,
    preprocess: Any,
    tokenizer: Any,
) -> Dataset[Tensor]:
    if args.modality == "vision":
        dataset: Dataset[Tensor] = ShareGPT4VImageTrainingDataset(
            args.annotations,
            args.image_root,
            preprocess,
            max_samples=POOL_SIZE,
            data_seed=DATA_SEED,
        )
    else:
        dataset = ShareGPT4VTextTrainingDataset(
            args.annotations,
            tokenizer,
            max_samples=POOL_SIZE,
            data_seed=DATA_SEED,
        )
    expected_sha = expected_dataset_sha256(args.modality)
    if len(dataset) != POOL_SIZE or dataset.ordered_sha256 != expected_sha:
        raise RuntimeError(
            "training pool differs from the main-experiment protocol: "
            f"samples={len(dataset)}, sha256={dataset.ordered_sha256}; "
            f"expected samples={POOL_SIZE}, sha256={expected_sha}"
        )
    return dataset


def make_loader(dataset: Dataset[Tensor], args: argparse.Namespace) -> DataLoader[Tensor]:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _checkpoint_payload(
    model: CLIPSparMoE,
    args: argparse.Namespace,
    step: int,
    identity: dict[str, Any],
    metrics: dict[str, Any],
    stage1_metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    spec = get_spec(args.model)
    payload: dict[str, Any] = {
        "format_version": 3,
        "method": "sparmoe_vl_clip_transfer",
        "study": "architecture_transfer",
        "protocol": protocol_for_stage(args.stage),
        "stage": args.stage,
        "modality": args.modality,
        "model_key": spec.key,
        "model_name": spec.open_clip_name,
        "training_seed": args.seed,
        "target_ratio": args.target_ratio,
        "capacity_factors": list(CAPACITY_FACTORS),
        "sparse_layers": list(range(spec.num_layers)),
        "step": step,
        "dataset": identity,
        "controller": controller_state(model),
        "metrics": metrics,
        "train_args": plain_args(args),
    }
    if args.stage == 2:
        payload["stage1_checkpoint"] = str(args.stage1_checkpoint.resolve())
        payload["stage1_step"] = stage1_metadata["checkpoint_step"]
        payload["stage1"] = {
            **stage1_metadata,
            "checkpoint": str(args.stage1_checkpoint.resolve()),
        }
    return payload


def _save(
    path: Path,
    model: CLIPSparMoE,
    args: argparse.Namespace,
    step: int,
    identity: dict[str, Any],
    metrics: dict[str, Any],
    stage1_metadata: Optional[dict[str, Any]],
) -> None:
    save_checkpoint_atomic(
        path,
        _checkpoint_payload(model, args, step, identity, metrics, stage1_metadata),
    )


@torch.no_grad()
def _routing_cosines(
    model: CLIPSparMoE,
    inputs: Tensor,
    dense: Tensor,
    random_evals: int,
) -> tuple[float, float]:
    model.eval()
    model.set_routing_mode("learned")
    learned, _ = model.encode_sparse(inputs)
    learned_cosine = float(F.cosine_similarity(learned, dense, dim=-1).mean())
    random_cosine = 0.0
    for _ in range(random_evals):
        model.set_routing_mode("random")
        random_features, _ = model.encode_sparse(inputs)
        random_cosine += (
            float(F.cosine_similarity(random_features, dense, dim=-1).mean()) / random_evals
        )
    return learned_cosine, random_cosine


def run(args: argparse.Namespace) -> None:
    validate_paths(args, require_stage1=args.stage == 2 and not args.check_only)
    if args.check_only:
        print(json.dumps(protocol_manifest(args), indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_reproducible_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output_dir / "train.log")
    backbone, preprocess, tokenizer = build_backbone(args, device)
    dataset = build_dataset(args, preprocess, tokenizer)
    identity = dataset_metadata(
        args.annotations,
        DATA_SEED,
        len(dataset),
        dataset.ordered_sha256,
        args.image_root if args.modality == "vision" else None,
    )
    model = build_model(
        backbone,
        args.modality,
        args.stage,
        args.target_ratio,
        CAPACITY_FACTORS,
        args.temperature,
    ).to(device)
    spec = get_spec(args.model)
    if len(model.blocks) != spec.num_layers or model.ffn_dim != (
        spec.vision_ffn_dim if args.modality == "vision" else spec.text_ffn_dim
    ):
        raise RuntimeError("loaded CLIP geometry differs from the registered version")

    stage1_metadata = None
    if args.stage == 2:
        stage1 = torch_load(args.stage1_checkpoint)
        stage1_metadata = checkpoint_metadata(stage1, spec, args.modality, 1)
        if stage1_metadata["training_seed"] != args.seed:
            raise ValueError("Stage 1 and Stage 2 training seeds differ")
        if stage1_metadata["dataset_sha256"] != identity["ordered_sha256"]:
            raise ValueError("Stage 1 and Stage 2 ordered training pools differ")
        ratios = initialize_stage2(model, stage1)
        logger(
            "stage_transition=validated same_data=true "
            f"ordered_sha256={identity['ordered_sha256']} "
            f"initial_layer_ratios={ratios}"
        )
        del stage1

    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loader = make_loader(dataset, args)
    iterator = iter(loader)
    logger(
        f"study=architecture_transfer family=CLIP version={args.model} "
        f"model={spec.open_clip_name} modality={args.modality} stage={args.stage} "
        f"seed={args.seed} data_seed={DATA_SEED} samples={len(dataset)} "
        f"ordered_sha256={identity['ordered_sha256']} "
        f"micro_batch={args.batch_size} effective_batch="
        f"{args.batch_size * args.grad_accumulation} "
        f"trainable={sum(parameter.numel() for parameter in parameters):,}"
    )
    best_cosine = -1.0
    best_step: Optional[int] = None
    stats: dict[str, Any] = {}
    for step in tqdm(
        range(1, args.steps + 1),
        desc=f"{args.model} {args.modality} stage {args.stage}",
    ):
        optimizer.zero_grad()
        for _ in range(args.grad_accumulation):
            try:
                inputs = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs = next(iterator)
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            with torch.no_grad():
                dense = model.dense_features(inputs)
            model.train()
            if args.stage == 2:
                model.set_routing_mode("learned")
            sparse, auxiliary = model.encode_sparse(inputs)
            if args.stage == 1:
                loss, stats = model.stage1_loss(sparse, dense, auxiliary)
            else:
                router_weight = min(1.0, step / max(args.router_warmup, 1))
                loss, stats = model.stage2_loss(sparse, dense, auxiliary, router_weight)
                stats["router_weight"] = router_weight
            (loss / args.grad_accumulation).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            if args.stage == 1:
                learned_cosine = float(
                    F.cosine_similarity(sparse.detach(), dense.detach(), dim=-1).mean()
                )
                random_cosine = None
                valid_budget = stats["Rp"] < 0.01
            else:
                learned_cosine, random_cosine = _routing_cosines(
                    model, inputs, dense, args.random_evals
                )
                valid_budget = True
            if valid_budget and learned_cosine > best_cosine:
                best_cosine = learned_cosine
                best_step = step
                best_metrics = dict(stats)
                best_metrics.update(
                    learned_cosine=learned_cosine,
                    random_cosine=random_cosine,
                )
                _save(
                    args.output_dir / "best.pt",
                    model,
                    args,
                    step,
                    identity,
                    best_metrics,
                    stage1_metadata,
                )
            logger(
                f"step={step} loss={stats['total']:.5f} "
                f"learned_cos={learned_cosine:.6f} "
                f"budget={stats['Rp'] if args.stage == 1 else stats['inherited_budget_error']:.6f} "
                f"best={best_cosine:.6f} best_step={best_step}"
            )
            if args.stage == 2:
                model.train()

        if step % args.save_every == 0:
            _save(
                args.output_dir / f"step_{step}.pt",
                model,
                args,
                step,
                identity,
                stats,
                stage1_metadata,
            )

    _save(
        args.output_dir / "final.pt",
        model,
        args,
        args.steps,
        identity,
        stats,
        stage1_metadata,
    )
    if best_step is None:
        raise RuntimeError("no checkpoint satisfied the registered budget criterion")
    logger(f"finished best_step={best_step} best_cosine={best_cosine:.6f}")


def main_stage1(model: str, argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv, model=model, stage=1))


def main_stage2(model: str, argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv, model=model, stage=2))

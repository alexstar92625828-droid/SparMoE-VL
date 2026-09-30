"""Train one component-ablation row with the strict two-stage protocol."""

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
from .data import ShareGPT4VImageTextTrainingDataset
from .model import (
    CLIPSparMoE,
    build_model,
    controller_state,
    distillation_loss,
    geometry_replacement_loss,
    initialize_stage2,
)
from .protocol import (
    CAPACITY_FACTORS,
    DATASET_SHA256,
    DATA_SEED,
    MASK_LEARNING_RATE,
    MODEL_KEY,
    MODEL_NAME,
    NUM_WORKERS,
    OUTPUT_ROOT,
    POOL_SIZE,
    PRETRAINED,
    REPLACEMENTS,
    ROUTER_TEMPERATURE,
    ROUTER_WARMUP,
    STAGE1_LEARNING_RATE,
    STAGE2_LEARNING_RATE,
    STUDY_NAME,
    TARGET_RATIO,
    TRAIN_ANNOTATIONS,
    TRAIN_BATCH_SIZE,
    TRAIN_IMAGES,
    TRAIN_STEPS,
    TRAINED_METHODS,
    WEIGHT_DECAY,
    training_manifest,
    validate_method_seed,
)


def parse_args(argv: Optional[Sequence[str]] = None, *, stage: int = 2) -> argparse.Namespace:
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=TRAINED_METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=TRAIN_ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=TRAIN_IMAGES)
    if stage == 2:
        parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    validate_method_seed(args.variant, args.seed)
    args.stage = stage
    args.stage1_checkpoint = getattr(args, "stage1_checkpoint", None)
    run_root = OUTPUT_ROOT / "training" / args.variant / f"seed_{args.seed}"
    if stage == 2 and args.stage1_checkpoint is None:
        args.stage1_checkpoint = run_root / "stage1" / "best.pt"
    args.output_dir = args.output_dir or run_root / f"stage{stage}"
    args.model_name = MODEL_NAME
    args.data_seed = DATA_SEED
    args.max_samples = POOL_SIZE
    args.steps = TRAIN_STEPS
    args.batch_size = TRAIN_BATCH_SIZE
    args.num_workers = NUM_WORKERS
    args.p = TARGET_RATIO
    args.levels = list(CAPACITY_FACTORS)
    args.learning_rate = STAGE1_LEARNING_RATE if stage == 1 else STAGE2_LEARNING_RATE
    if stage == 1 and args.variant == "without_spg":
        args.mask_learning_rate = MASK_LEARNING_RATE
    args.weight_decay = WEIGHT_DECAY
    args.temperature = ROUTER_TEMPERATURE
    if stage == 1:
        args.global_budget_weight = 50.0
        args.separation_weight = 1.0
        args.best_budget_loss = 0.01
    else:
        args.router_warmup = ROUTER_WARMUP
    args.log_every = 100
    args.replacement = REPLACEMENTS[args.variant]
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


def build_dataset(
    args: argparse.Namespace,
    preprocess: Any,
    tokenizer: Any = None,
) -> Dataset:
    if args.variant == "without_geometry_preservation":
        if tokenizer is None:
            raise ValueError("the geometry-preservation ablation requires a tokenizer")
        dataset: Dataset = ShareGPT4VImageTextTrainingDataset(
            args.annotations,
            args.image_root,
            preprocess,
            tokenizer,
            max_samples=POOL_SIZE,
            data_seed=DATA_SEED,
        )
    else:
        dataset = ShareGPT4VImageTrainingDataset(
            args.annotations,
            args.image_root,
            preprocess,
            max_samples=POOL_SIZE,
            data_seed=DATA_SEED,
        )
    ordered_sha256 = getattr(dataset, "ordered_sha256")
    if len(dataset) != POOL_SIZE or ordered_sha256 != DATASET_SHA256:
        raise RuntimeError(
            "training pool differs from the visual main experiment: "
            f"samples={len(dataset)}, sha256={ordered_sha256}; "
            f"expected samples={POOL_SIZE}, sha256={DATASET_SHA256}"
        )
    return dataset


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
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
        "method": "sparmoe_vl_component_ablation",
        "study": STUDY_NAME,
        "variant": args.variant,
        "replacement": REPLACEMENTS[args.variant],
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
        "controller": controller_state(model, args.variant),
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
def routing_cosines(model: CLIPSparMoE, images: Tensor, dense: Tensor) -> tuple[float, float]:
    model.eval()
    model.set_routing_mode("learned")
    learned, _ = model.encode_sparse(images)
    learned_cosine = float(F.cosine_similarity(learned, dense, dim=-1).mean())
    model.set_routing_mode("random")
    random_features, _ = model.encode_sparse(images)
    random_cosine = float(F.cosine_similarity(random_features, dense, dim=-1).mean())
    return learned_cosine, random_cosine


def run(args: argparse.Namespace) -> None:
    validate_paths(args)
    if args.check_only:

        def identity(image: Any) -> Any:
            return image

        def tokenizer(texts: Sequence[str]) -> Tensor:
            return torch.zeros(len(texts), 77, dtype=torch.long)

        dataset = build_dataset(args, identity, tokenizer)
        result = {
            **training_manifest(args.variant, args.seed, args.stage),
            "pretrained": str(args.pretrained.resolve()),
            "annotations": str(args.annotations.resolve()),
            "image_root": str(args.image_root.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "stage1_checkpoint": (
                str(args.stage1_checkpoint.resolve())
                if args.stage1_checkpoint is not None
                else None
            ),
            "validated_samples": len(dataset),
            "validated_ordered_sha256": getattr(dataset, "ordered_sha256"),
        }
        print(json.dumps(result, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_reproducible_seed(args.seed)
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
    model, mask_parameters = build_model(
        clip_model,
        args.variant,
        args.seed,
        args.temperature,
        training_stage=args.stage,
    )
    model = model.to(device)
    tokenizer = (
        open_clip.get_tokenizer(MODEL_NAME)
        if args.variant == "without_geometry_preservation"
        else None
    )
    dataset = build_dataset(args, preprocess, tokenizer)
    data_identity = dataset_metadata(
        args.annotations,
        DATA_SEED,
        len(dataset),
        getattr(dataset, "ordered_sha256"),
        args.image_root,
    )
    stage1_metadata = None
    if args.stage == 2:
        stage1_payload = torch_load(args.stage1_checkpoint)
        stage1_metadata = checkpoint_metadata(
            stage1_payload,
            args.variant,
            expected_stage=1,
        )
        if stage1_metadata["dataset_sha256"] != data_identity["ordered_sha256"]:
            raise ValueError("Stage 1 and Stage 2 must use the identical ordered pool")
        initialize_stage2(model, stage1_payload, args.variant)
        stage1_metadata = {
            **stage1_metadata,
            "checkpoint": str(args.stage1_checkpoint.resolve()),
        }
        del stage1_payload

    loader = make_loader(dataset, args, device)
    iterator = iter(loader)
    parameters = model.trainable_parameters()
    if not parameters:
        raise RuntimeError("training stage has no trainable parameters")
    if args.stage == 1 and args.variant == "without_spg":
        mask_ids = {id(parameter) for parameter in mask_parameters}
        ordinary = [parameter for parameter in parameters if id(parameter) not in mask_ids]
        parameter_groups = (
            {"params": ordinary, "lr": STAGE1_LEARNING_RATE},
            {"params": mask_parameters, "lr": MASK_LEARNING_RATE},
        )
    else:
        parameter_groups = ({"params": parameters, "lr": args.learning_rate},)
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=WEIGHT_DECAY)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output_dir / "train.log")
    protocol = protocol_for_stage(args.stage)
    logger(
        f"study={STUDY_NAME} stage={args.stage} protocol={protocol} "
        f"variant={args.variant} replacement={args.replacement} model={MODEL_NAME} "
        f"seed={args.seed} data_seed={DATA_SEED} samples={len(dataset)} "
        f"ordered_sha256={getattr(dataset, 'ordered_sha256')} "
        f"steps={TRAIN_STEPS} batch={TRAIN_BATCH_SIZE} "
        f"trainable={sum(parameter.numel() for parameter in parameters):,}"
    )

    geometry_replacement = args.variant == "without_geometry_preservation"
    best_value = float("inf") if geometry_replacement else -1.0
    best_step: Optional[int] = None
    last_stats: dict[str, Any] = {}
    for step in tqdm(
        range(1, TRAIN_STEPS + 1),
        desc=f"component ablation Stage {args.stage}: {args.variant}",
    ):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if geometry_replacement:
            images, tokens = batch
            tokens = tokens.to(device, non_blocking=device.type == "cuda")
        else:
            images = batch
            tokens = None
        images = images.to(device, non_blocking=device.type == "cuda")
        model.train()
        if args.stage == 2:
            model.set_routing_mode("learned")
        sparse, auxiliary = model.encode_sparse(images)
        router_weight = min(1.0, step / ROUTER_WARMUP) if args.stage == 2 else 0.0

        if geometry_replacement:
            if tokens is None:
                raise RuntimeError("contrastive ablation batch is missing tokens")
            with torch.no_grad():
                text_features = F.normalize(model.clip_model.encode_text(tokens), dim=-1)
                scale = model.clip_model.logit_scale.exp().clamp(max=100).detach()
            loss, last_stats = geometry_replacement_loss(
                model,
                sparse,
                text_features,
                auxiliary,
                scale,
                training_stage=args.stage,
                router_weight=router_weight,
            )
            objective = last_stats["contrastive"]
        else:
            with torch.no_grad():
                dense = model.dense_features(images)
            loss, last_stats = distillation_loss(
                model,
                sparse,
                dense,
                auxiliary,
                training_stage=args.stage,
                router_weight=router_weight,
            )
            objective = last_stats["distill"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == TRAIN_STEPS:
            learned_cosine = None
            random_cosine = None
            budget_valid = (
                last_stats["structure_budget_error"] < args.best_budget_loss
                if args.stage == 1
                else True
            )
            if geometry_replacement:
                is_best = budget_valid and objective < best_value
            else:
                if args.stage == 1:
                    learned_cosine = float(
                        F.cosine_similarity(sparse.detach(), dense, dim=-1).mean()
                    )
                else:
                    learned_cosine, random_cosine = routing_cosines(model, images, dense)
                is_best = budget_valid and learned_cosine > best_value
            logger(
                f"step={step} loss={last_stats['total']:.5f} "
                f"objective={objective:.6f} learned_cos={learned_cosine} "
                f"random_cos={random_cosine} "
                f"structure_budget={last_stats['structure_budget_error']:.6f} "
                f"router_acc={last_stats['router_acc']:.5f}"
            )
            if args.stage == 2:
                model.train()
            if is_best:
                best_value = objective if geometry_replacement else float(learned_cosine)
                best_step = step
                best_metrics = dict(last_stats)
                best_metrics.update(
                    objective=objective,
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

    save_checkpoint(
        args.output_dir / "final.pt",
        model,
        args,
        TRAIN_STEPS,
        data_identity,
        last_stats,
        stage1_metadata,
    )
    if best_step is None:
        raise RuntimeError("no checkpoint satisfied the registered selection criterion")
    logger(f"finished best_step={best_step} best_objective={best_value:.6f}")


def main(stage: int, argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv, stage=stage))

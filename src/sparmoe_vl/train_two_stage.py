"""Train the paper-defined two-stage CLIP ViT-L/14 SparMoE-VL models.

Stage 2 rebuilds the same ShareGPT4V subset and refuses to start unless its
ordered SHA-256 identity is identical to the one stored by Stage 1. Stage 1
learns only nested subspaces and layer capacities under global budget ``p``;
Stage 2 freezes that complete structure and optimizes only token routers. The
Stage-1 global budget remains an inherited structural upper bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .common.data import (
    ShareGPT4VImageTrainingDataset,
    ShareGPT4VTextTrainingDataset,
)
from .common.losses import (
    Stage1SubspaceObjective,
    Stage2RouterObjective,
    linear_warmup_weight,
)
from .common.training import (
    RunLogger,
    dataset_metadata,
    load_stage1_checkpoint,
    plain_args,
    save_checkpoint_atomic,
    seed_worker,
    set_reproducible_seed,
    validate_stage1_checkpoint,
)
from .common.two_stage import protocol_for_stage
from .paths import repository_root, workspace_root
from .text.encoder import SparMoETextEncoder
from .vision.encoder import SparMoEVisionEncoder


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
DEFAULT_PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
DEFAULT_ANNOTATIONS = RESEARCH_ROOT / "ShareGPT4V" / "annotations" / "sharegpt4v_1246k.json"
DEFAULT_IMAGE_ROOT = RESEARCH_ROOT / "ShareGPT4V" / "images"
CAPACITY_FACTORS = (0.7, 0.8, 0.9, 1.0)


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    allowed_target_ratios: Optional[Sequence[float]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("vision", "text"), required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="ViT-L-14")
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--target-ratio", type=float)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--router-warmup", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--random-evals", type=int, default=3)
    parser.add_argument(
        "--best-budget-loss",
        type=float,
        default=None,
        help="Stage-1-only checkpoint-selection threshold (default: 0.01).",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    apply_protocol_defaults(args)
    validate_args(args, allowed_target_ratios=allowed_target_ratios)
    if args.stage == 2:
        delattr(args, "best_budget_loss")
    return args


def apply_protocol_defaults(args: argparse.Namespace) -> None:
    if args.target_ratio is None:
        args.target_ratio = 0.7 if args.modality == "vision" else 0.6
    if args.batch_size is None:
        args.batch_size = (
            32
            if args.modality == "vision" and args.stage == 1
            else (24 if args.modality == "vision" else 256)
        )
    if args.learning_rate is None:
        args.learning_rate = 1e-3 if args.stage == 1 else 3e-4
    if args.num_workers is None:
        args.num_workers = 8 if args.modality == "vision" else 4
    if args.stage == 1 and args.best_budget_loss is None:
        args.best_budget_loss = 0.01


def validate_args(
    args: argparse.Namespace,
    *,
    allowed_target_ratios: Optional[Sequence[float]] = None,
) -> None:
    if not args.pretrained.is_file():
        raise FileNotFoundError(f"missing CLIP checkpoint: {args.pretrained}")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"missing ShareGPT4V annotations: {args.annotations}")
    if args.modality == "vision" and not args.image_root.is_dir():
        raise FileNotFoundError(f"missing ShareGPT4V images: {args.image_root}")
    if args.stage == 2:
        if args.best_budget_loss is not None:
            raise ValueError("--best-budget-loss applies only to stage 1")
        if args.stage1_checkpoint is None:
            raise ValueError("--stage1-checkpoint is required for stage 2")
        if not args.stage1_checkpoint.is_file():
            raise FileNotFoundError(args.stage1_checkpoint)
    positive = (
        args.steps,
        args.batch_size,
        args.learning_rate,
        args.temperature,
        args.max_samples,
        args.log_every,
        args.save_every,
        args.random_evals,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("steps, batch size, rates, and intervals must be positive")
    if args.num_workers < 0 or args.weight_decay < 0 or args.router_warmup < 0:
        raise ValueError("invalid worker, decay, or warm-up setting")
    default_ratio = 0.7 if args.modality == "vision" else 0.6
    allowed = (
        (default_ratio,)
        if allowed_target_ratios is None
        else tuple(float(value) for value in allowed_target_ratios)
    )
    if not allowed or not any(abs(args.target_ratio - value) <= 1e-8 for value in allowed):
        raise ValueError(
            f"target ratio {args.target_ratio} is not in the registered protocol: {allowed}"
        )


def build_model_and_data(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, Dataset[Tensor]]:
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("install open_clip_torch before training") from error

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=str(args.pretrained),
        force_quick_gelu=True,
    )
    clip_model = clip_model.to(device).eval()
    common = dict(
        clip_model=clip_model,
        sparse_layers=None,
        target_ratio=args.target_ratio,
        capacity_factors=CAPACITY_FACTORS,
        router_temperature=args.temperature,
        mask_temperature=args.temperature,
        training_stage=args.stage,
    )
    if args.modality == "vision":
        model = SparMoEVisionEncoder(**common).to(device)
        dataset = ShareGPT4VImageTrainingDataset(
            args.annotations,
            args.image_root,
            preprocess,
            max_samples=args.max_samples,
            data_seed=args.data_seed,
        )
    else:
        model = SparMoETextEncoder(
            **common,
            routing_target_scope="all_nonfirst",
        ).to(device)
        tokenizer = open_clip.get_tokenizer(args.model_name)
        dataset = ShareGPT4VTextTrainingDataset(
            args.annotations,
            tokenizer,
            max_samples=args.max_samples,
            data_seed=args.data_seed,
        )
    return model, dataset


def make_loader(
    dataset: Dataset[Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader[Tensor]:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def encoder_state(model: nn.Module) -> Dict[str, Any]:
    state = {
        "budget": model.budget.state_dict(),
        "sparse_pattern_generator": model.sparse_pattern_generator.state_dict(),
    }
    if model.training_stage == 2:
        state["routers"] = model.routers.state_dict()
    return state


def checkpoint_payload(
    model: nn.Module,
    args: argparse.Namespace,
    step: int,
    data_identity: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "format_version": 3,
        "method": "sparmoe_vl_two_stage",
        "protocol": protocol_for_stage(args.stage),
        "stage": args.stage,
        "modality": args.modality,
        "model_name": args.model_name,
        "sparse_layers": list(model.sparse_layers),
        "target_ratio": args.target_ratio,
        "capacity_factors": list(CAPACITY_FACTORS),
        "temperature": args.temperature,
        "training_seed": args.seed,
        "dataset": data_identity,
        "step": step,
        "encoder": encoder_state(model),
        "metrics": metrics,
        "train_args": plain_args(args),
    }
    if args.stage == 2:
        payload["stage1_checkpoint"] = str(args.stage1_checkpoint.resolve())
    study = getattr(args, "study", None)
    if study is not None:
        payload["study"] = str(study)
    study_protocol = getattr(args, "study_protocol", None)
    if study_protocol is not None:
        payload["study_protocol"] = study_protocol
    return payload


@torch.no_grad()
def batch_cosines(
    model: nn.Module,
    batch: Tensor,
    dense: Tensor,
    random_evals: int,
) -> Tuple[float, float]:
    model.eval()
    learned = model(batch, routing_mode="learned").features
    learned_cosine = F.cosine_similarity(learned.float(), dense.float()).mean()
    random_cosine = torch.zeros((), device=batch.device)
    for _ in range(random_evals):
        random_features = model(batch, routing_mode="random").features
        random_cosine += F.cosine_similarity(random_features.float(), dense.float()).mean()
    return float(learned_cosine), float(random_cosine / random_evals)


def run(args: argparse.Namespace) -> None:
    if args.check_only:
        print(json.dumps(plain_args(args), indent=2, ensure_ascii=True))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_reproducible_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output_dir / "train.log")
    model, dataset = build_model_and_data(args, device)
    data_identity = dataset_metadata(
        args.annotations,
        args.data_seed,
        len(dataset),
        dataset.ordered_sha256,
        args.image_root if args.modality == "vision" else None,
    )
    expected_ordered_sha256 = getattr(args, "expected_ordered_sha256", None)
    if (
        expected_ordered_sha256 is not None
        and data_identity["ordered_sha256"] != expected_ordered_sha256
    ):
        raise RuntimeError(
            "training data differs from the registered experiment pool: "
            f"{data_identity['ordered_sha256']} != {expected_ordered_sha256}"
        )

    if args.stage == 2:
        stage1 = load_stage1_checkpoint(args.stage1_checkpoint)
        validate_stage1_checkpoint(
            stage1,
            modality=args.modality,
            model_name=args.model_name,
            sparse_layers=model.sparse_layers,
            target_ratio=args.target_ratio,
            training_seed=args.seed,
            dataset=data_identity,
        )
        study = getattr(args, "study", None)
        if study is not None and stage1.get("study") != study:
            raise ValueError(
                f"stage-1 checkpoint study={stage1.get('study')!r}; expected {study!r}"
            )
        initial_ratios = model.initialize_stage2_from_stage1(stage1)
        logger(
            "stage_transition=validated same_data=true "
            f"ordered_sha256={data_identity['ordered_sha256']} "
            f"initial_layer_ratios={initial_ratios.cpu().tolist()}"
        )

    parameters = model.trainable_parameters()
    if not parameters:
        raise RuntimeError(f"Stage {args.stage} has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.stage == 1:
        objective: nn.Module = Stage1SubspaceObjective(
            target_budget_ratio=args.target_ratio,
        ).to(device)
    else:
        objective = Stage2RouterObjective(
            target_budget_ratio=args.target_ratio,
        ).to(device)

    logger(
        f"method=SparMoE-VL stage={args.stage} modality={args.modality} "
        f"seed={args.seed} data_seed={args.data_seed} samples={len(dataset)} "
        f"ordered_sha256={data_identity['ordered_sha256']} "
        f"batch_size={args.batch_size} trainable={sum(p.numel() for p in parameters):,}"
    )
    loader = make_loader(dataset, args, device)
    iterator = iter(loader)
    best_cosine = float("-inf")
    last_metrics: Dict[str, Any] = {}
    for step in tqdm(range(1, args.steps + 1), desc=f"{args.modality} stage {args.stage}"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            dense = model.dense_features(batch)
        model.train()
        sparse = model(batch, routing_mode="learned")
        if args.stage == 1:
            losses = objective(
                sparse.features,
                dense,
                sparse.base_ratios,
                sparse.retention_ratios,
            )
            budget_value = float(losses.budget.detach())
            last_metrics = {
                "total": float(losses.total.detach()),
                "representation": float(losses.distillation.detach()),
                "budget": budget_value,
                "separation": float(losses.separation.detach()),
                "base_ratio_mean": float(losses.base_ratio_mean.detach()),
                "base_ratios": losses.layer_ratios.detach().cpu().tolist(),
            }
        else:
            warmup = linear_warmup_weight(step, args.router_warmup)
            losses = objective(
                sparse.features,
                dense,
                sparse.routing_loss_inputs,
                sparse.base_ratios,
                sparse.retention_ratios,
                routing_weight=warmup,
            )
            budget_value = float(losses.inherited_budget_error.detach())
            last_metrics = {
                key: (value.detach().cpu().tolist() if value.ndim else float(value.detach()))
                for key, value in losses.as_dict().items()
            }
            last_metrics["routing_warmup"] = warmup

        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            if args.stage == 1:
                # The result-generating Stage-1 runner selected checkpoints
                # from the stochastic pre-update forward above.  Do not run
                # extra routing passes here: they would consume RNG state and
                # change every subsequent optimization step.
                learned_cosine = float(
                    F.cosine_similarity(
                        sparse.features.detach().float(),
                        dense.float(),
                    ).mean()
                )
                last_metrics["learned_cosine"] = learned_cosine
                logger(
                    f"step={step} loss={last_metrics['total']:.5f} "
                    f"learned_cos={learned_cosine:.6f} budget={budget_value:.6f}"
                )
            else:
                learned_cosine, random_cosine = batch_cosines(
                    model, batch, dense, args.random_evals
                )
                last_metrics.update(
                    learned_cosine=learned_cosine,
                    random_cosine=random_cosine,
                    routing_gap=learned_cosine - random_cosine,
                )
                logger(
                    f"step={step} loss={last_metrics['total']:.5f} "
                    f"learned_cos={learned_cosine:.6f} "
                    f"random_cos={random_cosine:.6f} "
                    f"inherited_budget_error={budget_value:.6f}"
                )
            # Only Stage 1 selects against its optimized budget loss. Stage 2
            # inherits a frozen, already validated structure and is selected
            # solely by representation preservation.
            selection_is_valid = (
                budget_value < args.best_budget_loss if args.stage == 1 else True
            )
            if selection_is_valid and learned_cosine > best_cosine:
                best_cosine = learned_cosine
                save_checkpoint_atomic(
                    args.output_dir / "best.pt",
                    checkpoint_payload(model, args, step, data_identity, last_metrics),
                )
            if args.stage == 2:
                model.train()
        if step % args.save_every == 0:
            save_checkpoint_atomic(
                args.output_dir / f"step_{step:06d}.pt",
                checkpoint_payload(model, args, step, data_identity, last_metrics),
            )

    save_checkpoint_atomic(
        args.output_dir / "final.pt",
        checkpoint_payload(model, args, args.steps, data_identity, last_metrics),
    )
    if best_cosine == float("-inf"):
        criterion = "the configured budget constraint" if args.stage == 1 else "selection"
        raise RuntimeError(f"no best checkpoint met {criterion}")
    logger(f"finished best_cosine={best_cosine:.6f}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

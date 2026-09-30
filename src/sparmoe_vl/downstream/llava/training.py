"""Stage-1 SPG learning then router-only CLIP-336 conversion for LLaVA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...common.data import ShareGPT4VImageTrainingDataset
from ...common.losses import symmetric_log_ratio
from ...common.training import (
    RunLogger,
    dataset_metadata,
    plain_args,
    save_checkpoint_atomic,
    seed_worker,
    set_reproducible_seed,
)
from ...common.two_stage import protocol_for_stage
from .protocol import (
    CAPACITY_FACTORS,
    DATA_SEED,
    DEFAULT_ANNOTATIONS,
    DEFAULT_CLIP,
    DEFAULT_IMAGE_ROOT,
    MODEL_KEY,
    MODEL_NAME,
    POOL_SIZE,
    SEEDS,
    STAGE_SETTINGS,
    TARGET_RATIO,
    TRAINING_DATASET_SHA256,
    checkpoint_metadata,
    torch_load,
)
from .vision import NUM_LAYERS, SparMoECLIP336VisionTower, load_local_clip_vision


def parse_args(
    stage: int,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    settings = STAGE_SETTINGS[stage]
    parser = argparse.ArgumentParser(
        description=f"Train Stage {stage} of the paper's CLIP-336 LLaVA transfer."
    )
    parser.add_argument("--clip-path", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--data-seed", type=int, default=DATA_SEED)
    parser.add_argument("--max-samples", type=int, default=POOL_SIZE)
    parser.add_argument("--target-ratio", type=float, default=TARGET_RATIO)
    parser.add_argument("--steps", type=int, default=settings["steps"])
    parser.add_argument("--batch-size", type=int, default=settings["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=settings["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=settings["weight_decay"])
    parser.add_argument("--temperature", type=float, default=settings["temperature"])
    parser.add_argument("--num-workers", type=int, default=settings["num_workers"])
    parser.add_argument("--log-every", type=int, default=settings["log_every"])
    parser.add_argument("--save-every", type=int, default=settings["save_every"])
    parser.add_argument("--dry-run", action="store_true")
    if stage == 1:
        parser.add_argument(
            "--best-budget-loss", type=float, default=settings["best_budget_loss"]
        )
    else:
        parser.add_argument("--stage1-checkpoint", type=Path, required=True)
        parser.add_argument("--router-warmup", type=int, default=settings["router_warmup"])
        parser.add_argument("--random-evals", type=int, default=settings["random_evals"])
    return parser.parse_args(argv)


def validate_training_protocol(args: argparse.Namespace, stage: int) -> None:
    settings = STAGE_SETTINGS[stage]
    checks = {
        "data_seed": (args.data_seed, DATA_SEED),
        "max_samples": (args.max_samples, POOL_SIZE),
        "target_ratio": (args.target_ratio, TARGET_RATIO),
        "steps": (args.steps, settings["steps"]),
        "batch_size": (args.batch_size, settings["batch_size"]),
        "learning_rate": (args.learning_rate, settings["learning_rate"]),
        "weight_decay": (args.weight_decay, settings["weight_decay"]),
        "temperature": (args.temperature, settings["temperature"]),
        "num_workers": (args.num_workers, settings["num_workers"]),
        "log_every": (args.log_every, settings["log_every"]),
        "save_every": (args.save_every, settings["save_every"]),
    }
    if stage == 1:
        checks["best_budget_loss"] = (
            args.best_budget_loss,
            settings["best_budget_loss"],
        )
    else:
        checks.update(
            router_warmup=(args.router_warmup, settings["router_warmup"]),
            random_evals=(args.random_evals, settings["random_evals"]),
        )
    for name, (actual, expected) in checks.items():
        matches = (
            abs(float(actual) - float(expected)) <= 1e-12
            if isinstance(expected, float)
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"Stage {stage} {name}={actual}; paper protocol requires {expected}"
            )


def hidden_alignment(sparse_states: Sequence[Tensor], dense_states: Sequence[Tensor]) -> Tensor:
    if len(sparse_states) != len(dense_states) or len(sparse_states) != NUM_LAYERS + 1:
        raise ValueError("CLIP-336 hidden-state sequences do not match")
    losses = []
    for sparse, dense in zip(sparse_states[1:], dense_states[1:]):
        sparse_patch = F.normalize(sparse[:, 1:].float(), dim=-1)
        dense_patch = F.normalize(dense[:, 1:].float().detach(), dim=-1)
        losses.append(1.0 - (sparse_patch * dense_patch).sum(dim=-1).mean())
    return torch.stack(losses).mean()


def pooled_alignment(sparse: Tensor, dense: Tensor) -> Tensor:
    sparse = F.normalize(sparse.float(), dim=-1)
    dense = F.normalize(dense.float().detach(), dim=-1)
    return 1.0 - (sparse * dense).sum(dim=-1).mean()


def stage1_regularization(
    output: Any,
    target_ratio: float,
) -> tuple[Tensor, dict[str, Any]]:
    budget = symmetric_log_ratio(
        output.base_ratios.mean(),
        torch.tensor(target_ratio, device=output.base_ratios.device),
    )
    gaps = output.retention_ratios[:, 1:] - output.retention_ratios[:, :-1]
    separation = F.relu(0.03 - gaps).mean()
    regularization = 50.0 * budget + separation
    return regularization, {
        "budget": float(budget.detach()),
        "separation": float(separation.detach()),
        "layer_ratios": output.base_ratios.detach().cpu().tolist(),
    }


def stage2_regularization(
    output: Any,
    *,
    routing_weight: float,
) -> tuple[Tensor, dict[str, Any]]:
    routing = torch.stack(
        [
            F.cross_entropy(layer.routing.logits.float(), layer.router_targets)
            for layer in output.layers
        ]
    ).mean()
    router_accuracy = torch.stack(
        [
            (layer.routing.logits.argmax(dim=-1) == layer.router_targets).float().mean()
            for layer in output.layers
        ]
    ).mean()
    budget = symmetric_log_ratio(
        output.base_ratios.mean(),
        torch.tensor(TARGET_RATIO, device=output.base_ratios.device),
    )
    gaps = output.retention_ratios[:, 1:] - output.retention_ratios[:, :-1]
    separation = F.relu(0.03 - gaps).mean()
    usage = torch.stack(
        [layer.routing.gates.float().mean(dim=0) for layer in output.layers]
    ).mean(dim=0)
    expected = torch.stack(
        [
            (
                layer.routing.gates.float().mean(dim=0) * layer.sparse_pattern.retention_ratios
            ).sum()
            for layer in output.layers
        ]
    ).mean()
    regularization = routing_weight * routing
    return regularization, {
        "routing": float(routing.detach()),
        "router_accuracy": float(router_accuracy.detach()),
        "inherited_budget_error": float(budget.detach()),
        "inherited_separation": float(separation.detach()),
        "base_ratio_mean": float(output.base_ratios.mean().detach()),
        "expected_ratio_mean": float(expected.detach()),
        "expert_usage": usage.detach().cpu().tolist(),
        "base_ratios": output.base_ratios.detach().cpu().tolist(),
    }


def checkpoint_payload(
    model: SparMoECLIP336VisionTower,
    args: argparse.Namespace,
    stage: int,
    step: int,
    data_identity: Mapping[str, Any],
    metrics: Mapping[str, Any],
    stage1_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "format_version": 3,
        "method": "sparmoe_vl_two_stage",
        "protocol": protocol_for_stage(stage),
        "stage": stage,
        "modality": "vision",
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
        "sparse_layers": list(range(NUM_LAYERS)),
        "target_ratio": TARGET_RATIO,
        "capacity_factors": list(CAPACITY_FACTORS),
        "temperature": args.temperature,
        "training_seed": args.seed,
        "dataset": dict(data_identity),
        "step": step,
        "encoder": model.encoder_state(),
        "metrics": dict(metrics),
        "train_args": plain_args(args),
    }
    if stage1_metadata is not None:
        payload["stage1_checkpoint"] = str(args.stage1_checkpoint.resolve())
        payload["stage1_step"] = stage1_metadata["checkpoint_step"]
    return payload


def save_training_checkpoint(
    path: Path,
    model: SparMoECLIP336VisionTower,
    args: argparse.Namespace,
    stage: int,
    step: int,
    data_identity: Mapping[str, Any],
    metrics: Mapping[str, Any],
    stage1_metadata: Mapping[str, Any] | None,
) -> None:
    save_checkpoint_atomic(
        path,
        checkpoint_payload(
            model,
            args,
            stage,
            step,
            data_identity,
            metrics,
            stage1_metadata,
        ),
    )


def validate_stage1_transition(
    checkpoint: Mapping[str, Any],
    args: argparse.Namespace,
    data_identity: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = checkpoint_metadata(checkpoint, expected_stage=1)
    if metadata["training_seed"] != args.seed:
        raise ValueError("Stage 1 and Stage 2 training seeds differ")
    checks = {
        "data_seed": data_identity["data_seed"],
        "pool_size": data_identity["samples"],
        "dataset_sha256": data_identity["ordered_sha256"],
    }
    for key, expected in checks.items():
        if metadata[key] != expected:
            raise ValueError(
                f"Stage 1 and Stage 2 data differ: {key}={metadata[key]!r}; "
                f"expected {expected!r}"
            )
    return metadata


def image_transform(processor: Any):
    def transform(image: Any) -> Tensor:
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    return transform


def run_training(stage: int, argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(stage, argv)
    validate_training_protocol(args, stage)
    if args.dry_run:
        print(json.dumps(plain_args(args), indent=2, ensure_ascii=True))
        return
    for path, label in (
        (args.clip_path / "config.json", "CLIP-336 config"),
        (args.clip_path / "pytorch_model.bin", "CLIP-336 weights"),
        (args.annotations, "ShareGPT4V annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"missing ShareGPT4V image root: {args.image_root}")
    if stage == 2 and not args.stage1_checkpoint.is_file():
        raise FileNotFoundError(f"missing Stage-1 checkpoint: {args.stage1_checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_reproducible_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output_dir / "train.log")
    try:
        from transformers import CLIPImageProcessor
    except ImportError as error:
        raise RuntimeError("install the llava optional dependencies") from error
    processor = CLIPImageProcessor.from_pretrained(
        str(args.clip_path),
        local_files_only=True,
    )
    clip = load_local_clip_vision(args.clip_path)
    model = SparMoECLIP336VisionTower(
        clip,
        training_stage=stage,
        target_ratio=TARGET_RATIO,
        capacity_factors=CAPACITY_FACTORS,
        router_temperature=args.temperature,
        mask_temperature=args.temperature,
    ).to(device)
    dataset = ShareGPT4VImageTrainingDataset(
        args.annotations,
        args.image_root,
        image_transform(processor),
        max_samples=args.max_samples,
        data_seed=args.data_seed,
    )
    data_identity = dataset_metadata(
        args.annotations,
        args.data_seed,
        len(dataset),
        dataset.ordered_sha256,
        args.image_root,
    )
    if data_identity["samples"] != POOL_SIZE:
        raise RuntimeError(f"the paper protocol requires exactly {POOL_SIZE:,} samples")
    if data_identity["ordered_sha256"] != TRAINING_DATASET_SHA256:
        raise RuntimeError("CLIP-336 training data differ from the visual main experiment")
    stage1_metadata = None
    if stage == 2:
        stage1_checkpoint = torch_load(args.stage1_checkpoint)
        stage1_metadata = validate_stage1_transition(
            stage1_checkpoint,
            args,
            data_identity,
        )
        model.initialize_stage2_from_stage1(stage1_checkpoint)
        del stage1_checkpoint
    model.train()

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    iterator = iter(loader)
    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_alignment = float("inf")
    best_step = None
    last_metrics: dict[str, Any] = {}
    logger(
        f"study=llava_transfer stage={stage} model={MODEL_NAME} "
        f"seed={args.seed} data_seed={args.data_seed} samples={len(dataset)} "
        f"ordered_sha256={dataset.ordered_sha256} target_ratio={TARGET_RATIO} "
        f"capacity_factors={list(CAPACITY_FACTORS)} batch_size={args.batch_size} "
        f"alignment=all_hidden_patch_tokens_plus_final_pooled_feature "
        f"trainable_parameters={sum(parameter.numel() for parameter in parameters):,}"
    )

    for step in tqdm(range(1, args.steps + 1), desc=f"CLIP-336 Stage {stage}"):
        try:
            pixel_values = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            pixel_values = next(iterator)
        pixel_values = pixel_values.to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            dense = model.dense_forward(pixel_values)
        sparse = model(pixel_values, output_hidden_states=True)
        hidden_loss = hidden_alignment(sparse.hidden_states, dense.hidden_states)
        pooled_loss = pooled_alignment(sparse.pooler_output, dense.pooler_output)
        alignment = hidden_loss + pooled_loss
        if stage == 1:
            regularization, regularization_stats = stage1_regularization(
                sparse,
                TARGET_RATIO,
            )
        else:
            routing_weight = min(1.0, step / max(args.router_warmup, 1))
            regularization, regularization_stats = stage2_regularization(
                sparse,
                routing_weight=routing_weight,
            )
        loss = 100.0 * alignment + regularization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            if stage == 1:
                measured_alignment = float(alignment.detach())
                reported_hidden_states = sparse.hidden_states
                pooled_cosine = float(
                    F.cosine_similarity(
                        sparse.pooler_output.float(),
                        dense.pooler_output.float(),
                        dim=-1,
                    )
                    .mean()
                    .detach()
                )
                random_alignment = None
            else:
                model.eval()
                with torch.no_grad():
                    learned = model(pixel_values, output_hidden_states=True)
                    measured_alignment = float(
                        hidden_alignment(learned.hidden_states, dense.hidden_states)
                        + pooled_alignment(learned.pooler_output, dense.pooler_output)
                    )
                    pooled_cosine = float(
                        F.cosine_similarity(
                            learned.pooler_output.float(),
                            dense.pooler_output.float(),
                            dim=-1,
                        ).mean()
                    )
                    reported_hidden_states = learned.hidden_states
                    random_alignment = 0.0
                    for _ in range(args.random_evals):
                        random_output = model(
                            pixel_values,
                            routing_mode="random",
                            output_hidden_states=True,
                        )
                        random_alignment += (
                            float(
                                hidden_alignment(
                                    random_output.hidden_states,
                                    dense.hidden_states,
                                )
                                + pooled_alignment(
                                    random_output.pooler_output,
                                    dense.pooler_output,
                                )
                            )
                            / args.random_evals
                        )
                model.train()
            layer22_cosine = float(
                (
                    F.normalize(reported_hidden_states[-2][:, 1:].float(), dim=-1)
                    * F.normalize(dense.hidden_states[-2][:, 1:].float(), dim=-1)
                )
                .sum(dim=-1)
                .mean()
                .detach()
            )
            last_metrics = {
                "loss": float(loss.detach()),
                "alignment": measured_alignment,
                "train_alignment": float(alignment.detach()),
                "hidden_alignment": float(hidden_loss.detach()),
                "pooled_alignment": float(pooled_loss.detach()),
                "pooled_cosine": pooled_cosine,
                "layer22_patch_cosine": layer22_cosine,
                "random_alignment": random_alignment,
                **regularization_stats,
            }
            budget_value = (
                regularization_stats["budget"]
                if stage == 1
                else regularization_stats["inherited_budget_error"]
            )
            budget_is_valid = budget_value < args.best_budget_loss if stage == 1 else True
            if budget_is_valid and measured_alignment < best_alignment:
                best_alignment = measured_alignment
                best_step = step
                save_training_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    args,
                    stage,
                    step,
                    data_identity,
                    last_metrics,
                    stage1_metadata,
                )
            logger(
                f"step={step} loss={float(loss.detach()):.6f} "
                f"alignment={measured_alignment:.6f} budget={budget_value:.6f} "
                f"pooled_cosine={pooled_cosine:.6f} "
                f"layer22_patch_cosine={layer22_cosine:.6f} "
                f"best_step={best_step}"
            )
        if step % args.save_every == 0:
            save_training_checkpoint(
                args.output_dir / f"step_{step:05d}.pt",
                model,
                args,
                stage,
                step,
                data_identity,
                last_metrics,
                stage1_metadata,
            )

    save_training_checkpoint(
        args.output_dir / "final.pt",
        model,
        args,
        stage,
        args.steps,
        data_identity,
        last_metrics,
        stage1_metadata,
    )
    if best_step is None:
        raise RuntimeError(f"Stage {stage} produced no valid best checkpoint")
    logger(f"complete best_step={best_step} best_alignment={best_alignment:.6f}")


def main(stage: int, argv: Optional[Sequence[str]] = None) -> None:
    run_training(stage, argv)

"""Distributed, fixed-protocol two-stage recovery for visual MoPE."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.nn.parallel import DistributedDataParallel

from .common import (
    ANNOTATIONS,
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_PROCESSING_ORDER_SHA256,
    IMAGE_ROOT,
    PRETRAINED,
    atomic_torch_save,
    build_main_image_text_pool,
    create_clip,
    file_sha256,
    make_image_slice_loader,
    seed_everything,
)
from .mope import (
    CHECKPOINT_METHOD,
    GLOBAL_BATCH_SIZE,
    GROUP_SIZE,
    HiddenCapture,
    PER_GPU_BATCH_SIZE,
    RETAINED_WIDTH,
    STAGE_EXPOSURES,
    STEPS_PER_STAGE,
    TARGET_FFN_REDUCTION,
    TEXT_FEATURE_CACHE,
    TOTAL_EXPOSURES,
    WORLD_SIZE,
    cosine_warmup_lambda,
    kept_channel_indices,
    load_visual_state_dict,
    recovery_data_manifest,
    recovery_rank_indices,
    selected_groups,
    soft_cross_entropy,
    statistics,
    structurally_prune_visual_ffn,
    validate_dense_text_cache,
    validate_pruned_visual,
    validate_recovery_data_manifest,
    validate_selection,
    visual_state_dict,
)


def parse_args(stage: int, argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=tuple(EXPECTED_PROCESSING_ORDER_SHA256),
    )
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--text-feature-cache", type=Path, default=TEXT_FEATURE_CACHE)
    parser.add_argument("--steps", type=int, default=STEPS_PER_STAGE)
    parser.add_argument("--per-gpu-batch-size", type=int, default=PER_GPU_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1000.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    args.stage = stage
    if stage not in (1, 2):
        parser.error("stage must be 1 or 2")
    if stage == 2 and args.stage1_checkpoint is None:
        parser.error("Stage 2 requires --stage1-checkpoint")
    if args.steps != STEPS_PER_STAGE or args.per_gpu_batch_size != PER_GPU_BATCH_SIZE:
        parser.error("formal MoPE recovery fixes 1,954 steps and batch 32/GPU per stage")
    if (args.learning_rate, args.weight_decay, args.warmup_ratio) != (2e-5, 3e-4, 0.1):
        parser.error("formal MoPE recovery fixes lr=2e-5, decay=3e-4, warmup=0.1")
    if (args.alpha, args.beta, args.gamma) != (1.0, 1000.0, 1.0):
        parser.error("formal MoPE loss weights are (alpha,beta,gamma)=(1,1000,1)")
    if args.workers < 0 or args.save_every <= 0 or args.log_every <= 0:
        parser.error("worker and progress settings are invalid")
    return args


def optimizer_to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: optimizer_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [optimizer_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(optimizer_to_cpu(item) for item in value)
    return value


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def gather_without_gradient(tensor: Tensor, world_size: int) -> Tensor:
    if world_size == 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def distributed_cross_modal_losses(
    student_image: Tensor,
    teacher_image: Tensor,
    text_features: Tensor,
    scale: Tensor,
    rank: int,
    world_size: int,
) -> tuple[Tensor, Tensor]:
    """Compute symmetric CLIP and teacher-similarity losses across all ranks."""

    student_images = torch.cat(differentiable_all_gather(student_image), dim=0)
    teacher_images = gather_without_gradient(teacher_image, world_size)
    all_text = gather_without_gradient(text_features, world_size)
    labels = torch.arange(student_image.shape[0], device=student_image.device)
    labels += rank * student_image.shape[0]
    student_i2t = scale * student_image @ all_text.T
    student_t2i = scale * text_features @ student_images.T
    teacher_i2t = scale * teacher_image @ all_text.T
    teacher_t2i = scale * text_features @ teacher_images.T
    contrastive = 0.5 * (
        F.cross_entropy(student_i2t, labels) + F.cross_entropy(student_t2i, labels)
    )
    similarity = 0.5 * (
        soft_cross_entropy(student_i2t, teacher_i2t.detach())
        + soft_cross_entropy(student_t2i, teacher_t2i.detach())
    )
    return contrastive, similarity


def validate_transition_checkpoint(
    checkpoint: Mapping[str, Any],
    args: argparse.Namespace,
    data_manifest: Mapping[str, Any],
    selection_sha256: str,
    kept_groups: list[int],
) -> None:
    expected = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": 1,
        "stage_step": STEPS_PER_STAGE,
        "global_step": STEPS_PER_STAGE,
        "seed": args.seed,
        "data_manifest": dict(data_manifest),
        "selection_sha256": selection_sha256,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "group_size": GROUP_SIZE,
        "kept_group_indices": kept_groups,
        "retained_width": RETAINED_WIDTH,
        "target_ffn_reduction": TARGET_FFN_REDUCTION,
        "actual_ffn_reduction": TARGET_FFN_REDUCTION,
        "statistics": statistics(),
        "complete": True,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    for key in (
        "visual_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
    ):
        if key not in checkpoint:
            mismatches[key] = ("missing", "present")
    if mismatches:
        raise ValueError(f"invalid MoPE Stage-1 transition checkpoint: {mismatches}")


def validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    args: argparse.Namespace,
    data_manifest: Mapping[str, Any],
    selection_sha256: str,
    kept_groups: list[int],
) -> None:
    expected = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": args.stage,
        "seed": args.seed,
        "data_manifest": dict(data_manifest),
        "selection_sha256": selection_sha256,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "group_size": GROUP_SIZE,
        "kept_group_indices": kept_groups,
        "retained_width": RETAINED_WIDTH,
        "target_ffn_reduction": TARGET_FFN_REDUCTION,
        "actual_ffn_reduction": TARGET_FFN_REDUCTION,
        "statistics": statistics(),
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    stage_step = checkpoint.get("stage_step")
    global_step = checkpoint.get("global_step")
    expected_global = (args.stage - 1) * STEPS_PER_STAGE + int(stage_step or 0)
    if not isinstance(stage_step, int) or not 0 <= stage_step <= STEPS_PER_STAGE:
        mismatches["stage_step"] = (stage_step, f"0..{STEPS_PER_STAGE}")
    if global_step != expected_global:
        mismatches["global_step"] = (global_step, expected_global)
    for key in (
        "visual_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
    ):
        if key not in checkpoint:
            mismatches[key] = ("missing", "present")
    if mismatches:
        raise ValueError(f"invalid MoPE resume checkpoint: {mismatches}")


def checkpoint_payload(
    student: nn.Module,
    args: argparse.Namespace,
    data_manifest: Mapping[str, Any],
    selection_sha256: str,
    kept_groups: list[int],
    stage_step: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, Any]:
    payload = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": args.stage,
        "stage_step": stage_step,
        "global_step": global_step,
        "seed": args.seed,
        "data_manifest": dict(data_manifest),
        "selection": str(args.selection.resolve()),
        "selection_sha256": selection_sha256,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "group_size": GROUP_SIZE,
        "kept_group_indices": kept_groups,
        "retained_width": RETAINED_WIDTH,
        "target_ffn_reduction": TARGET_FFN_REDUCTION,
        "actual_ffn_reduction": TARGET_FFN_REDUCTION,
        "loss_weights": {
            "contrastive": 1.0,
            "similarity_distillation": args.alpha,
            "feature_distillation": args.beta,
            "hidden_distillation": args.gamma,
        },
        "visual_state_dict": visual_state_dict(student),
        "statistics": statistics(),
        "complete": stage_step == STEPS_PER_STAGE,
    }
    if optimizer is not None and scheduler is not None and scaler is not None:
        payload["optimizer_state_dict"] = optimizer_to_cpu(optimizer.state_dict())
        payload["scheduler_state_dict"] = scheduler.state_dict()
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def run(stage: int, argv: Sequence[str] | None = None) -> None:
    args = parse_args(stage, argv)
    data_manifest = recovery_data_manifest(args.seed)
    configuration = {
        "method": CHECKPOINT_METHOD,
        "stage": stage,
        "seed": args.seed,
        "data_manifest": data_manifest,
        "stage_exposures": STAGE_EXPOSURES,
        "two_stage_exposures": TOTAL_EXPOSURES,
        "stage_sequences_identical": True,
        "selection": str(args.selection.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.check_only:
        print(json.dumps(configuration, indent=2))
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != WORLD_SIZE or not 0 <= rank < world_size:
        raise RuntimeError("formal visual MoPE recovery requires torchrun with exactly 8 ranks")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing full visual MoPE recovery")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        _run_distributed(args, configuration, data_manifest, rank, local_rank, world_size)
    finally:
        dist.destroy_process_group()


def _run_distributed(
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
    data_manifest: dict[str, Any],
    rank: int,
    local_rank: int,
    world_size: int,
) -> None:
    is_main = rank == 0
    device = torch.device("cuda", local_rank)
    if file_sha256(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")
    selection_sha256 = file_sha256(args.selection)
    selection = torch.load(args.selection, map_location="cpu", weights_only=False)
    validate_selection(selection)
    cache = torch.load(args.text_feature_cache, map_location="cpu", weights_only=False)
    validate_dense_text_cache(cache)
    text_features = cache["features"]
    paths, _captions, pool_manifest = build_main_image_text_pool(
        args.annotations,
        args.image_root,
    )
    if (
        pool_manifest["dataset_sha256"] != data_manifest["dataset_sha256"]
        or pool_manifest["paired_caption_sha256"] != data_manifest["paired_caption_sha256"]
    ):
        raise RuntimeError("current visual pairs differ from the MoPE recovery manifest")
    validate_recovery_data_manifest(data_manifest)

    seed_everything(args.seed)
    teacher, preprocess, _ = create_clip(device, args.pretrained)
    student, _, _ = create_clip(device, args.pretrained)
    kept_groups = selected_groups(selection)
    retained = kept_channel_indices(selection["groups"], kept_groups)
    structurally_prune_visual_ffn(student, retained)
    validate_pruned_visual(student)
    teacher.requires_grad_(False).eval()
    student.requires_grad_(False)
    for parameter in student.visual.parameters():
        parameter.requires_grad_(True)
    student.visual.train()

    train_visual = DistributedDataParallel(
        student.visual,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )
    parameters = [
        parameter for parameter in train_visual.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(step, 2 * STEPS_PER_STAGE),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=256.0,
        growth_interval=2 * STEPS_PER_STAGE + 1,
        enabled=args.amp,
    )

    stage_step = 0
    global_step = (args.stage - 1) * STEPS_PER_STAGE
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(
            checkpoint,
            args,
            data_manifest,
            selection_sha256,
            kept_groups,
        )
        load_visual_state_dict(student, checkpoint["visual_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        stage_step = int(checkpoint["stage_step"])
        global_step = int(checkpoint["global_step"])
    elif args.stage == 2:
        checkpoint = torch.load(
            args.stage1_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        validate_transition_checkpoint(
            checkpoint,
            args,
            data_manifest,
            selection_sha256,
            kept_groups,
        )
        load_visual_state_dict(student, checkpoint["visual_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    rank_indices = recovery_rank_indices(args.seed, rank, world_size)
    sample_offset = stage_step * PER_GPU_BATCH_SIZE
    loader = make_image_slice_loader(
        paths,
        rank_indices[sample_offset:],
        preprocess,
        PER_GPU_BATCH_SIZE,
        args.workers,
        str(device),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.jsonl"
    if is_main:
        if args.resume is None:
            log_path.write_text("", encoding="utf-8")
        append_jsonl(
            log_path,
            {
                "event": "protocol",
                **configuration,
                "same_sequence_as_other_stage": True,
                "stage_start_step": stage_step,
                "global_start_step": global_step,
                "retained_width": RETAINED_WIDTH,
            },
        )

    teacher_hidden = HiddenCapture(teacher)
    student_hidden = HiddenCapture(student)
    scale = teacher.logit_scale.exp().detach().clamp(max=100)
    running = {key: 0.0 for key in ("loss", "itc", "similarity", "feature", "hidden")}
    running_steps = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    try:
        for offset, images in enumerate(loader):
            local_index = stage_step + offset
            start = local_index * PER_GPU_BATCH_SIZE
            batch_indices = rank_indices[start : start + images.shape[0]].to(torch.int64)
            images = images.to(device, non_blocking=True)
            paired_text = text_features[batch_indices].to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )
            teacher_hidden.clear()
            student_hidden.clear()
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                with torch.no_grad():
                    teacher_image = F.normalize(teacher.visual(images), dim=-1)
                student_image = F.normalize(train_visual(images), dim=-1)
                itc, similarity = distributed_cross_modal_losses(
                    student_image,
                    teacher_image,
                    paired_text,
                    scale,
                    rank,
                    world_size,
                )
                feature = 0.5 * F.mse_loss(student_image.float(), teacher_image.float())
                if len(teacher_hidden.outputs) != len(student_hidden.outputs):
                    raise RuntimeError("MoPE teacher/student hidden capture count differs")
                hidden = 0.5 * sum(
                    F.mse_loss(student_state.float(), teacher_state.float())
                    for student_state, teacher_state in zip(
                        student_hidden.outputs,
                        teacher_hidden.outputs,
                    )
                )
                loss = itc + args.alpha * similarity + args.beta * feature + args.gamma * hidden

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                raise FloatingPointError("AMP overflow skipped a fixed-protocol MoPE update")
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            stage_step = local_index + 1
            global_step += 1
            running_steps += 1
            for key, value in {
                "loss": loss,
                "itc": itc,
                "similarity": similarity,
                "feature": feature,
                "hidden": hidden,
            }.items():
                running[key] += float(value.detach())

            if is_main and (stage_step % args.log_every == 0 or stage_step == STEPS_PER_STAGE):
                averages = {key: value / running_steps for key, value in running.items()}
                append_jsonl(
                    log_path,
                    {
                        "event": "train",
                        "stage": args.stage,
                        "stage_step": stage_step,
                        "global_step": global_step,
                        "global_samples_in_stage": min(
                            STAGE_EXPOSURES,
                            stage_step * GLOBAL_BATCH_SIZE,
                        ),
                        "lr": optimizer.param_groups[0]["lr"],
                        **averages,
                        "elapsed_seconds": time.time() - started,
                    },
                )
                print(
                    f"stage={args.stage} step={stage_step}/{STEPS_PER_STAGE} "
                    f"loss={averages['loss']:.6f}",
                    flush=True,
                )
                running = {key: 0.0 for key in running}
                running_steps = 0

            if is_main and stage_step % args.save_every == 0:
                atomic_torch_save(
                    checkpoint_payload(
                        student,
                        args,
                        data_manifest,
                        selection_sha256,
                        kept_groups,
                        stage_step,
                        global_step,
                        optimizer,
                        scheduler,
                        scaler,
                    ),
                    args.output_dir / "latest.pt",
                )
    finally:
        teacher_hidden.close()
        student_hidden.close()

    if stage_step != STEPS_PER_STAGE:
        raise RuntimeError(f"MoPE Stage {args.stage} stopped at {stage_step}/{STEPS_PER_STAGE}")
    dist.barrier()
    if is_main:
        final_path = args.output_dir / f"stage{args.stage}.pt"
        keep_training_state = args.stage == 1
        atomic_torch_save(
            checkpoint_payload(
                student,
                args,
                data_manifest,
                selection_sha256,
                kept_groups,
                stage_step,
                global_step,
                optimizer if keep_training_state else None,
                scheduler if keep_training_state else None,
                scaler if keep_training_state else None,
            ),
            final_path,
        )
        append_jsonl(
            log_path,
            {
                "event": "complete",
                "stage": args.stage,
                "stage_step": stage_step,
                "global_step": global_step,
                "stage_exposures": STAGE_EXPOSURES,
                "checkpoint": str(final_path.resolve()),
                "elapsed_seconds": time.time() - started,
            },
        )
    dist.barrier()

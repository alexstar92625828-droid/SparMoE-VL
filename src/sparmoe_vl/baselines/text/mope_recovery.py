"""Fixed-protocol two-stage recovery runner for the MoPE text baseline."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .common import (
    EXPECTED_PRETRAINED_SHA256,
    IMAGE_FEATURE_CACHE,
    PRETRAINED,
    TOKEN_CACHE,
    create_clip,
    file_sha256,
    prepare_token_cache,
    seed_everything,
    validate_image_feature_cache,
)
from .mope import (
    CHECKPOINT_METHOD,
    GROUP_SIZE,
    HiddenCapture,
    STAGE_EXPOSURES,
    TARGET_KEEP_GROUPS,
    TARGET_REDUCTIONS,
    TOTAL_EXPOSURES,
    TRAIN_BATCH_SIZE,
    TRAIN_STEPS_PER_STAGE,
    cosine_warmup_lambda,
    cross_modal_losses,
    kept_channel_indices,
    load_text_state_dict,
    one_stage_exposure_indices,
    recovery_data_manifest,
    selected_groups,
    statistics,
    structurally_prune_text_ffn,
    text_parameters,
    text_state_dict,
    validate_pruned_text,
    validate_recovery_data_manifest,
    validate_selection,
)


def parse_args(stage: int, argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, required=True, choices=tuple(TARGET_KEEP_GROUPS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--token-cache", type=Path, default=TOKEN_CACHE)
    parser.add_argument("--image-feature-cache", type=Path, default=IMAGE_FEATURE_CACHE)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS_PER_STAGE)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1000.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    args.stage = stage
    if stage not in (1, 2):
        parser.error("stage must be 1 or 2")
    if stage == 2 and args.stage1_checkpoint is None:
        parser.error("Stage 2 requires --stage1-checkpoint")
    if args.steps != TRAIN_STEPS_PER_STAGE or args.batch_size != TRAIN_BATCH_SIZE:
        parser.error("formal MoPE recovery fixes 5,000 steps and batch size 256 per stage")
    if (args.learning_rate, args.weight_decay, args.warmup_ratio) != (2e-5, 3e-4, 0.1):
        parser.error("formal MoPE recovery fixes lr=2e-5, decay=3e-4, warmup=0.1")
    if (args.alpha, args.beta, args.gamma) != (1.0, 1000.0, 1.0):
        parser.error("formal MoPE loss weights are (alpha,beta,gamma)=(1,1000,1)")
    if args.save_every <= 0 or args.log_every <= 0:
        parser.error("progress intervals must be positive")
    return args


def optimizer_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: optimizer_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [optimizer_to_cpu(item) for item in value]
    return value


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def validate_transition_checkpoint(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    data_manifest: dict[str, Any],
    selection_sha256: str,
) -> None:
    expected = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": 1,
        "stage_step": TRAIN_STEPS_PER_STAGE,
        "global_step": TRAIN_STEPS_PER_STAGE,
        "seed": args.seed,
        "data_manifest": data_manifest,
        "selection_sha256": selection_sha256,
        "retained_width": TARGET_KEEP_GROUPS[args.seed] * GROUP_SIZE,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    required_state = ("text_state_dict", "optimizer_state_dict", "scheduler_state_dict")
    for key in required_state:
        if key not in checkpoint:
            mismatches[key] = ("missing", "present")
    if mismatches:
        raise ValueError(f"invalid MoPE Stage-1 transition checkpoint: {mismatches}")


def checkpoint_payload(
    student: torch.nn.Module,
    args: argparse.Namespace,
    data_manifest: dict[str, Any],
    selection_sha256: str,
    kept_groups: list[int],
    stage_step: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, Any]:
    retained_width = len(kept_groups) * GROUP_SIZE
    payload = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": args.stage,
        "stage_step": stage_step,
        "global_step": global_step,
        "seed": args.seed,
        "data_manifest": data_manifest,
        "selection": str(args.selection.resolve()),
        "selection_sha256": selection_sha256,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "group_size": GROUP_SIZE,
        "kept_group_indices": kept_groups,
        "retained_width": retained_width,
        "target_ffn_reduction": TARGET_REDUCTIONS[args.seed],
        "actual_ffn_reduction": 1.0 - retained_width / 3072,
        "loss_weights": {
            "contrastive": 1.0,
            "similarity_distillation": args.alpha,
            "feature_distillation": args.beta,
            "hidden_distillation": args.gamma,
        },
        "text_state_dict": text_state_dict(student),
        "statistics": statistics(retained_width),
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
        "selection": str(args.selection.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.check_only:
        print(json.dumps(configuration, indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing full MoPE recovery on CPU")
    if file_sha256(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise RuntimeError("Dense CLIP checkpoint SHA-256 mismatch")

    selection_sha256 = file_sha256(args.selection)
    selection = torch.load(args.selection, map_location="cpu", weights_only=False)
    validate_selection(selection)
    token_cache = prepare_token_cache(args.token_cache)
    image_cache = torch.load(args.image_feature_cache, map_location="cpu", weights_only=False)
    validate_image_feature_cache(image_cache)
    tokens = token_cache["tokens"]
    image_features = image_cache["features"]
    indices = one_stage_exposure_indices(args.seed)
    validate_recovery_data_manifest(data_manifest)

    seed_everything(args.seed)
    device = torch.device(args.device)
    teacher, _, _ = create_clip(device, args.pretrained)
    student, _, _ = create_clip(device, args.pretrained)
    kept_groups = selected_groups(selection, args.seed)
    retained = kept_channel_indices(selection["groups"], kept_groups)
    structurally_prune_text_ffn(student, retained)
    retained_width = len(kept_groups) * GROUP_SIZE
    validate_pruned_text(student, retained_width)

    teacher.requires_grad_(False).eval()
    student.requires_grad_(False)
    parameters = text_parameters(student)
    for parameter in parameters:
        parameter.requires_grad_(True)
    student.logit_scale.requires_grad_(False)
    student.train()

    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(step, 2 * TRAIN_STEPS_PER_STAGE),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=256.0,
        growth_interval=2 * TRAIN_STEPS_PER_STAGE + 1,
        enabled=args.amp and device.type == "cuda",
    )

    stage_step = 0
    global_step = (stage - 1) * TRAIN_STEPS_PER_STAGE
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        expected = {
            "method": CHECKPOINT_METHOD,
            "stage": stage,
            "seed": args.seed,
            "data_manifest": data_manifest,
            "selection_sha256": selection_sha256,
            "retained_width": retained_width,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("MoPE resume checkpoint differs from the requested run")
        load_text_state_dict(student, checkpoint["text_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        stage_step = int(checkpoint["stage_step"])
        global_step = int(checkpoint["global_step"])
    elif stage == 2:
        checkpoint = torch.load(
            args.stage1_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        validate_transition_checkpoint(checkpoint, args, data_manifest, selection_sha256)
        load_text_state_dict(student, checkpoint["text_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.jsonl"
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
            "retained_width": retained_width,
        },
    )

    teacher_hidden = HiddenCapture(teacher)
    student_hidden = HiddenCapture(student)
    scale = teacher.logit_scale.exp().detach().clamp(max=100)
    running = {key: 0.0 for key in ("loss", "itc", "similarity", "feature", "hidden")}
    running_steps = 0
    started = time.time()
    try:
        for local_index in range(stage_step, TRAIN_STEPS_PER_STAGE):
            start = local_index * TRAIN_BATCH_SIZE
            batch_indices = indices[start : start + TRAIN_BATCH_SIZE].to(torch.int64)
            batch_tokens = tokens[batch_indices].to(device, non_blocking=device.type == "cuda")
            batch_images = image_features[batch_indices].to(
                device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            teacher_hidden.clear()
            student_hidden.clear()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                with torch.no_grad():
                    teacher_text = F.normalize(
                        teacher.encode_text(batch_tokens).float(), dim=-1
                    )
                student_text = F.normalize(student.encode_text(batch_tokens).float(), dim=-1)
                itc, similarity = cross_modal_losses(
                    student_text,
                    teacher_text,
                    batch_images,
                    scale,
                )
                feature = 0.5 * F.mse_loss(student_text, teacher_text)
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
            values = {
                "loss": loss,
                "itc": itc,
                "similarity": similarity,
                "feature": feature,
                "hidden": hidden,
            }
            for key, value in values.items():
                running[key] += float(value.detach())

            if stage_step % args.log_every == 0 or stage_step == TRAIN_STEPS_PER_STAGE:
                averages = {key: value / running_steps for key, value in running.items()}
                append_jsonl(
                    log_path,
                    {
                        "event": "train",
                        "stage": stage,
                        "stage_step": stage_step,
                        "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        **averages,
                        "elapsed_seconds": time.time() - started,
                    },
                )
                print(
                    f"stage={stage} step={stage_step}/{TRAIN_STEPS_PER_STAGE} "
                    f"loss={averages['loss']:.6f}",
                    flush=True,
                )
                running = {key: 0.0 for key in running}
                running_steps = 0

            if stage_step % args.save_every == 0:
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

    final_path = args.output_dir / f"stage{stage}.pt"
    keep_training_state = stage == 1
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
            "stage": stage,
            "stage_step": stage_step,
            "global_step": global_step,
            "checkpoint": str(final_path.resolve()),
            "elapsed_seconds": time.time() - started,
        },
    )


def main(stage: int) -> None:
    run(stage)

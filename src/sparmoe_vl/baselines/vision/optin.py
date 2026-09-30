"""Full-pool OPTIN adaptation for CLIP ViT-L/14 visual FFNs."""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import (
    DATA_SEED,
    D_FFN,
    D_MODEL,
    EMBED_DIM,
    EXPECTED_IMAGE_POOL_SHA256,
    EXPECTED_PROCESSING_ORDER_SHA256,
    EXPECTED_VISION_CAPTION_POOL_SHA256,
    N_LAYERS,
    N_TOKENS,
    POOL_SIZE,
    full_pool_permutation,
    make_image_text_loader,
    tensor_sha256,
    visual_blocks,
)


METHOD = "OPTIN-CLIP-FFN (Vision, full-pool adaptation)"
CHECKPOINT_METHOD = "OPTIN-CLIP Vision-FFN"
PAPER = "https://openreview.net/forum?id=MVmT6uQ3cQ"
OFFICIAL_REPOSITORY = "https://github.com/Skhaki18/optin-transformer-pruning"
OFFICIAL_COMMIT = "6c8d7caef2193a48c766f31e56008992dfc0c3bd"

MANIFOLD_SAMPLE_K = 768
TOTAL_CANDIDATES = N_LAYERS * D_FFN
DENSE_VISUAL_PARAMETERS = 303_966_208
DENSE_TOTAL_MACS_G = 81.012768768
DENSE_FFN_MACS_G = 51.740934144
NON_FFN_MACS_G = DENSE_TOTAL_MACS_G - DENSE_FFN_MACS_G


def full_calibration_manifest(seed: int, batch_size: int) -> tuple[Tensor, dict[str, Any]]:
    """Describe an OPTIN score pass over every visual-main sample."""

    if batch_size != 32:
        raise ValueError("the released visual OPTIN comparison fixes batch_size=32")
    if POOL_SIZE % batch_size:
        raise RuntimeError("the 500k visual pool must divide evenly into OPTIN batches")
    order = full_pool_permutation(seed)
    manifest = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "paired_caption_sha256": EXPECTED_VISION_CAPTION_POOL_SHA256,
        "processing_seed": int(seed),
        "processing_order_sha256": tensor_sha256(order),
        "batch_size": batch_size,
        "batches": POOL_SIZE // batch_size,
        "selection": "all visual-main samples; processing seed changes order only",
    }
    validate_data_manifest(manifest)
    return order, manifest


def validate_data_manifest(manifest: Mapping[str, Any]) -> None:
    seed = manifest.get("processing_seed")
    expected = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "paired_caption_sha256": EXPECTED_VISION_CAPTION_POOL_SHA256,
        "batch_size": 32,
        "batches": POOL_SIZE // 32,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if seed not in EXPECTED_PROCESSING_ORDER_SHA256:
        mismatches["processing_seed"] = (seed, tuple(EXPECTED_PROCESSING_ORDER_SHA256))
    elif manifest.get("processing_order_sha256") != EXPECTED_PROCESSING_ORDER_SHA256[seed]:
        mismatches["processing_order_sha256"] = (
            manifest.get("processing_order_sha256"),
            EXPECTED_PROCESSING_ORDER_SHA256[seed],
        )
    if mismatches:
        raise ValueError(f"OPTIN visual data differs from the main experiment: {mismatches}")


def candidate_partition(shard_index: int, num_shards: int) -> tuple[int, int]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid OPTIN candidate shard")
    start = TOTAL_CANDIDATES * shard_index // num_shards
    end = TOTAL_CANDIDATES * (shard_index + 1) // num_shards
    if start == end:
        raise ValueError("the number of shards exceeds the OPTIN candidate count")
    return start, end


def manifold_sampler_seed(
    run_seed: int,
    batch_index: int,
    layer: int,
    neuron: int,
    downstream: int,
) -> int:
    payload = (
        f"OPTIN-CLIP-VISION-FFN:{run_seed}:{batch_index}:{layer}:{neuron}:{downstream}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


class VisualLayerNormTrajectory:
    """Capture OPTIN's visual ``layernorm_before`` trajectory."""

    def __init__(self, model: nn.Module) -> None:
        self.outputs: list[Tensor | None] = [None] * N_LAYERS
        self.handles = [
            block.ln_1.register_forward_hook(self._hook(layer))
            for layer, block in enumerate(visual_blocks(model))
        ]

    def _hook(self, layer: int):
        def capture(_module, _inputs, output):
            if output.ndim != 3 or output.shape[-2:] != (N_TOKENS, D_MODEL):
                raise RuntimeError(f"unexpected visual trajectory: {tuple(output.shape)}")
            self.outputs[layer] = output.detach().contiguous()

        return capture

    def clear(self) -> None:
        self.outputs = [None] * N_LAYERS

    def take(self) -> list[Tensor]:
        if any(value is None for value in self.outputs):
            raise RuntimeError("an OPTIN visual trajectory hook did not run")
        result = [value for value in self.outputs if value is not None]
        self.clear()
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def neuron_mask_hook(block: nn.Module, neurons: Sequence[int], base_batch_size: int):
    neuron_list = [int(neuron) for neuron in neurons]

    def mask(_module, inputs):
        hidden = inputs[0]
        expected_batch = len(neuron_list) * base_batch_size
        if hidden.ndim != 3 or hidden.shape != (expected_batch, N_TOKENS, D_FFN):
            raise RuntimeError(f"unexpected visual FFN activation: {tuple(hidden.shape)}")
        masked = hidden.clone().view(len(neuron_list), base_batch_size, N_TOKENS, D_FFN)
        for candidate, neuron in enumerate(neuron_list):
            masked[candidate, :, :, neuron] = 0
        return (masked.view_as(hidden),)

    return block.mlp.c_proj.register_forward_pre_hook(mask)


@torch.inference_mode()
def encode_image_with_trajectory(
    model: nn.Module,
    capture: VisualLayerNormTrajectory,
    images: Tensor,
) -> tuple[Tensor, list[Tensor]]:
    capture.clear()
    features = model.encode_image(images, normalize=True)
    return features, capture.take()


@torch.inference_mode()
def official_manifold_cost_batched(
    teacher: Tensor,
    students: Tensor,
    samplers: Tensor,
) -> Tensor:
    """Compute official K=768 Gram-matrix loss for independent candidates."""

    teacher = F.normalize(teacher, dim=-1)
    students = F.normalize(students, dim=-1)
    candidates, batch, tokens, hidden = students.shape
    if batch * tokens < MANIFOLD_SAMPLE_K:
        raise RuntimeError("OPTIN K=768 exceeds the tokens available in one batch")
    if tuple(samplers.shape) != (candidates, MANIFOLD_SAMPLE_K):
        raise ValueError("OPTIN sampler tensor has the wrong shape")
    flat_teacher = teacher.reshape(batch * tokens, hidden)
    flat_students = students.reshape(candidates, batch * tokens, hidden)
    positions = samplers.to(students.device).unsqueeze(-1).expand(-1, -1, hidden)
    sampled_teacher = flat_teacher.unsqueeze(0).expand(candidates, -1, -1).gather(1, positions)
    sampled_students = flat_students.gather(1, positions)
    teacher_gram = torch.bmm(sampled_teacher, sampled_teacher.transpose(1, 2))
    student_gram = torch.bmm(sampled_students, sampled_students.transpose(1, 2))
    return (teacher_gram - student_gram).square().sum(dim=(1, 2))


@torch.inference_mode()
def clip_output_kl_batched(teacher_logits: Tensor, student_logits: Tensor) -> Tensor:
    """Apply OPTIN temperature-4 KL in both CLIP retrieval directions."""

    candidates = student_logits.shape[0]
    if student_logits.ndim != 3 or teacher_logits.shape != student_logits.shape[1:]:
        raise ValueError("OPTIN CLIP logits have incompatible shapes")

    def direction(teacher: Tensor, students: Tensor) -> Tensor:
        teacher_log = F.log_softmax(teacher, dim=-1).unsqueeze(0).expand(candidates, -1, -1)
        student_log = F.log_softmax(students, dim=-1)
        return torch.clamp(
            F.kl_div(
                student_log,
                teacher_log,
                reduction="none",
                log_target=True,
            ).sum(dim=(1, 2))
            * 16.0,
            min=0,
        )

    return 0.5 * (
        direction(teacher_logits, student_logits)
        + direction(teacher_logits.T, student_logits.transpose(1, 2))
    )


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _fresh_progress(
    manifest: dict[str, Any],
    candidate_start: int,
    candidate_end: int,
) -> dict[str, Any]:
    candidates = candidate_end - candidate_start
    return {
        "format_version": 1,
        "manifest": manifest,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "mmd_sums": torch.zeros(candidates, dtype=torch.float64),
        "kl_sums": torch.zeros(candidates, dtype=torch.float64),
        "elapsed_seconds": torch.zeros(candidates, dtype=torch.float64),
        "next_batch_index": 0,
        "next_flat_index": candidate_start,
        "candidate_evaluations": 0,
    }


def _load_progress(
    path: Path,
    manifest: dict[str, Any],
    candidate_start: int,
    candidate_end: int,
    resume: bool,
) -> dict[str, Any]:
    if not path.is_file():
        return _fresh_progress(manifest, candidate_start, candidate_end)
    if not resume:
        raise RuntimeError(f"progress already exists at {path}; pass --resume")
    progress = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "manifest": manifest,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
    }
    if not isinstance(progress, dict) or any(
        progress.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("saved OPTIN progress belongs to another shard or protocol")
    size = candidate_end - candidate_start
    for key in ("mmd_sums", "kl_sums", "elapsed_seconds"):
        value = progress.get(key)
        if not isinstance(value, Tensor) or tuple(value.shape) != (size,):
            raise RuntimeError(f"invalid OPTIN progress tensor: {key}")
    return progress


@torch.inference_mode()
def search_full_pool_score_shard(
    model: nn.Module,
    preprocess: Any,
    tokenizer: Any,
    paths: Sequence[Path],
    captions: Sequence[str],
    order: Tensor,
    manifest: dict[str, Any],
    candidate_start: int,
    candidate_end: int,
    device: str,
    output_dir: Path,
    workers: int = 8,
    candidate_batch: int = 1,
    resume: bool = False,
    save_every: int = 32,
    log_every: int = 256,
    max_candidate_evaluations: int | None = None,
) -> tuple[Tensor | None, Tensor | None, dict[str, Any]]:
    """Score a candidate shard against every one of the 500,000 images."""

    validate_data_manifest(manifest)
    if len(paths) != POOL_SIZE or len(captions) != POOL_SIZE:
        raise ValueError("OPTIN visual search requires all 500,000 image-text pairs")
    if (
        order.numel() != POOL_SIZE
        or tensor_sha256(order) != manifest["processing_order_sha256"]
    ):
        raise ValueError("OPTIN processing order differs from its manifest")
    if not 0 <= candidate_start < candidate_end <= TOTAL_CANDIDATES:
        raise ValueError("invalid OPTIN candidate range")
    if candidate_batch <= 0 or workers < 0 or save_every <= 0 or log_every <= 0:
        raise ValueError("OPTIN execution parameters must be positive")
    if max_candidate_evaluations is not None and max_candidate_evaluations <= 0:
        raise ValueError("diagnostic candidate limit must be positive")

    progress_path = Path(output_dir) / "optin_search_progress.pt"
    progress = _load_progress(
        progress_path,
        manifest,
        candidate_start,
        candidate_end,
        resume,
    )
    batch_start = int(progress["next_batch_index"])
    flat_start = int(progress["next_flat_index"])
    if not 0 <= batch_start <= int(manifest["batches"]):
        raise RuntimeError("invalid resumed OPTIN batch position")
    if batch_start < int(manifest["batches"]) and not (
        candidate_start <= flat_start < candidate_end
    ):
        raise RuntimeError("invalid resumed OPTIN candidate position")

    sample_offset = batch_start * int(manifest["batch_size"])
    loader = make_image_text_loader(
        paths,
        captions,
        order[sample_offset:],
        preprocess,
        tokenizer,
        int(manifest["batch_size"]),
        workers,
        device,
    )
    blocks = visual_blocks(model)
    capture = VisualLayerNormTrajectory(model)
    logit_scale = model.logit_scale.exp().detach()
    evaluated_this_call = 0
    invocation_start = time.time()
    stopped_early = False

    try:
        for offset, (images, tokens) in enumerate(loader):
            batch_index = batch_start + offset
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            if images.shape[0] != int(manifest["batch_size"]):
                raise RuntimeError("visual OPTIN requires complete 32-sample batches")
            text_features = model.encode_text(tokens, normalize=True)
            dense_image, dense_trajectory = encode_image_with_trajectory(model, capture, images)
            dense_logits = logit_scale * dense_image @ text_features.T
            flat = flat_start if batch_index == batch_start else candidate_start

            while flat < candidate_end:
                layer, neuron = divmod(flat, D_FFN)
                group_size = min(
                    candidate_batch,
                    D_FFN - neuron,
                    candidate_end - flat,
                )
                if max_candidate_evaluations is not None:
                    remaining = max_candidate_evaluations - evaluated_this_call
                    if remaining <= 0:
                        stopped_early = True
                        break
                    group_size = min(group_size, remaining)
                neurons = tuple(range(neuron, neuron + group_size))
                started = time.time()
                handle = neuron_mask_hook(blocks[layer], neurons, images.shape[0])
                try:
                    student_image, student_trajectory = encode_image_with_trajectory(
                        model,
                        capture,
                        images.repeat(group_size, 1, 1, 1),
                    )
                finally:
                    handle.remove()
                grouped_image = student_image.view(group_size, images.shape[0], EMBED_DIM)
                student_logits = logit_scale * grouped_image @ text_features.T

                downstream_layers = list(range(layer + 1, N_LAYERS))
                if layer == N_LAYERS - 1:
                    downstream_layers.append(N_LAYERS - 1)
                mmd = torch.zeros(group_size, device=device)
                total_tokens = images.shape[0] * N_TOKENS
                for downstream in downstream_layers:
                    samplers = torch.empty((group_size, MANIFOLD_SAMPLE_K), dtype=torch.long)
                    for candidate, current_neuron in enumerate(neurons):
                        generator = torch.Generator().manual_seed(
                            manifold_sampler_seed(
                                int(manifest["processing_seed"]),
                                batch_index,
                                layer,
                                current_neuron,
                                downstream,
                            )
                        )
                        samplers[candidate] = torch.randperm(total_tokens, generator=generator)[
                            :MANIFOLD_SAMPLE_K
                        ]
                    current = student_trajectory[downstream].view(
                        group_size,
                        images.shape[0],
                        N_TOKENS,
                        D_MODEL,
                    )
                    mmd += official_manifold_cost_batched(
                        dense_trajectory[downstream], current, samplers
                    )
                kl = clip_output_kl_batched(dense_logits, student_logits)
                local_start = flat - candidate_start
                local_end = local_start + group_size
                progress["mmd_sums"][local_start:local_end] += mmd.double().cpu()
                progress["kl_sums"][local_start:local_end] += kl.double().cpu()
                progress["elapsed_seconds"][local_start:local_end] += (
                    time.time() - started
                ) / group_size
                progress["candidate_evaluations"] += group_size
                evaluated_this_call += group_size
                flat += group_size
                next_batch = batch_index
                next_flat = flat
                if flat == candidate_end:
                    next_batch = batch_index + 1
                    next_flat = candidate_start
                progress["next_batch_index"] = next_batch
                progress["next_flat_index"] = next_flat

                if evaluated_this_call % save_every == 0:
                    _atomic_save(progress_path, progress)
                if evaluated_this_call % log_every == 0:
                    print(
                        f"batch={batch_index + 1}/{manifest['batches']} "
                        f"candidate={flat}/{candidate_end} "
                        f"shard=[{candidate_start},{candidate_end}) "
                        f"evaluations={progress['candidate_evaluations']}",
                        flush=True,
                    )
                if (
                    max_candidate_evaluations is not None
                    and evaluated_this_call >= max_candidate_evaluations
                ):
                    stopped_early = True
                    break

            _atomic_save(progress_path, progress)
            if stopped_early:
                break
            flat_start = candidate_start
    finally:
        capture.close()

    total_batches = int(manifest["batches"])
    complete = int(progress["next_batch_index"]) == total_batches
    report = {
        "complete": complete,
        "uses_complete_main_pool": complete,
        "selected_samples": POOL_SIZE if complete else 0,
        "completed_samples": min(
            POOL_SIZE,
            int(progress["next_batch_index"]) * int(manifest["batch_size"]),
        ),
        "total_batches": total_batches,
        "completed_batches": int(progress["next_batch_index"]),
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "next_flat_index": int(progress["next_flat_index"]),
        "candidate_evaluations": int(progress["candidate_evaluations"]),
        "candidate_evaluations_this_call": evaluated_this_call,
        "invocation_seconds": time.time() - invocation_start,
    }
    if not complete:
        return None, None, report
    raw_mmd = progress["mmd_sums"] / total_batches
    raw_kl = progress["kl_sums"] / total_batches
    if not torch.isfinite(raw_mmd).all() or not torch.isfinite(raw_kl).all():
        raise RuntimeError("complete OPTIN visual shard contains non-finite scores")
    return raw_mmd, raw_kl, report


def save_score_shard(
    path: Path,
    raw_mmd: Tensor,
    raw_kl: Tensor,
    manifest: dict[str, Any],
    candidate_start: int,
    candidate_end: int,
) -> None:
    validate_data_manifest(manifest)
    size = candidate_end - candidate_start
    if tuple(raw_mmd.shape) != (size,) or tuple(raw_kl.shape) != (size,):
        raise ValueError("OPTIN shard scores do not cover their declared range")
    _atomic_save(
        path,
        {
            "format_version": 1,
            "method": CHECKPOINT_METHOD,
            "stage": "score_shard",
            "manifest": manifest,
            "candidate_start": candidate_start,
            "candidate_end": candidate_end,
            "raw_mmd": raw_mmd,
            "raw_kl": raw_kl,
            "complete": True,
        },
    )


def merge_score_shards(paths: Sequence[Path]) -> tuple[Tensor, Tensor, dict[str, Any]]:
    if not paths:
        raise ValueError("no OPTIN visual score shards were supplied")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    payloads.sort(key=lambda item: int(item.get("candidate_start", -1)))
    manifest = payloads[0].get("manifest")
    validate_data_manifest(manifest)
    cursor = 0
    mmd_parts, kl_parts = [], []
    for payload in payloads:
        start = int(payload.get("candidate_start", -1))
        end = int(payload.get("candidate_end", -1))
        size = end - start
        if (
            payload.get("method") != CHECKPOINT_METHOD
            or payload.get("stage") != "score_shard"
            or payload.get("complete") is not True
            or payload.get("manifest") != manifest
            or start != cursor
            or not isinstance(payload.get("raw_mmd"), Tensor)
            or not isinstance(payload.get("raw_kl"), Tensor)
            or tuple(payload["raw_mmd"].shape) != (size,)
            or tuple(payload["raw_kl"].shape) != (size,)
        ):
            raise ValueError("OPTIN score shards are incomplete, overlapping, or mismatched")
        mmd_parts.append(payload["raw_mmd"])
        kl_parts.append(payload["raw_kl"])
        cursor = end
    if cursor != TOTAL_CANDIDATES:
        raise ValueError(f"OPTIN score coverage ends at {cursor}, expected {TOTAL_CANDIDATES}")
    raw_mmd = torch.cat(mmd_parts).reshape(N_LAYERS, D_FFN)
    raw_kl = torch.cat(kl_parts).reshape(N_LAYERS, D_FFN)
    if not torch.isfinite(raw_mmd).all() or not torch.isfinite(raw_kl).all():
        raise ValueError("merged OPTIN scores contain non-finite values")
    return raw_mmd, raw_kl, dict(manifest)


def combine_official_scores(raw_mmd: Tensor, raw_kl: Tensor) -> tuple[Tensor, Tensor]:
    expected_shape = (N_LAYERS, D_FFN)
    if tuple(raw_mmd.shape) != expected_shape or tuple(raw_kl.shape) != expected_shape:
        raise ValueError("OPTIN score tensors have the wrong shape")
    if not torch.isfinite(raw_mmd).all() or not torch.isfinite(raw_kl).all():
        raise RuntimeError("cannot combine incomplete OPTIN visual scores")
    combined = torch.empty_like(raw_mmd)
    kl_scalings = torch.ones_like(raw_kl)
    average_scaling: list[float] = []
    target_ratio = math.log10(100.0)
    for flat in range(TOTAL_CANDIDATES):
        layer, neuron = divmod(flat, D_FFN)
        mmd = float(raw_mmd[layer, neuron])
        kl = float(raw_kl[layer, neuron])
        scaling = 1.0
        if mmd > kl and mmd > 0 and kl > 0:
            try:
                ratio = math.log10(mmd) - math.log10(kl)
                scaling = 10.0 ** int(ratio - target_ratio)
                average_scaling.append(scaling)
            except (ValueError, OverflowError):
                scaling = float(np.mean(average_scaling)) if average_scaling else 1.0
        score = mmd + scaling * kl
        if layer == N_LAYERS - 1:
            score *= 0.0001
        combined[layer, neuron] = score
        kl_scalings[layer, neuron] = scaling
    return combined, kl_scalings


def select_channels(scores: Tensor, target_reduction: float) -> list[Tensor]:
    if tuple(scores.shape) != (N_LAYERS, D_FFN):
        raise ValueError("OPTIN scores must cover all visual FFN neurons")
    total = scores.numel()
    remove_count = round(total * target_reduction)
    keep_count = total - remove_count
    if not 0 < keep_count < total:
        raise ValueError("target reduction must keep some but not all visual neurons")
    keep_flat = torch.argsort(scores.reshape(-1), descending=True)[:keep_count]
    keep_mask = torch.zeros(total, dtype=torch.bool)
    keep_mask[keep_flat] = True
    keep_mask = keep_mask.reshape(N_LAYERS, D_FFN)
    return [torch.where(keep_mask[layer])[0] for layer in range(N_LAYERS)]


@torch.no_grad()
def structurally_prune_pair(
    c_fc: nn.Linear,
    c_proj: nn.Linear,
    kept: Tensor,
) -> tuple[nn.Linear, nn.Linear]:
    indices = kept.to(device=c_fc.weight.device, dtype=torch.long)
    if indices.unique().numel() != indices.numel():
        raise ValueError("retained OPTIN channel indices contain duplicates")
    width = int(indices.numel())
    new_fc = nn.Linear(
        c_fc.in_features,
        width,
        bias=c_fc.bias is not None,
        device=c_fc.weight.device,
        dtype=c_fc.weight.dtype,
    )
    new_proj = nn.Linear(
        width,
        c_proj.out_features,
        bias=c_proj.bias is not None,
        device=c_proj.weight.device,
        dtype=c_proj.weight.dtype,
    )
    new_fc.weight.copy_(c_fc.weight.index_select(0, indices))
    if c_fc.bias is not None:
        new_fc.bias.copy_(c_fc.bias.index_select(0, indices))
    new_proj.weight.copy_(c_proj.weight.index_select(1, indices))
    if c_proj.bias is not None:
        new_proj.bias.copy_(c_proj.bias)
    return new_fc, new_proj


@torch.no_grad()
def apply_structural_pruning(model: nn.Module, kept_indices: Sequence[Tensor]) -> None:
    if len(kept_indices) != N_LAYERS:
        raise ValueError(f"expected {N_LAYERS} retained-index tensors")
    for block, kept in zip(visual_blocks(model), kept_indices):
        block.mlp.c_fc, block.mlp.c_proj = structurally_prune_pair(
            block.mlp.c_fc,
            block.mlp.c_proj,
            kept,
        )


def ffn_statistics(hidden_sizes: Sequence[int]) -> dict[str, float | int]:
    if len(hidden_sizes) != N_LAYERS or any(not 0 <= size <= D_FFN for size in hidden_sizes):
        raise ValueError("invalid pruned visual FFN widths")
    retained_sum = int(sum(hidden_sizes))
    if retained_sum <= 0:
        raise ValueError("OPTIN cannot remove every visual FFN neuron")
    total = N_LAYERS * D_FFN
    removed = total - retained_sum
    retained = retained_sum / total
    reduction = 1.0 - retained
    ffn_macs = 2.0 * N_TOKENS * D_MODEL * retained_sum / 1e9
    total_macs = NON_FFN_MACS_G + ffn_macs
    active_params = DENSE_VISUAL_PARAMETERS - removed * (2 * D_MODEL + 1)
    return {
        "hidden_size_sum": retained_sum,
        "removed_channels": removed,
        "retained_fraction": retained,
        "ffn_reduction_fraction": reduction,
        "ffn_reduction_percent": 100.0 * reduction,
        "visual_ffn_macs_g": ffn_macs,
        "visual_total_macs_g": total_macs,
        "active_visual_parameters_m": active_params / 1e6,
    }


def save_final_checkpoint(
    model: nn.Module,
    raw_mmd: Tensor,
    raw_kl: Tensor,
    manifest: dict[str, Any],
    target_reduction: float,
    pretrained_sha256: str,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Materialize pruning only after full-pool score-shard validation."""

    validate_data_manifest(manifest)
    combined, kl_scalings = combine_official_scores(raw_mmd, raw_kl)
    kept_indices = select_channels(combined, target_reduction)
    hidden_sizes = [int(indices.numel()) for indices in kept_indices]
    statistics = ffn_statistics(hidden_sizes)
    apply_structural_pruning(model, kept_indices)

    score_path = Path(output_dir) / "optin_scores.pt"
    _atomic_save(
        score_path,
        {
            "raw_mmd": raw_mmd,
            "raw_kl": raw_kl,
            "kl_scalings": kl_scalings,
            "combined_importance": combined,
            "data_manifest": manifest,
        },
    )
    checkpoint_path = Path(output_dir) / "optin_vision_pruned.pt"
    checkpoint = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "seed": int(manifest["processing_seed"]),
        "data_seed": DATA_SEED,
        "pretrained_sha256": pretrained_sha256,
        "data_manifest": manifest,
        "target_ffn_reduction": target_reduction,
        "kept_indices": [indices.cpu() for indices in kept_indices],
        "hidden_sizes": hidden_sizes,
        "statistics": statistics,
        "visual_state_dict": {
            key: value.detach().cpu() for key, value in model.visual.state_dict().items()
        },
        "complete": True,
    }
    _atomic_save(checkpoint_path, checkpoint)
    validate_checkpoint(checkpoint)
    return checkpoint_path, score_path, checkpoint


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("method") != CHECKPOINT_METHOD or checkpoint.get("complete") is not True:
        raise ValueError("not a complete OPTIN visual checkpoint")
    manifest = checkpoint.get("data_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("OPTIN visual checkpoint has no data manifest")
    validate_data_manifest(manifest)
    kept = checkpoint.get("kept_indices")
    if not isinstance(kept, list) or len(kept) != N_LAYERS:
        raise ValueError("invalid retained-channel list in OPTIN visual checkpoint")
    if any(not isinstance(indices, Tensor) for indices in kept):
        raise ValueError("OPTIN retained-channel entries must be tensors")
    hidden_sizes = [int(indices.numel()) for indices in kept]
    if hidden_sizes != checkpoint.get("hidden_sizes"):
        raise ValueError("OPTIN visual hidden sizes disagree with retained indices")
    calculated = ffn_statistics(hidden_sizes)
    recorded = checkpoint.get("statistics")
    if not isinstance(recorded, Mapping):
        raise ValueError("OPTIN visual checkpoint has no statistics")
    for key, value in calculated.items():
        if key not in recorded or not math.isclose(
            float(recorded[key]), float(value), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"OPTIN visual checkpoint statistic mismatch: {key}")
    if not isinstance(checkpoint.get("visual_state_dict"), Mapping):
        raise ValueError("OPTIN visual checkpoint has no visual state dictionary")


def load_visual_state_dict(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    incompatible = model.visual.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "OPTIN visual checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

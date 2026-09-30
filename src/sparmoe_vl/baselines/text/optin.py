"""Full-pool OPTIN adaptation for CLIP ViT-L/14 text FFNs."""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import (
    DATA_SEED,
    DENSE_FFN_MACS_G,
    DENSE_TOTAL_MACS_G,
    DENSE_TRANSFORMER_PARAMETERS,
    D_FFN,
    D_MODEL,
    EXPECTED_PAIRED_PATH_POOL_SHA256,
    EXPECTED_TEXT_POOL_SHA256,
    N_LAYERS,
    N_TOKENS,
    POOL_SIZE,
    full_pool_permutation,
    text_blocks,
)


METHOD = "OPTIN-CLIP-FFN (Text, full-pool adaptation)"
CHECKPOINT_METHOD = "OPTIN-CLIP Text-FFN"
PAPER = "https://openreview.net/forum?id=MVmT6uQ3cQ"
OFFICIAL_REPOSITORY = "https://github.com/Skhaki18/optin-transformer-pruning"
OFFICIAL_COMMIT = "6c8d7caef2193a48c766f31e56008992dfc0c3bd"

OUTPUT_DIM = 768
MANIFOLD_SAMPLE_K = 768
TOTAL_CANDIDATES = N_LAYERS * D_FFN


def tensor_sha256(values: Tensor) -> str:
    array = values.detach().cpu().to(torch.int64).numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def full_calibration_manifest(seed: int, batch_size: int) -> tuple[Tensor, dict[str, Any]]:
    """Describe a run that consumes every main-pool example exactly once."""

    if batch_size != 32:
        raise ValueError("the released OPTIN comparison fixes batch_size=32")
    if POOL_SIZE % batch_size:
        raise RuntimeError("the full pool must divide evenly into OPTIN batches")
    indices = full_pool_permutation(seed)
    if indices.numel() != POOL_SIZE or torch.unique(indices).numel() != POOL_SIZE:
        raise RuntimeError("OPTIN processing order is not a complete pool permutation")
    manifest = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "uses_complete_pool": True,
        "text_pool_sha256": EXPECTED_TEXT_POOL_SHA256,
        "paired_path_pool_sha256": EXPECTED_PAIRED_PATH_POOL_SHA256,
        "processing_seed": int(seed),
        "processing_order_sha256": tensor_sha256(indices),
        "batch_size": batch_size,
        "batches": POOL_SIZE // batch_size,
        "selection": "all main-pool samples; processing seed changes order only",
    }
    validate_data_manifest(manifest)
    return indices, manifest


def validate_data_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "uses_complete_pool": True,
        "text_pool_sha256": EXPECTED_TEXT_POOL_SHA256,
        "paired_path_pool_sha256": EXPECTED_PAIRED_PATH_POOL_SHA256,
        "batch_size": 32,
        "batches": POOL_SIZE // 32,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"OPTIN data manifest differs from the main pool: {mismatches}")
    if not manifest.get("processing_order_sha256"):
        raise ValueError("OPTIN data manifest has no processing-order fingerprint")


def manifold_sampler_seed(
    run_seed: int,
    batch_index: int,
    layer: int,
    neuron: int,
    downstream: int,
) -> int:
    payload = (
        f"OPTIN-CLIP-TEXT-FFN:{run_seed}:{batch_index}:{layer}:{neuron}:{downstream}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


class TextFfnOutputTrajectory:
    """Capture OPTIN's language ``output.dense`` analogue: ``mlp.c_proj``."""

    def __init__(self, model: nn.Module) -> None:
        self.outputs: list[Tensor | None] = [None] * N_LAYERS
        self.handles = [
            block.mlp.c_proj.register_forward_hook(self._hook(layer))
            for layer, block in enumerate(text_blocks(model))
        ]

    def _hook(self, layer: int):
        def capture(_module, _inputs, output):
            if output.ndim != 3 or output.shape[-2:] != (N_TOKENS, D_MODEL):
                raise RuntimeError(f"unexpected text trajectory: {tuple(output.shape)}")
            self.outputs[layer] = output.detach().contiguous()

        return capture

    def clear(self) -> None:
        self.outputs = [None] * N_LAYERS

    def take(self) -> list[Tensor]:
        if any(value is None for value in self.outputs):
            raise RuntimeError("an OPTIN trajectory hook did not run")
        result = [value for value in self.outputs if value is not None]
        self.clear()
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def neuron_mask_hook(block: nn.Module, neuron: int):
    def mask(_module, inputs):
        hidden = inputs[0]
        if hidden.ndim != 3 or hidden.shape[-1] != D_FFN:
            raise RuntimeError(f"unexpected text FFN activation: {tuple(hidden.shape)}")
        masked = hidden.clone()
        masked[:, :, int(neuron)] = 0
        return (masked,)

    return block.mlp.c_proj.register_forward_pre_hook(mask)


@torch.inference_mode()
def encode_text_with_trajectory(
    model: nn.Module,
    capture: TextFfnOutputTrajectory,
    tokens: Tensor,
) -> tuple[Tensor, list[Tensor]]:
    capture.clear()
    features = model.encode_text(tokens, normalize=True)
    return features, capture.take()


@torch.inference_mode()
def official_manifold_cost(teacher: Tensor, student: Tensor, sampler: Tensor) -> Tensor:
    teacher = F.normalize(teacher, dim=-1)
    student = F.normalize(student, dim=-1)
    batch, tokens, hidden = student.shape
    positions = sampler.to(student.device)
    flat_teacher = teacher.reshape(batch * tokens, hidden)[positions]
    flat_student = student.reshape(batch * tokens, hidden)[positions]
    difference = flat_teacher.mm(flat_teacher.T) - flat_student.mm(flat_student.T)
    return difference.square().sum()


@torch.inference_mode()
def official_kl_cost(teacher_logits: Tensor, student_logits: Tensor) -> Tensor:
    value = (
        F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.log_softmax(teacher_logits, dim=1),
            reduction="sum",
            log_target=True,
        )
        * 16.0
    )
    return torch.clamp(value, min=0)


@torch.inference_mode()
def clip_output_kl(teacher_logits: Tensor, student_logits: Tensor) -> Tensor:
    return 0.5 * (
        official_kl_cost(teacher_logits, student_logits)
        + official_kl_cost(teacher_logits.T, student_logits.T)
    )


def _save_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _fresh_progress(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "manifest": manifest,
        "mmd_sums": torch.zeros((N_LAYERS, D_FFN), dtype=torch.float64),
        "kl_sums": torch.zeros((N_LAYERS, D_FFN), dtype=torch.float64),
        "elapsed_seconds": torch.zeros((N_LAYERS, D_FFN), dtype=torch.float64),
        "next_batch_index": 0,
        "next_flat_index": 0,
        "candidate_evaluations": 0,
    }


def _load_progress(
    path: Path,
    manifest: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if not path.exists():
        return _fresh_progress(manifest)
    if not resume:
        raise RuntimeError(f"progress already exists at {path}; pass --resume")
    progress = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(progress, dict) or progress.get("manifest") != manifest:
        raise RuntimeError("saved OPTIN progress belongs to a different data/configuration")
    expected_shape = (N_LAYERS, D_FFN)
    for key in ("mmd_sums", "kl_sums", "elapsed_seconds"):
        value = progress.get(key)
        if not isinstance(value, Tensor) or tuple(value.shape) != expected_shape:
            raise RuntimeError(f"invalid OPTIN progress tensor: {key}")
    return progress


@torch.inference_mode()
def search_full_pool_scores(
    model: nn.Module,
    tokens: Tensor,
    image_features: Tensor,
    order: Tensor,
    manifest: dict[str, Any],
    device: str,
    output_dir: Path,
    resume: bool = False,
    save_every: int = 64,
    log_every: int = 256,
    max_candidate_evaluations: int | None = None,
) -> tuple[Tensor | None, Tensor | None, dict[str, Any]]:
    """Accumulate OPTIN scores over all 15,625 batches of the 500k pool.

    A diagnostic invocation may stop early, but incomplete state can never be
    converted into a released pruning checkpoint.
    """

    validate_data_manifest(manifest)
    if tuple(tokens.shape) != (POOL_SIZE, N_TOKENS):
        raise ValueError("OPTIN tokens must contain the complete main text pool")
    if tuple(image_features.shape) != (POOL_SIZE, OUTPUT_DIM):
        raise ValueError("OPTIN image features must contain the complete paired pool")
    if (
        order.numel() != POOL_SIZE
        or tensor_sha256(order) != manifest["processing_order_sha256"]
    ):
        raise ValueError("OPTIN processing order differs from its manifest")
    if save_every <= 0 or log_every <= 0:
        raise ValueError("progress intervals must be positive")

    progress_path = Path(output_dir) / "optin_search_progress.pt"
    progress = _load_progress(progress_path, manifest, resume)
    batch_start = int(progress["next_batch_index"])
    flat_start = int(progress["next_flat_index"])
    evaluated_this_call = 0
    invocation_start = time.time()
    capture = TextFfnOutputTrajectory(model)
    blocks = text_blocks(model)
    logit_scale = model.logit_scale.exp().detach()
    total_batches = int(manifest["batches"])
    batch_size = int(manifest["batch_size"])
    stopped_early = False

    try:
        for batch_index in range(batch_start, total_batches):
            start = batch_index * batch_size
            selection = order[start : start + batch_size].to(torch.int64)
            token_batch = tokens[selection].to(device, non_blocking=True)
            image_batch = image_features[selection].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            image_batch = F.normalize(image_batch, dim=-1)
            dense_text, dense_trajectory = encode_text_with_trajectory(
                model, capture, token_batch
            )
            dense_logits = logit_scale * image_batch @ dense_text.T
            candidate_start = flat_start if batch_index == batch_start else 0

            for flat in range(candidate_start, TOTAL_CANDIDATES):
                layer, neuron = divmod(flat, D_FFN)
                started = time.time()
                handle = neuron_mask_hook(blocks[layer], neuron)
                try:
                    current_text, current_trajectory = encode_text_with_trajectory(
                        model, capture, token_batch
                    )
                finally:
                    handle.remove()
                current_logits = logit_scale * image_batch @ current_text.T

                downstream_layers = list(range(layer + 1, N_LAYERS))
                if layer == N_LAYERS - 1:
                    downstream_layers.append(N_LAYERS - 1)
                mmd = torch.zeros((), device=device, dtype=torch.float32)
                total_tokens = token_batch.shape[0] * N_TOKENS
                if total_tokens < MANIFOLD_SAMPLE_K:
                    raise RuntimeError("OPTIN K=768 exceeds tokens in one batch")
                for downstream in downstream_layers:
                    generator = torch.Generator().manual_seed(
                        manifold_sampler_seed(
                            int(manifest["processing_seed"]),
                            batch_index,
                            layer,
                            neuron,
                            downstream,
                        )
                    )
                    sampler = torch.randperm(total_tokens, generator=generator)[
                        :MANIFOLD_SAMPLE_K
                    ]
                    mmd += official_manifold_cost(
                        dense_trajectory[downstream],
                        current_trajectory[downstream],
                        sampler,
                    )
                kl = clip_output_kl(dense_logits, current_logits)
                progress["mmd_sums"][layer, neuron] += float(mmd)
                progress["kl_sums"][layer, neuron] += float(kl)
                progress["elapsed_seconds"][layer, neuron] += time.time() - started
                progress["candidate_evaluations"] += 1
                evaluated_this_call += 1

                next_flat = flat + 1
                next_batch = batch_index
                if next_flat == TOTAL_CANDIDATES:
                    next_flat = 0
                    next_batch = batch_index + 1
                progress["next_batch_index"] = next_batch
                progress["next_flat_index"] = next_flat
                if evaluated_this_call % save_every == 0 or next_batch == total_batches:
                    _save_progress(progress_path, progress)
                if evaluated_this_call % log_every == 0:
                    print(
                        f"batch={batch_index + 1}/{total_batches} "
                        f"candidate={flat + 1}/{TOTAL_CANDIDATES} "
                        f"evaluations={progress['candidate_evaluations']}",
                        flush=True,
                    )
                if (
                    max_candidate_evaluations is not None
                    and evaluated_this_call >= max_candidate_evaluations
                ):
                    _save_progress(progress_path, progress)
                    stopped_early = True
                    break
            if stopped_early:
                break
            flat_start = 0
    finally:
        capture.close()

    complete = int(progress["next_batch_index"]) == total_batches
    report = {
        "complete": complete,
        "uses_complete_pool": complete,
        "selected_samples": POOL_SIZE if complete else 0,
        "total_batches": total_batches,
        "completed_batches": int(progress["next_batch_index"]),
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
        raise RuntimeError("complete OPTIN scores contain non-finite values")
    return raw_mmd, raw_kl, report


def combine_official_scores(raw_mmd: Tensor, raw_kl: Tensor) -> tuple[Tensor, Tensor]:
    if tuple(raw_mmd.shape) != (N_LAYERS, D_FFN) or tuple(raw_kl.shape) != (
        N_LAYERS,
        D_FFN,
    ):
        raise ValueError("OPTIN score tensors have the wrong shape")
    if not torch.isfinite(raw_mmd).all() or not torch.isfinite(raw_kl).all():
        raise RuntimeError("cannot combine incomplete OPTIN scores")

    combined = torch.empty_like(raw_mmd)
    kl_scalings = torch.ones_like(raw_kl)
    average_scaling: list[float] = []
    target_ratio = math.log10(100.0)
    envelope = torch.tensor(
        np.geomspace(start=1, stop=100, num=D_FFN)[::-1].copy(),
        dtype=torch.float64,
    )
    for flat in range(TOTAL_CANDIDATES):
        layer, neuron = divmod(flat, D_FFN)
        mmd = float(raw_mmd[layer, neuron])
        kl = float(raw_kl[layer, neuron])
        scaling = float(np.mean(average_scaling)) if average_scaling else 1.0
        if mmd > kl and mmd > 0 and kl > 0:
            try:
                ratio = math.log10(mmd) - math.log10(kl)
                scaling = 10.0 ** int(ratio - target_ratio)
                average_scaling.append(scaling)
            except (ValueError, OverflowError):
                pass
        score = (mmd + scaling * kl) * float(envelope[neuron])
        if layer == N_LAYERS - 1:
            score *= 0.5
        combined[layer, neuron] = score
        kl_scalings[layer, neuron] = scaling
    return combined, kl_scalings


def select_channels(scores: Tensor, target_reduction: float) -> list[Tensor]:
    if tuple(scores.shape) != (N_LAYERS, D_FFN):
        raise ValueError("OPTIN scores must cover all text FFN neurons")
    total = scores.numel()
    remove_count = round(total * target_reduction)
    keep_count = total - remove_count
    if not 0 < keep_count < total:
        raise ValueError("target reduction must keep some but not all neurons")
    keep_flat = torch.argsort(scores.reshape(-1), descending=True)[:keep_count]
    keep_mask = torch.zeros(total, dtype=torch.bool)
    keep_mask[keep_flat] = True
    keep_mask = keep_mask.reshape(N_LAYERS, D_FFN)
    kept = [torch.where(keep_mask[layer])[0] for layer in range(N_LAYERS)]
    if any(indices.numel() == 0 for indices in kept):
        raise RuntimeError("global OPTIN allocation removed an entire text FFN")
    return kept


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
    for block, kept in zip(text_blocks(model), kept_indices):
        block.mlp.c_fc, block.mlp.c_proj = structurally_prune_pair(
            block.mlp.c_fc,
            block.mlp.c_proj,
            kept,
        )


def ffn_statistics(hidden_sizes: Sequence[int]) -> dict[str, float | int]:
    if len(hidden_sizes) != N_LAYERS or any(not 0 < size <= D_FFN for size in hidden_sizes):
        raise ValueError("invalid pruned text FFN widths")
    retained_sum = int(sum(hidden_sizes))
    total = N_LAYERS * D_FFN
    removed = total - retained_sum
    retained = retained_sum / total
    reduction = 1.0 - retained
    ffn_macs = 2.0 * N_TOKENS * D_MODEL * retained_sum / 1e9
    total_macs = DENSE_TOTAL_MACS_G - DENSE_FFN_MACS_G + ffn_macs
    active_params = DENSE_TRANSFORMER_PARAMETERS - removed * (2 * D_MODEL + 1)
    return {
        "hidden_size_sum": retained_sum,
        "removed_channels": removed,
        "retained_fraction": retained,
        "ffn_reduction_fraction": reduction,
        "ffn_reduction_percent": 100.0 * reduction,
        "text_ffn_macs_g": ffn_macs,
        "text_total_macs_g": total_macs,
        "active_text_parameters_m": active_params / 1e6,
    }


def text_state_dict(model: nn.Module) -> dict[str, Tensor]:
    prefixes = ("token_embedding.", "transformer.", "ln_final.")
    exact = {"positional_embedding", "text_projection"}
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes) or key in exact
    }


def load_text_state_dict(model: nn.Module, state_dict: dict[str, Tensor]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("visual.") and key != "logit_scale"
    ]
    if unexpected or missing:
        raise RuntimeError(
            f"OPTIN text checkpoint mismatch: unexpected={unexpected}, missing={missing}"
        )


def save_final_checkpoint(
    model: nn.Module,
    raw_mmd: Tensor,
    raw_kl: Tensor,
    manifest: dict[str, Any],
    target_reduction: float,
    pretrained_sha256: str,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Prune and save only after all 500k samples passed the score search."""

    validate_data_manifest(manifest)
    combined, kl_scalings = combine_official_scores(raw_mmd, raw_kl)
    kept_indices = select_channels(combined, target_reduction)
    hidden_sizes = [int(indices.numel()) for indices in kept_indices]
    statistics = ffn_statistics(hidden_sizes)
    apply_structural_pruning(model, kept_indices)

    score_path = Path(output_dir) / "optin_scores.pt"
    _save_progress(
        score_path,
        {
            "raw_mmd": raw_mmd,
            "raw_kl": raw_kl,
            "kl_scalings": kl_scalings,
            "combined_importance": combined,
            "data_manifest": manifest,
        },
    )
    checkpoint_path = Path(output_dir) / "optin_text_pruned.pt"
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
        "text_state_dict": text_state_dict(model),
        "complete": True,
    }
    _save_progress(checkpoint_path, checkpoint)
    validate_checkpoint(checkpoint)
    return checkpoint_path, score_path, checkpoint


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("method") != CHECKPOINT_METHOD or checkpoint.get("complete") is not True:
        raise ValueError("not a complete OPTIN text checkpoint")
    validate_data_manifest(checkpoint.get("data_manifest", {}))
    kept = checkpoint.get("kept_indices")
    if not isinstance(kept, list) or len(kept) != N_LAYERS:
        raise ValueError("invalid retained-channel list in OPTIN checkpoint")
    hidden_sizes = [int(indices.numel()) for indices in kept]
    if hidden_sizes != checkpoint.get("hidden_sizes"):
        raise ValueError("OPTIN hidden sizes disagree with retained indices")
    calculated = ffn_statistics(hidden_sizes)
    recorded = checkpoint.get("statistics", {})
    for key, value in calculated.items():
        if key not in recorded or not math.isclose(
            float(recorded[key]), float(value), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"OPTIN checkpoint statistic mismatch: {key}")

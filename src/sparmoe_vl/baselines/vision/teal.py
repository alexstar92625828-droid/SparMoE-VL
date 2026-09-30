"""TEAL activation sparsification for CLIP ViT-L/14 visual FFNs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .common import (
    DATA_SEED,
    D_FFN,
    D_MODEL,
    EXPECTED_IMAGE_POOL_SHA256,
    EXPECTED_PROCESSING_ORDER_SHA256,
    N_LAYERS,
    N_TOKENS,
    POOL_SIZE,
    build_main_image_pool,
    full_pool_permutation,
    make_image_loader,
    tensor_sha256,
    visual_blocks,
)


METHOD = "TEAL-CLIP-FFN (Vision, training-free controlled adaptation)"
OFFICIAL_REPOSITORY = "https://github.com/FasterDecoding/TEAL"
OFFICIAL_COMMIT = "fb7373c93ac3594817c9ee64d4e08b47430a1822"

DENSE_VISUAL_PARAMETERS = 303_966_208
FFN_WEIGHT_PARAMETERS = N_LAYERS * 2 * D_MODEL * D_FFN
FFN_BIAS_PARAMETERS = N_LAYERS * (D_FFN + D_MODEL)
NON_FFN_PARAMETERS = DENSE_VISUAL_PARAMETERS - FFN_WEIGHT_PARAMETERS - FFN_BIAS_PARAMETERS
DENSE_TOTAL_MACS_G = 81.012768768
DENSE_FFN_MACS_G = 51.740934144
NON_FFN_MACS_G = DENSE_TOTAL_MACS_G - DENSE_FFN_MACS_G


def full_calibration_pool(
    seed: int,
    annotations: Path,
    image_root: Path,
) -> tuple[tuple[Path, ...], Tensor, dict[str, Any]]:
    """Return all visual-main samples in a deterministic processing order."""

    paths, pool = build_main_image_pool(annotations, image_root)
    indices = full_pool_permutation(seed)
    manifest = {
        **pool,
        "selected_samples": POOL_SIZE,
        "processing_seed": int(seed),
        "processing_order_sha256": tensor_sha256(indices),
        "patch_tokens_per_image": N_TOKENS - 1,
        "calibration_patch_tokens": POOL_SIZE * (N_TOKENS - 1),
    }
    validate_data_manifest(manifest)
    return paths, indices, manifest


def validate_data_manifest(manifest: Mapping[str, Any]) -> None:
    seed = manifest.get("processing_seed")
    expected = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "unique_records": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "selected_samples": POOL_SIZE,
        "patch_tokens_per_image": N_TOKENS - 1,
        "calibration_patch_tokens": POOL_SIZE * (N_TOKENS - 1),
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
        raise ValueError(f"TEAL visual data differs from the main experiment: {mismatches}")


def calibration_levels(target: float, base_step: float) -> tuple[float, ...]:
    if not 0 < target < 1 or not 0 < base_step < 1:
        raise ValueError("TEAL target and step must lie in (0, 1)")
    maximum = min(0.95, math.ceil((2 * target + base_step) / base_step) * base_step)
    return tuple(
        round(index * base_step, 10) for index in range(int(round(maximum / base_step)) + 1)
    )


def histogram_thresholds_from_counts(
    counts: Tensor,
    maximum: float,
    levels: Sequence[float],
) -> dict[float, float]:
    if counts.ndim != 1 or counts.numel() <= 1 or torch.any(counts < 0):
        raise ValueError("TEAL histogram must be a non-negative vector")
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("TEAL histogram has no observations")
    if maximum <= 0:
        return {float(level): 0.0 for level in levels}
    cumulative = torch.cumsum(counts, dim=0)
    thresholds = {0.0: 0.0}
    for level in levels:
        level = float(level)
        if level <= 0:
            continue
        rank = max(1, min(total, math.ceil(level * total)))
        index = int(torch.searchsorted(cumulative, cumulative.new_tensor(rank)))
        thresholds[level] = maximum * (index + 1) / counts.numel()
    return thresholds


def threshold_for(table: Mapping[float, float], sparsity: float) -> float:
    key = min(table, key=lambda value: abs(value - sparsity))
    if abs(key - sparsity) > 1e-8:
        raise KeyError(f"TEAL threshold for sparsity {sparsity} is unavailable")
    return float(table[key])


def effective_sparsity(sparsities: Mapping[str, float]) -> float:
    return 0.5 * (float(sparsities["fc"]) + float(sparsities["proj"]))


def sparsify_patch_tokens(
    values: Tensor,
    threshold: float,
    sparsity: float,
) -> tuple[Tensor, int, int]:
    if values.ndim != 3 or values.shape[1] != N_TOKENS:
        raise ValueError(f"visual TEAL expects [batch, {N_TOKENS}, width]")
    patch = values[:, 1:]
    if sparsity <= 0:
        return values, patch.numel(), patch.numel()
    mask = patch.abs().gt(float(threshold))
    sparse = torch.cat((values[:, :1], patch * mask), dim=1)
    return sparse, int(mask.sum()), mask.numel()


@torch.inference_mode()
def visual_layer_activations(
    model: nn.Module,
    images: Tensor,
    layer_index: int,
) -> tuple[Tensor, Tensor, Tensor]:
    blocks = visual_blocks(model)
    if not 0 <= layer_index < len(blocks):
        raise IndexError(f"invalid visual layer: {layer_index}")
    visual = model.visual
    cast_dtype = visual.transformer.get_cast_dtype()
    hidden = visual.conv1(images.to(cast_dtype)).flatten(2).permute(0, 2, 1)
    class_token = visual.class_embedding.to(cast_dtype) + torch.zeros(
        hidden.shape[0],
        1,
        hidden.shape[-1],
        device=hidden.device,
        dtype=hidden.dtype,
    )
    hidden = torch.cat((class_token, hidden), dim=1)
    hidden = hidden + visual.positional_embedding.to(cast_dtype)
    hidden = visual.patch_dropout(hidden)
    hidden = visual.ln_pre(hidden)

    for index, block in enumerate(blocks):
        after_attention = hidden + block.ls_1(
            block.attention(q_x=block.ln_1(hidden), attn_mask=None)
        )
        normalized = block.ln_2(after_attention)
        ffn_hidden = block.mlp.gelu(block.mlp.c_fc(normalized))
        intermediate_norm = getattr(block.mlp, "ln", None)
        if intermediate_norm is not None:
            ffn_hidden = intermediate_norm(ffn_hidden)
        dense_output = block.mlp.c_proj(ffn_hidden)
        if index == layer_index:
            return normalized, ffn_hidden, dense_output
        hidden = after_attention + block.ls_2(dense_output)
    raise AssertionError("unreachable visual layer traversal")


@torch.inference_mode()
def candidate_output(
    block: nn.Module,
    normalized: Tensor,
    sparsities: Mapping[str, float],
    fc_thresholds: Mapping[float, float],
    proj_thresholds: Mapping[float, float],
) -> Tensor:
    sparse_input, _, _ = sparsify_patch_tokens(
        normalized,
        threshold_for(fc_thresholds, sparsities["fc"]),
        sparsities["fc"],
    )
    hidden = block.mlp.gelu(block.mlp.c_fc(sparse_input))
    intermediate_norm = getattr(block.mlp, "ln", None)
    if intermediate_norm is not None:
        hidden = intermediate_norm(hidden)
    sparse_hidden, _, _ = sparsify_patch_tokens(
        hidden,
        threshold_for(proj_thresholds, sparsities["proj"]),
        sparsities["proj"],
    )
    return block.mlp.c_proj(sparse_hidden)


@torch.inference_mode()
def calibrate_layer_streaming(
    model: nn.Module,
    paths: Sequence[Path],
    indices: Tensor,
    preprocess: Any,
    layer_index: int,
    device: str,
    batch_size: int,
    workers: int,
    target: float,
    base_step: float,
    histogram_bins: int,
) -> dict[str, Any]:
    """Calibrate one visual layer on every main-pool image with bounded memory."""

    if histogram_bins <= 1:
        raise ValueError("TEAL histogram bins must be greater than one")
    levels = calibration_levels(target, base_step)
    loader = make_image_loader(
        paths,
        indices,
        preprocess,
        batch_size,
        workers,
        device,
    )
    maxima = {"fc": 0.0, "proj": 0.0}
    for images in loader:
        normalized, ffn_hidden, _ = visual_layer_activations(
            model,
            images.to(device, non_blocking=True),
            layer_index,
        )
        maxima["fc"] = max(maxima["fc"], float(normalized[:, 1:].abs().max()))
        maxima["proj"] = max(maxima["proj"], float(ffn_hidden[:, 1:].abs().max()))

    histograms = {
        "fc": torch.zeros(histogram_bins, dtype=torch.float64),
        "proj": torch.zeros(histogram_bins, dtype=torch.float64),
    }
    zero_counts = {"fc": 0, "proj": 0}
    value_counts = {"fc": 0, "proj": 0}
    for images in loader:
        normalized, ffn_hidden, _ = visual_layer_activations(
            model,
            images.to(device, non_blocking=True),
            layer_index,
        )
        for name, values in (("fc", normalized[:, 1:]), ("proj", ffn_hidden[:, 1:])):
            absolute = values.detach().float().abs()
            if maxima[name] > 0:
                counts = torch.histc(
                    absolute,
                    bins=histogram_bins,
                    min=0.0,
                    max=maxima[name],
                )
                histograms[name].add_(counts.cpu().double())
            else:
                histograms[name][0] += absolute.numel()
            zero_counts[name] += int((absolute == 0).sum())
            value_counts[name] += absolute.numel()

    fc_thresholds = histogram_thresholds_from_counts(histograms["fc"], maxima["fc"], levels)
    proj_thresholds = histogram_thresholds_from_counts(
        histograms["proj"], maxima["proj"], levels
    )
    sparsities = {"fc": 0.0, "proj": 0.0}
    trace = [
        {
            "effective_sparsity": 0.0,
            "activation_error": 0.0,
            "fc": 0.0,
            "proj": 0.0,
        }
    ]
    maximum_level = max(levels)
    block = visual_blocks(model)[layer_index]
    while effective_sparsity(sparsities) < target:
        candidates = []
        for name in ("fc", "proj"):
            candidate = dict(sparsities)
            candidate[name] = round(candidate[name] + base_step, 10)
            if candidate[name] <= maximum_level + 1e-8 and candidate[name] < 1:
                candidates.append((name, candidate))
        if not candidates:
            raise RuntimeError(f"no TEAL greedy candidate at visual layer {layer_index}")

        error_sums = {name: 0.0 for name, _ in candidates}
        error_counts = {name: 0 for name, _ in candidates}
        for images in loader:
            normalized, _, dense_output = visual_layer_activations(
                model,
                images.to(device, non_blocking=True),
                layer_index,
            )
            for name, candidate in candidates:
                output = candidate_output(
                    block,
                    normalized,
                    candidate,
                    fc_thresholds,
                    proj_thresholds,
                )
                differences = torch.norm((dense_output - output)[:, 1:], dim=1)
                error_sums[name] += float(differences.sum())
                error_counts[name] += differences.numel()
        errors = {name: error_sums[name] / error_counts[name] for name, _ in candidates}
        selected_name, selected = min(candidates, key=lambda item: errors[item[0]])
        sparsities = selected
        trace.append(
            {
                "effective_sparsity": effective_sparsity(sparsities),
                "activation_error": errors[selected_name],
                "fc": sparsities["fc"],
                "proj": sparsities["proj"],
                "selected_branch": selected_name,
            }
        )

    selected_row = min(trace, key=lambda row: abs(row["effective_sparsity"] - target))
    return {
        "layer": layer_index,
        "effective_sparsity": selected_row["effective_sparsity"],
        "fc_sparsity": selected_row["fc"],
        "fc_threshold": threshold_for(fc_thresholds, selected_row["fc"]),
        "proj_sparsity": selected_row["proj"],
        "proj_threshold": threshold_for(proj_thresholds, selected_row["proj"]),
        "selected_activation_error": selected_row["activation_error"],
        "histograms": {
            "fc_input": {
                "bins": histogram_bins,
                "max_abs": maxima["fc"],
                "values": value_counts["fc"],
                "zero_fraction": zero_counts["fc"] / value_counts["fc"],
            },
            "proj_input": {
                "bins": histogram_bins,
                "max_abs": maxima["proj"],
                "values": value_counts["proj"],
                "zero_fraction": zero_counts["proj"] / value_counts["proj"],
            },
        },
        "greedy_trace": trace,
    }


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    expected = {
        "format_version": 1,
        "method": METHOD,
        "stage": "calibrated",
        "complete": True,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid or incomplete TEAL visual checkpoint: {mismatches}")
    manifest = checkpoint.get("calibration_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("TEAL visual checkpoint has no data manifest")
    validate_data_manifest(manifest)
    layers = checkpoint.get("layers")
    if not isinstance(layers, list) or len(layers) != N_LAYERS:
        raise ValueError(f"TEAL visual checkpoint must contain {N_LAYERS} layers")


class TEALVisualMLP(nn.Module):
    """Apply calibrated TEAL thresholds to patch tokens; keep CLS dense."""

    def __init__(self, mlp: nn.Module, schedule: Mapping[str, Any]) -> None:
        super().__init__()
        self.c_fc = mlp.c_fc
        self.gelu = mlp.gelu
        self.c_proj = mlp.c_proj
        self.ln = getattr(mlp, "ln", None)
        self.fc_sparsity = float(schedule["fc_sparsity"])
        self.fc_threshold = float(schedule["fc_threshold"])
        self.proj_sparsity = float(schedule["proj_sparsity"])
        self.proj_threshold = float(schedule["proj_threshold"])
        self.layer_index = int(schedule["layer"])
        self.reset_activity()

    def reset_activity(self) -> None:
        self.fc_kept = self.fc_total = 0
        self.proj_kept = self.proj_total = 0
        self.images_seen = 0

    def forward(self, values: Tensor) -> Tensor:
        values, kept, total = sparsify_patch_tokens(
            values,
            self.fc_threshold,
            self.fc_sparsity,
        )
        hidden = self.gelu(self.c_fc(values))
        if self.ln is not None:
            hidden = self.ln(hidden)
        hidden, proj_kept, proj_total = sparsify_patch_tokens(
            hidden,
            self.proj_threshold,
            self.proj_sparsity,
        )
        if not torch.is_grad_enabled():
            self.fc_kept += kept
            self.fc_total += total
            self.proj_kept += proj_kept
            self.proj_total += proj_total
            self.images_seen += int(values.shape[0])
        return self.c_proj(hidden)


def install_teal_visual(
    model: nn.Module,
    checkpoint: Mapping[str, Any],
) -> tuple[TEALVisualMLP, ...]:
    validate_checkpoint(checkpoint)
    wrappers = []
    for layer, (block, schedule) in enumerate(zip(visual_blocks(model), checkpoint["layers"])):
        if schedule.get("layer") != layer:
            raise ValueError("TEAL visual layer schedules are out of order")
        wrapper = TEALVisualMLP(block.mlp, schedule)
        block.mlp = wrapper
        wrappers.append(wrapper)
    return tuple(wrappers)


def reset_activity(wrappers: Iterable[TEALVisualMLP]) -> None:
    for wrapper in wrappers:
        wrapper.reset_activity()


@dataclass(frozen=True)
class VisualActivity:
    images: int
    fc_kept: int
    fc_total: int
    proj_kept: int
    proj_total: int
    ffn_macs_g: float
    total_macs_g: float
    reduction_percent: float
    active_parameters_m: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "images": self.images,
            "fc_kept": self.fc_kept,
            "fc_total": self.fc_total,
            "fc_sparsity": 1.0 - self.fc_kept / self.fc_total,
            "proj_kept": self.proj_kept,
            "proj_total": self.proj_total,
            "proj_sparsity": 1.0 - self.proj_kept / self.proj_total,
            "ffn_macs_vision_g": self.ffn_macs_g,
            "macs_vision_g": self.total_macs_g,
            "ffn_macs_reduction_percent": self.reduction_percent,
            "active_visual_parameters_m": self.active_parameters_m,
        }


def activity_statistics(wrappers: Iterable[TEALVisualMLP]) -> VisualActivity:
    wrappers = tuple(wrappers)
    image_counts = {wrapper.images_seen for wrapper in wrappers}
    if len(image_counts) != 1 or next(iter(image_counts), 0) <= 0:
        raise RuntimeError(f"inconsistent TEAL visual activity counters: {image_counts}")
    images = next(iter(image_counts))
    fc_kept = sum(item.fc_kept for item in wrappers)
    fc_total = sum(item.fc_total for item in wrappers)
    proj_kept = sum(item.proj_kept for item in wrappers)
    proj_total = sum(item.proj_total for item in wrappers)
    cls_macs = N_LAYERS * 2 * D_MODEL * D_FFN
    patch_macs = (fc_kept * D_FFN + proj_kept * D_MODEL) / images
    ffn_macs = cls_macs + patch_macs
    ffn_macs_g = ffn_macs / 1e9
    reduction = 1.0 - ffn_macs_g / DENSE_FFN_MACS_G
    active_ffn_weights = ffn_macs / N_TOKENS
    active_parameters = NON_FFN_PARAMETERS + FFN_BIAS_PARAMETERS + active_ffn_weights
    return VisualActivity(
        images=images,
        fc_kept=fc_kept,
        fc_total=fc_total,
        proj_kept=proj_kept,
        proj_total=proj_total,
        ffn_macs_g=ffn_macs_g,
        total_macs_g=NON_FFN_MACS_G + ffn_macs_g,
        reduction_percent=100.0 * reduction,
        active_parameters_m=active_parameters / 1e6,
    )

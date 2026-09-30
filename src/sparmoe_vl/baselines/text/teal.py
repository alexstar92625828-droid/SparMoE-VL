"""TEAL activation sparsification adapted to CLIP ViT-L/14 text FFNs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .common import (
    DATA_SEED,
    DENSE_FFN_MACS_G,
    DENSE_TOTAL_MACS_G,
    DENSE_TRANSFORMER_PARAMETERS,
    D_FFN,
    D_MODEL,
    EXPECTED_TEXT_POOL_SHA256,
    N_LAYERS,
    N_TOKENS,
    POOL_SIZE,
    full_pool_permutation,
    iter_token_batches,
    prepare_token_cache,
    text_blocks,
)


METHOD = "TEAL-CLIP-FFN (Text, training-free controlled adaptation)"
OFFICIAL_REPOSITORY = "https://github.com/FasterDecoding/TEAL"
OFFICIAL_COMMIT = "fb7373c93ac3594817c9ee64d4e08b47430a1822"

FFN_WEIGHT_PARAMETERS = N_LAYERS * 2 * D_MODEL * D_FFN
FFN_BIAS_PARAMETERS = N_LAYERS * (D_FFN + D_MODEL)
NON_FFN_PARAMETERS = DENSE_TRANSFORMER_PARAMETERS - FFN_WEIGHT_PARAMETERS - FFN_BIAS_PARAMETERS
NON_FFN_MACS_G = DENSE_TOTAL_MACS_G - DENSE_FFN_MACS_G


def full_calibration_pool(seed: int, cache_path, annotations):
    """Return every main-experiment token row, reordered but never subsampled."""

    cache = prepare_token_cache(cache_path, annotations)
    if cache["dataset_sha256"] != EXPECTED_TEXT_POOL_SHA256:
        raise RuntimeError("text pool fingerprint differs from the main experiment")
    indices = full_pool_permutation(seed)
    selection_sha256 = hashlib.sha256(
        indices.to(torch.int64).numpy().astype("<i8", copy=False).tobytes()
    ).hexdigest()
    manifest = {
        "pool_size": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "uses_complete_pool": True,
        "data_seed": DATA_SEED,
        "pool_sha256": cache["dataset_sha256"],
        "processing_seed": int(seed),
        "tokens_per_sample": N_TOKENS,
        "calibration_tokens": POOL_SIZE * N_TOKENS,
        "processing_order_sha256": selection_sha256,
    }
    return cache["tokens"], indices, manifest


def histogram_thresholds_from_counts(
    counts: Tensor,
    maximum: float,
    levels: Sequence[float],
) -> dict[float, float]:
    """Convert a complete-pool absolute-value histogram to TEAL thresholds."""

    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("counts must be a non-empty one-dimensional histogram")
    if torch.any(counts < 0):
        raise ValueError("histogram counts must be non-negative")
    total = int(counts.sum().item())
    if total <= 0:
        raise ValueError("histogram must contain observations")
    if maximum <= 0:
        return {float(level): 0.0 for level in levels}
    cumulative = torch.cumsum(counts, dim=0)
    thresholds = {0.0: 0.0}
    for level in levels:
        level = float(level)
        if level <= 0:
            continue
        rank = max(1, min(total, int(math.ceil(level * total))))
        value = cumulative.new_tensor(rank)
        index = int(torch.searchsorted(cumulative, value).item())
        thresholds[level] = maximum * (index + 1) / counts.numel()
    return thresholds


def threshold_for(table: Mapping[float, float], sparsity: float) -> float:
    key = min(table, key=lambda value: abs(value - sparsity))
    if abs(key - sparsity) > 1e-8:
        raise KeyError(f"threshold for sparsity {sparsity} is unavailable")
    return float(table[key])


def effective_sparsity(sparsities: Mapping[str, float]) -> float:
    return 0.5 * (float(sparsities["fc"]) + float(sparsities["proj"]))


def sparsify_tokens(
    values: Tensor,
    threshold: float,
    sparsity: float,
) -> tuple[Tensor, int, int]:
    if sparsity <= 0:
        return values, values.numel(), values.numel()
    mask = values.abs().gt(float(threshold))
    return values * mask, int(mask.sum().item()), mask.numel()


@torch.inference_mode()
def candidate_output(
    block: nn.Module,
    normalized_states: Tensor,
    sparsities: Mapping[str, float],
    fc_thresholds: Mapping[float, float],
    proj_thresholds: Mapping[float, float],
) -> Tensor:
    sparse_input, _, _ = sparsify_tokens(
        normalized_states,
        threshold_for(fc_thresholds, sparsities["fc"]),
        sparsities["fc"],
    )
    hidden = block.mlp.gelu(block.mlp.c_fc(sparse_input))
    sparse_hidden, _, _ = sparsify_tokens(
        hidden,
        threshold_for(proj_thresholds, sparsities["proj"]),
        sparsities["proj"],
    )
    return block.mlp.c_proj(sparse_hidden)


@torch.inference_mode()
def dense_layer_activations(
    model: nn.Module,
    token_batch: Tensor,
    layer_index: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Recompute dense states at one layer without caching the 500k pool."""

    blocks = text_blocks(model)
    if not 0 <= layer_index < len(blocks):
        raise IndexError(f"invalid text layer: {layer_index}")
    cast_dtype = model.transformer.get_cast_dtype()
    hidden = model.token_embedding(token_batch).to(cast_dtype)
    hidden = hidden + model.positional_embedding.to(cast_dtype)
    attention_mask = model.attn_mask
    if attention_mask is not None:
        attention_mask = attention_mask.to(token_batch.device)

    for index, block in enumerate(blocks):
        after_attention = hidden + block.ls_1(
            block.attention(
                q_x=block.ln_1(hidden),
                attn_mask=attention_mask,
            )
        )
        normalized = block.ln_2(after_attention)
        ffn_hidden = block.mlp.gelu(block.mlp.c_fc(normalized))
        dense_output = block.mlp.c_proj(ffn_hidden)
        if index == layer_index:
            return normalized, ffn_hidden, dense_output
        hidden = after_attention + block.ls_2(dense_output)
    raise AssertionError("unreachable layer traversal")


def _calibration_levels(target: float, base_step: float) -> tuple[float, ...]:
    if not 0 < target < 1:
        raise ValueError("target must lie in (0, 1)")
    if not 0 < base_step < 1:
        raise ValueError("base_step must lie in (0, 1)")
    maximum = min(
        0.95,
        math.ceil((2 * target + base_step) / base_step) * base_step,
    )
    return tuple(
        round(index * base_step, 10) for index in range(int(round(maximum / base_step)) + 1)
    )


@torch.inference_mode()
def calibrate_layer_streaming(
    model: nn.Module,
    tokens: Tensor,
    indices: Tensor,
    layer_index: int,
    device: str,
    batch_size: int,
    target: float,
    base_step: float,
    histogram_bins: int,
) -> dict[str, Any]:
    """Calibrate TEAL on all 500k samples with bounded accelerator memory.

    Dense activations are recomputed for each streaming pass. This is slower
    than holding a small calibration subset in memory, but it guarantees that
    the baseline consumes the exact main-experiment pool without subsampling.
    """

    if histogram_bins <= 1:
        raise ValueError("histogram_bins must be greater than one")
    levels = _calibration_levels(target, base_step)

    maxima = {"fc": 0.0, "proj": 0.0}
    for batch in iter_token_batches(tokens, indices, batch_size):
        normalized, ffn_hidden, _ = dense_layer_activations(
            model,
            batch.to(device, non_blocking=True),
            layer_index,
        )
        maxima["fc"] = max(maxima["fc"], float(normalized.abs().max()))
        maxima["proj"] = max(maxima["proj"], float(ffn_hidden.abs().max()))

    histograms = {
        "fc": torch.zeros(histogram_bins, dtype=torch.float64),
        "proj": torch.zeros(histogram_bins, dtype=torch.float64),
    }
    zero_counts = {"fc": 0, "proj": 0}
    value_counts = {"fc": 0, "proj": 0}
    for batch in iter_token_batches(tokens, indices, batch_size):
        normalized, ffn_hidden, _ = dense_layer_activations(
            model,
            batch.to(device, non_blocking=True),
            layer_index,
        )
        for name, values in (("fc", normalized), ("proj", ffn_hidden)):
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
    block = text_blocks(model)[layer_index]
    while effective_sparsity(sparsities) < target:
        candidates = []
        for name in ("fc", "proj"):
            candidate = dict(sparsities)
            candidate[name] = round(candidate[name] + base_step, 10)
            if candidate[name] <= maximum_level + 1e-8 and candidate[name] < 1:
                candidates.append((name, candidate))
        if not candidates:
            raise RuntimeError(f"no TEAL greedy candidate at layer {layer_index}")

        error_sums = {name: 0.0 for name, _ in candidates}
        error_counts = {name: 0 for name, _ in candidates}
        for batch in iter_token_batches(tokens, indices, batch_size):
            normalized, _, dense_output = dense_layer_activations(
                model,
                batch.to(device, non_blocking=True),
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
                differences = torch.norm(dense_output - output, dim=1)
                error_sums[name] += float(differences.sum())
                error_counts[name] += differences.numel()
        errors = {name: error_sums[name] / error_counts[name] for name, _ in candidates}
        selected_name, selected = min(
            candidates,
            key=lambda item: errors[item[0]],
        )
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

    selected_row = min(
        trace,
        key=lambda row: abs(row["effective_sparsity"] - target),
    )
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
    if checkpoint.get("method") != METHOD or not checkpoint.get("complete"):
        raise ValueError("invalid or incomplete TEAL text checkpoint")
    manifest = checkpoint.get("calibration_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("TEAL checkpoint has no calibration manifest")
    expected = {
        "pool_size": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "uses_complete_pool": True,
        "data_seed": DATA_SEED,
        "pool_sha256": EXPECTED_TEXT_POOL_SHA256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"TEAL calibration_manifest.{key}={manifest.get(key)!r}; expected {value!r}"
            )
    layers = checkpoint.get("layers")
    if not isinstance(layers, list) or len(layers) != N_LAYERS:
        raise ValueError(f"TEAL checkpoint must contain {N_LAYERS} layers")


class TEALTextMLP(nn.Module):
    """Apply calibrated magnitude thresholds at both text FFN projections."""

    def __init__(self, mlp: nn.Module, schedule: Mapping[str, Any]) -> None:
        super().__init__()
        self.c_fc = mlp.c_fc
        self.gelu = mlp.gelu
        self.c_proj = mlp.c_proj
        self.fc_sparsity = float(schedule["fc_sparsity"])
        self.fc_threshold = float(schedule["fc_threshold"])
        self.proj_sparsity = float(schedule["proj_sparsity"])
        self.proj_threshold = float(schedule["proj_threshold"])
        self.layer_index = int(schedule["layer"])
        self.reset_activity()

    def reset_activity(self) -> None:
        self.fc_kept = self.fc_total = 0
        self.proj_kept = self.proj_total = 0
        self.samples_seen = 0

    def forward(self, values: Tensor) -> Tensor:
        values, kept, total = sparsify_tokens(
            values,
            self.fc_threshold,
            self.fc_sparsity,
        )
        hidden = self.gelu(self.c_fc(values))
        hidden, proj_kept, proj_total = sparsify_tokens(
            hidden,
            self.proj_threshold,
            self.proj_sparsity,
        )
        if not torch.is_grad_enabled():
            self.fc_kept += kept
            self.fc_total += total
            self.proj_kept += proj_kept
            self.proj_total += proj_total
            self.samples_seen += int(values.shape[0])
        return self.c_proj(hidden)


def install_teal_text(
    model: nn.Module,
    checkpoint: Mapping[str, Any],
) -> tuple[TEALTextMLP, ...]:
    validate_checkpoint(checkpoint)
    wrappers = []
    for index, (block, schedule) in enumerate(zip(text_blocks(model), checkpoint["layers"])):
        if schedule.get("layer") != index:
            raise ValueError("TEAL text layer schedules are out of order")
        wrapper = TEALTextMLP(block.mlp, schedule)
        block.mlp = wrapper
        wrappers.append(wrapper)
    return tuple(wrappers)


def reset_activity(wrappers: Iterable[TEALTextMLP]) -> None:
    for wrapper in wrappers:
        wrapper.reset_activity()


@dataclass(frozen=True)
class TextActivity:
    samples: int
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
            "samples": self.samples,
            "fc_kept": self.fc_kept,
            "fc_total": self.fc_total,
            "fc_sparsity": 1.0 - self.fc_kept / self.fc_total,
            "proj_kept": self.proj_kept,
            "proj_total": self.proj_total,
            "proj_sparsity": 1.0 - self.proj_kept / self.proj_total,
            "ffn_macs_text_g": self.ffn_macs_g,
            "macs_text_g": self.total_macs_g,
            "ffn_macs_reduction_percent": self.reduction_percent,
            "active_text_parameters_m": self.active_parameters_m,
        }


def activity_statistics(wrappers: Iterable[TEALTextMLP]) -> TextActivity:
    wrappers = tuple(wrappers)
    sample_counts = {wrapper.samples_seen for wrapper in wrappers}
    if len(sample_counts) != 1 or next(iter(sample_counts), 0) <= 0:
        raise RuntimeError(f"inconsistent text activity counters: {sample_counts}")
    samples = next(iter(sample_counts))
    fc_kept = sum(item.fc_kept for item in wrappers)
    fc_total = sum(item.fc_total for item in wrappers)
    proj_kept = sum(item.proj_kept for item in wrappers)
    proj_total = sum(item.proj_total for item in wrappers)
    ffn_macs_per_sample = (fc_kept * D_FFN + proj_kept * D_MODEL) / samples
    ffn_macs_g = ffn_macs_per_sample / 1e9
    reduction = 1.0 - ffn_macs_g / DENSE_FFN_MACS_G
    active_weights = ffn_macs_per_sample / N_TOKENS
    active_parameters = NON_FFN_PARAMETERS + FFN_BIAS_PARAMETERS + active_weights
    return TextActivity(
        samples=samples,
        fc_kept=fc_kept,
        fc_total=fc_total,
        proj_kept=proj_kept,
        proj_total=proj_total,
        ffn_macs_g=ffn_macs_g,
        total_macs_g=NON_FFN_MACS_G + ffn_macs_g,
        reduction_percent=100.0 * reduction,
        active_parameters_m=active_parameters / 1e6,
    )

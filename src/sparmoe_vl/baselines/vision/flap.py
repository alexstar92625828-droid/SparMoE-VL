"""Full-pool FLAP WIFV pruning for CLIP ViT-L/14 visual FFNs."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    atomic_torch_save,
    build_main_image_pool,
    full_pool_permutation,
    make_image_slice_loader,
    tensor_sha256,
    visual_blocks,
)


METHOD = "FLAP-CLIP-FFN (Vision, full-pool adaptation)"
CHECKPOINT_METHOD = "FLAP-CLIP Vision-FFN"
PAPER = "https://arxiv.org/abs/2312.11983"
OFFICIAL_REPOSITORY = "https://github.com/CASIA-LMC-Lab/FLAP"
OFFICIAL_COMMIT = "3bb57db3449dd2fa04a5c2192de80e87e33be2b1"

CALIBRATION_BATCH_SIZE = 128
TARGET_FFN_REDUCTION = 0.3563
DENSE_VISUAL_PARAMETERS = 303_966_208
DENSE_TOTAL_MACS_G = 81.012768768
DENSE_FFN_MACS_G = 51.740934144
NON_FFN_MACS_G = DENSE_TOTAL_MACS_G - DENSE_FFN_MACS_G


def full_calibration_manifest(seed: int, batch_size: int) -> tuple[Tensor, dict[str, Any]]:
    """Describe one complete pass over the exact 500k visual-main pool."""

    if batch_size != CALIBRATION_BATCH_SIZE:
        raise ValueError(f"the released visual FLAP batch_size is {CALIBRATION_BATCH_SIZE}")
    order = full_pool_permutation(seed)
    batches = math.ceil(POOL_SIZE / batch_size)
    final_batch_size = POOL_SIZE - (batches - 1) * batch_size
    manifest = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "unique_samples": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "calibration_exposures": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "processing_seed": int(seed),
        "processing_order_sha256": tensor_sha256(order),
        "batch_size": batch_size,
        "batches": batches,
        "final_batch_size": final_batch_size,
        "image_tokens_per_layer": POOL_SIZE * N_TOKENS,
        "selection": "every visual-main image exactly once in first-epoch order",
    }
    validate_data_manifest(manifest)
    return order, manifest


def full_calibration_pool(
    seed: int,
    annotations: Path,
    image_root: Path,
    batch_size: int = CALIBRATION_BATCH_SIZE,
) -> tuple[tuple[Path, ...], Tensor, dict[str, Any]]:
    paths, pool = build_main_image_pool(annotations, image_root)
    order, manifest = full_calibration_manifest(seed, batch_size)
    for key in (
        "data_seed",
        "pool_size",
        "uses_complete_main_pool",
        "dataset_sha256",
    ):
        if pool[key] != manifest[key]:
            raise RuntimeError(f"FLAP visual pool metadata mismatch: {key}")
    return paths, order, manifest


def validate_data_manifest(manifest: Mapping[str, Any]) -> None:
    seed = manifest.get("processing_seed")
    expected = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "unique_samples": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "calibration_exposures": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "batch_size": CALIBRATION_BATCH_SIZE,
        "batches": math.ceil(POOL_SIZE / CALIBRATION_BATCH_SIZE),
        "final_batch_size": POOL_SIZE % CALIBRATION_BATCH_SIZE,
        "image_tokens_per_layer": POOL_SIZE * N_TOKENS,
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
        raise ValueError(f"FLAP visual data differs from the main experiment: {mismatches}")


class RunningFeatureMoments:
    """Numerically stable streaming moments for one FFN projection input."""

    def __init__(self, device: torch.device, width: int = D_FFN) -> None:
        if width <= 0:
            raise ValueError("FLAP feature width must be positive")
        self.width = int(width)
        self.count = 0
        self.mean = torch.zeros(self.width, device=device, dtype=torch.float32)
        self.m2 = torch.zeros(self.width, device=device, dtype=torch.float32)

    @torch.no_grad()
    def add(self, activation: Tensor) -> None:
        if activation.shape[-1] != self.width:
            raise ValueError("FLAP activation width differs from its accumulator")
        values = activation.detach().reshape(-1, self.width).float()
        batch_count = values.shape[0]
        batch_variance, batch_mean = torch.var_mean(values, dim=0, correction=0)
        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_variance * batch_count)
            self.count = batch_count
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2.add_(batch_variance * batch_count)
        self.m2.add_(delta.square() * (self.count * batch_count / total))
        self.mean.add_(delta * (batch_count / total))
        self.count = total

    def state_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean.detach().cpu(),
            "m2": self.m2.detach().cpu(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        mean, m2 = state.get("mean"), state.get("m2")
        if (
            not isinstance(mean, Tensor)
            or tuple(mean.shape) != (self.width,)
            or not isinstance(m2, Tensor)
            or tuple(m2.shape) != (self.width,)
            or int(state.get("count", -1)) < 0
        ):
            raise RuntimeError("invalid FLAP visual streaming-moment checkpoint")
        self.count = int(state["count"])
        self.mean.copy_(mean.to(self.mean))
        self.m2.copy_(m2.to(self.m2))

    def finish(self) -> tuple[Tensor, Tensor]:
        if self.count <= 1:
            raise RuntimeError("insufficient image-token observations for FLAP")
        return (
            self.mean.detach().cpu(),
            (self.m2 / (self.count - 1)).clamp_min(0).detach().cpu(),
        )


def _progress_payload(
    manifest: dict[str, Any],
    next_batch_index: int,
    images_seen: int,
    moments: Sequence[RunningFeatureMoments],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "manifest": manifest,
        "next_batch_index": next_batch_index,
        "images_seen": images_seen,
        "moments": [moment.state_dict() for moment in moments],
    }


def _load_progress(
    path: Path,
    manifest: dict[str, Any],
    moments: Sequence[RunningFeatureMoments],
    resume: bool,
) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    if not resume:
        raise RuntimeError(f"FLAP progress already exists at {path}; pass --resume")
    progress = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(progress, dict) or progress.get("manifest") != manifest:
        raise RuntimeError("saved FLAP progress belongs to another data protocol")
    states = progress.get("moments")
    if not isinstance(states, list) or len(states) != N_LAYERS:
        raise RuntimeError("saved FLAP progress has incomplete visual moments")
    for moment, state in zip(moments, states):
        moment.load_state_dict(state)
    next_batch = int(progress.get("next_batch_index", -1))
    images_seen = int(progress.get("images_seen", -1))
    expected_seen = min(POOL_SIZE, next_batch * CALIBRATION_BATCH_SIZE)
    if not 0 <= next_batch <= manifest["batches"] or images_seen != expected_seen:
        raise RuntimeError("saved FLAP progress has an invalid sample position")
    expected_tokens = images_seen * N_TOKENS
    if any(moment.count != expected_tokens for moment in moments):
        raise RuntimeError("saved FLAP progress has inconsistent image-token counts")
    return next_batch, images_seen


@torch.inference_mode()
def collect_wifv(
    model: nn.Module,
    preprocess: Any,
    paths: Sequence[Path],
    order: Tensor,
    manifest: dict[str, Any],
    device: str,
    output_dir: Path,
    workers: int = 8,
    log_every: int = 20,
    save_every: int = 100,
    resume: bool = False,
    max_batches: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Collect WIFV statistics from all 500,000 main-pool images."""

    validate_data_manifest(manifest)
    if len(paths) != POOL_SIZE:
        raise ValueError("FLAP requires the complete 500k visual path pool")
    if (
        order.numel() != POOL_SIZE
        or tensor_sha256(order) != manifest["processing_order_sha256"]
    ):
        raise ValueError("FLAP visual processing order differs from its manifest")
    if workers < 0 or log_every <= 0 or save_every <= 0:
        raise ValueError("FLAP workers/intervals must be non-negative and positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("FLAP diagnostic batch limit must be positive")

    torch_device = torch.device(device)
    moments = [RunningFeatureMoments(torch_device) for _ in range(N_LAYERS)]
    progress_path = Path(output_dir) / "flap_calibration_progress.pt"
    batch_start, images_seen = _load_progress(progress_path, manifest, moments, resume)
    sample_offset = min(POOL_SIZE, batch_start * CALIBRATION_BATCH_SIZE)
    loader = make_image_slice_loader(
        paths,
        order[sample_offset:],
        preprocess,
        CALIBRATION_BATCH_SIZE,
        workers,
        device,
    )
    handles = []
    for layer, block in enumerate(visual_blocks(model)):

        def hook(_module, inputs, layer_index=layer):
            moments[layer_index].add(inputs[0])

        handles.append(block.mlp.c_proj.register_forward_pre_hook(hook))

    started = time.time()
    processed_this_call = 0
    next_batch = batch_start
    try:
        for offset, images in enumerate(loader):
            batch_index = batch_start + offset
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=torch_device.type == "cuda",
            ):
                model.encode_image(images)
            images_seen += int(images.shape[0])
            processed_this_call += 1
            next_batch = batch_index + 1
            if processed_this_call % save_every == 0:
                atomic_torch_save(
                    _progress_payload(manifest, next_batch, images_seen, moments),
                    progress_path,
                )
            if processed_this_call % log_every == 0 or next_batch == manifest["batches"]:
                print(
                    f"batch={next_batch}/{manifest['batches']} "
                    f"images={images_seen}/{POOL_SIZE}",
                    flush=True,
                )
            if max_batches is not None and processed_this_call >= max_batches:
                break
    finally:
        for handle in handles:
            handle.remove()
    atomic_torch_save(
        _progress_payload(manifest, next_batch, images_seen, moments),
        progress_path,
    )

    complete = next_batch == manifest["batches"] and images_seen == POOL_SIZE
    report = {
        "complete": complete,
        "uses_complete_main_pool": complete,
        "selected_samples": POOL_SIZE if complete else 0,
        "completed_samples": images_seen,
        "total_batches": manifest["batches"],
        "completed_batches": next_batch,
        "batches_this_call": processed_this_call,
        "elapsed_seconds_this_call": time.time() - started,
    }
    if not complete:
        return None, report

    means, variances, raw_wifv, token_counts = [], [], [], []
    for layer, (block, moment) in enumerate(zip(visual_blocks(model), moments)):
        if moment.count != manifest["image_tokens_per_layer"]:
            raise RuntimeError(f"FLAP visual layer {layer} did not observe every image token")
        mean, variance = moment.finish()
        weight_norm_sq = block.mlp.c_proj.weight.detach().float().cpu().square().sum(0)
        score = variance * weight_norm_sq
        if not torch.isfinite(score).all() or float(score.std()) == 0:
            raise FloatingPointError(f"invalid FLAP WIFV scores in visual layer {layer}")
        means.append(mean)
        variances.append(variance)
        raw_wifv.append(score)
        token_counts.append(moment.count)
    return {
        "means": torch.stack(means),
        "variances": torch.stack(variances),
        "raw_wifv": torch.stack(raw_wifv),
        "images_seen": images_seen,
        "tokens_per_layer": token_counts,
        "formula": "Var(c_proj_input_channel) * sum(c_proj_weight_column ** 2)",
    }, report


def select_channels(
    raw_wifv: Tensor,
    target_reduction: float,
) -> tuple[list[Tensor], Tensor, int]:
    if tuple(raw_wifv.shape) != (N_LAYERS, D_FFN):
        raise ValueError("FLAP WIFV tensor has the wrong shape")
    if not torch.isfinite(raw_wifv).all():
        raise ValueError("FLAP WIFV tensor contains non-finite values")
    deviations = raw_wifv.std(dim=1, keepdim=True, correction=1)
    if torch.any(deviations == 0):
        raise ValueError("FLAP cannot standardize a constant layer")
    standardized = (raw_wifv - raw_wifv.mean(dim=1, keepdim=True)) / deviations
    total = standardized.numel()
    remove_count = round(total * target_reduction)
    if not 0 < remove_count < total:
        raise ValueError("FLAP target must remove some but not all visual channels")
    remove_flat = torch.argsort(standardized.reshape(-1))[:remove_count]
    keep_mask = torch.ones(total, dtype=torch.bool)
    keep_mask[remove_flat] = False
    keep_mask = keep_mask.reshape(N_LAYERS, D_FFN)
    kept = [torch.where(keep_mask[layer])[0] for layer in range(N_LAYERS)]
    if any(indices.numel() == 0 for indices in kept):
        raise RuntimeError("FLAP global allocation removed an entire visual FFN")
    return kept, standardized, remove_count


@torch.no_grad()
def prune_linear_pair_with_compensation(
    c_fc: nn.Linear,
    c_proj: nn.Linear,
    kept_indices: Tensor,
    mean_input: Tensor,
) -> tuple[nn.Linear, nn.Linear, Tensor]:
    if c_fc.out_features != c_proj.in_features or mean_input.numel() != c_fc.out_features:
        raise ValueError("FLAP projection pair and mean vector are incompatible")
    device, dtype = c_fc.weight.device, c_fc.weight.dtype
    kept = kept_indices.to(device=device, dtype=torch.long)
    if kept.numel() == 0 or kept.unique().numel() != kept.numel():
        raise ValueError("FLAP retained channels must be non-empty and unique")
    if int(kept.min()) < 0 or int(kept.max()) >= c_fc.out_features:
        raise ValueError("FLAP retained channel is out of range")
    removed_mask = torch.ones(c_fc.out_features, device=device, dtype=torch.bool)
    removed_mask[kept] = False
    removed = torch.where(removed_mask)[0]
    compensation = (
        c_proj.weight[:, removed]
        .float()
        .matmul(mean_input.to(device=device, dtype=torch.float32)[removed])
    )
    original_bias = (
        c_proj.bias.detach().float()
        if c_proj.bias is not None
        else torch.zeros(c_proj.out_features, device=device, dtype=torch.float32)
    )
    new_fc = nn.Linear(
        c_fc.in_features,
        int(kept.numel()),
        bias=c_fc.bias is not None,
        device=device,
        dtype=dtype,
    )
    new_proj = nn.Linear(
        int(kept.numel()),
        c_proj.out_features,
        bias=True,
        device=device,
        dtype=dtype,
    )
    new_fc.weight.copy_(c_fc.weight.index_select(0, kept))
    if c_fc.bias is not None:
        new_fc.bias.copy_(c_fc.bias.index_select(0, kept))
    new_proj.weight.copy_(c_proj.weight.index_select(1, kept))
    new_proj.bias.copy_((original_bias + compensation).to(dtype))
    return new_fc, new_proj, compensation.detach().cpu()


@torch.no_grad()
def apply_pruning(
    model: nn.Module,
    means: Tensor,
    kept_indices: Sequence[Tensor],
) -> list[Tensor]:
    if tuple(means.shape) != (N_LAYERS, D_FFN) or len(kept_indices) != N_LAYERS:
        raise ValueError("FLAP pruning requires 24 means and retained-index tensors")
    compensations = []
    for layer, (block, kept) in enumerate(zip(visual_blocks(model), kept_indices)):
        block.mlp.c_fc, block.mlp.c_proj, compensation = prune_linear_pair_with_compensation(
            block.mlp.c_fc,
            block.mlp.c_proj,
            kept,
            means[layer],
        )
        compensations.append(compensation)
    return compensations


@torch.no_grad()
def apply_checkpoint_structure(model: nn.Module, kept_indices: Sequence[Tensor]) -> None:
    zero_means = torch.zeros((N_LAYERS, D_FFN))
    apply_pruning(model, zero_means, kept_indices)


def ffn_statistics(hidden_sizes: Sequence[int]) -> dict[str, float | int]:
    if len(hidden_sizes) != N_LAYERS or any(not 0 < size <= D_FFN for size in hidden_sizes):
        raise ValueError("invalid pruned visual FFN widths")
    retained_sum = int(sum(hidden_sizes))
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


def validate_moment_pack(moment_pack: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    expected_shape = (N_LAYERS, D_FFN)
    for key in ("means", "variances", "raw_wifv"):
        value = moment_pack.get(key)
        if not isinstance(value, Tensor) or tuple(value.shape) != expected_shape:
            raise ValueError(f"incomplete FLAP visual statistic: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite FLAP visual statistic: {key}")
    if torch.any(moment_pack["variances"] < 0):
        raise ValueError("FLAP visual variances must be non-negative")
    if moment_pack.get("images_seen") != POOL_SIZE:
        raise ValueError("FLAP statistics do not cover all 500,000 images")
    if moment_pack.get("tokens_per_layer") != [POOL_SIZE * N_TOKENS] * N_LAYERS:
        raise ValueError("FLAP statistics do not cover every image token")
    validate_data_manifest(manifest)


def save_final_checkpoint(
    model: nn.Module,
    moment_pack: dict[str, Any],
    manifest: dict[str, Any],
    pretrained_sha256: str,
    output_dir: Path,
    target_reduction: float = TARGET_FFN_REDUCTION,
) -> tuple[Path, Path, dict[str, Any]]:
    validate_moment_pack(moment_pack, manifest)
    kept, standardized, remove_count = select_channels(
        moment_pack["raw_wifv"], target_reduction
    )
    compensations = apply_pruning(model, moment_pack["means"], kept)
    hidden_sizes = [int(indices.numel()) for indices in kept]
    statistics = ffn_statistics(hidden_sizes)

    statistics_path = Path(output_dir) / "flap_statistics.pt"
    atomic_torch_save(
        {
            **moment_pack,
            "standardized_wifv": standardized,
            "data_manifest": manifest,
            "target_ffn_reduction": target_reduction,
            "removed_channels": remove_count,
        },
        statistics_path,
    )
    checkpoint_path = Path(output_dir) / "flap_vision_pruned.pt"
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
        "kept_indices": [indices.cpu() for indices in kept],
        "hidden_sizes": hidden_sizes,
        "compensation_biases": compensations,
        "statistics": statistics,
        "visual_state_dict": {
            key: value.detach().cpu() for key, value in model.visual.state_dict().items()
        },
        "complete": True,
    }
    atomic_torch_save(checkpoint, checkpoint_path)
    validate_checkpoint(checkpoint)
    return checkpoint_path, statistics_path, checkpoint


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("method") != CHECKPOINT_METHOD or checkpoint.get("complete") is not True:
        raise ValueError("not a complete FLAP visual checkpoint")
    if not math.isclose(
        float(checkpoint.get("target_ffn_reduction", -1)),
        TARGET_FFN_REDUCTION,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("FLAP visual checkpoint has the wrong pruning target")
    manifest = checkpoint.get("data_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("FLAP visual checkpoint has no data manifest")
    validate_data_manifest(manifest)
    if checkpoint.get("seed") != manifest.get("processing_seed"):
        raise ValueError("FLAP visual checkpoint seed differs from its data manifest")
    kept = checkpoint.get("kept_indices")
    if not isinstance(kept, list) or len(kept) != N_LAYERS:
        raise ValueError("invalid retained-channel list in FLAP visual checkpoint")
    hidden_sizes = []
    for indices in kept:
        if (
            not isinstance(indices, Tensor)
            or indices.numel() == 0
            or indices.unique().numel() != indices.numel()
            or int(indices.min()) < 0
            or int(indices.max()) >= D_FFN
        ):
            raise ValueError("invalid FLAP visual retained-channel tensor")
        hidden_sizes.append(int(indices.numel()))
    if hidden_sizes != checkpoint.get("hidden_sizes"):
        raise ValueError("FLAP visual hidden sizes disagree with retained indices")
    calculated = ffn_statistics(hidden_sizes)
    expected_removed = round(N_LAYERS * D_FFN * TARGET_FFN_REDUCTION)
    if calculated["removed_channels"] != expected_removed:
        raise ValueError(
            "FLAP visual checkpoint structure does not match its declared pruning target"
        )
    recorded = checkpoint.get("statistics")
    if not isinstance(recorded, Mapping):
        raise ValueError("FLAP visual checkpoint has no statistics")
    for key, value in calculated.items():
        if key not in recorded or not math.isclose(
            float(recorded[key]), float(value), rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError(f"FLAP visual checkpoint statistic mismatch: {key}")
    if not isinstance(checkpoint.get("visual_state_dict"), Mapping):
        raise ValueError("FLAP visual checkpoint has no visual state dictionary")


def load_visual_state_dict(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    incompatible = model.visual.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "FLAP visual checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

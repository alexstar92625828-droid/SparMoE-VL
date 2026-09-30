"""FLAP WIFV pruning for the CLIP ViT-L/14 text FFNs."""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

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
    text_blocks,
)


METHOD = "FLAP-CLIP-FFN (Text, paired two-stage exposure)"
CHECKPOINT_METHOD = "FLAP-style CLIP Text-FFN"
PAPER = "https://arxiv.org/abs/2312.11983"
OFFICIAL_REPOSITORY = "https://github.com/CASIA-LMC-Lab/FLAP"
OFFICIAL_COMMIT = "3bb57db3449dd2fa04a5c2192de80e87e33be2b1"

TRAIN_STEPS_PER_STAGE = 5_000
TRAIN_BATCH_SIZE = 256
STAGE_EXPOSURES = TRAIN_STEPS_PER_STAGE * TRAIN_BATCH_SIZE
TOTAL_EXPOSURES = 2 * STAGE_EXPOSURES
TARGET_REDUCTIONS = {
    42: 0.4374995800700163,
    123: 0.4535723188148742,
    2026: 0.4624093863898526,
}
EXPECTED_EXPOSURE_SHA256 = {
    42: "9a9134e10a5a42e5a15a4781f81929e4fbf5feff9276baf7ab35fa096e710cf6",
    123: "e1f44c859c6f35648707924b4704fe57df790bb4fd12bb6c8ffe1860e2144998",
    2026: "fc90a97ba90fb3f6065c66a96d9a17260e4a5092dbc6aadf7c47c02784206fb3",
}
BUDGET_REFERENCE_SHA256 = "2204bebbdbc17c132e2baf5db0b2ca365074c0eb314c5db3ce03c971a76d3453"


def tensor_sha256(values: Tensor) -> str:
    array = values.detach().cpu().to(torch.int64).numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def one_stage_exposure_indices(seed: int) -> Tensor:
    """Reproduce the main text DataLoader's 5,000-step sample sequence."""

    generator = torch.Generator().manual_seed(seed)
    usable_per_epoch = (POOL_SIZE // TRAIN_BATCH_SIZE) * TRAIN_BATCH_SIZE
    parts = []
    remaining = STAGE_EXPOSURES
    while remaining:
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        permutation = torch.randperm(POOL_SIZE, generator=generator)
        take = min(remaining, usable_per_epoch)
        parts.append(permutation[:take].to(torch.int32))
        remaining -= take
    return torch.cat(parts)


def full_exposure_manifest(seed: int) -> tuple[Tensor, dict[str, Any]]:
    """Return the identical Stage-1/Stage-2 exposure protocol used by main."""

    if seed not in TARGET_REDUCTIONS:
        raise ValueError(f"unsupported paper seed: {seed}")
    stage = one_stage_exposure_indices(seed)
    indices = torch.cat((stage, stage))
    fingerprint = tensor_sha256(indices)
    if fingerprint != EXPECTED_EXPOSURE_SHA256[seed]:
        raise RuntimeError("FLAP exposure order differs from the recorded main run")
    unique_samples = int(torch.unique(indices).numel())
    if unique_samples != POOL_SIZE:
        raise RuntimeError("FLAP exposures do not cover the complete 500k pool")
    manifest = {
        "run_seed": int(seed),
        "data_seed": DATA_SEED,
        "dataset_sha256": EXPECTED_TEXT_POOL_SHA256,
        "pool_size": POOL_SIZE,
        "unique_samples": unique_samples,
        "uses_complete_pool": True,
        "train_steps_per_stage": TRAIN_STEPS_PER_STAGE,
        "batch_size": TRAIN_BATCH_SIZE,
        "stage1_exposures": STAGE_EXPOSURES,
        "stage2_exposures": STAGE_EXPOSURES,
        "stage_sequences_identical": True,
        "total_exposures": int(indices.numel()),
        "exposure_indices_sha256": fingerprint,
    }
    validate_data_manifest(manifest)
    return indices, manifest


def validate_data_manifest(manifest: dict[str, Any]) -> None:
    seed = manifest.get("run_seed")
    expected = {
        "data_seed": DATA_SEED,
        "dataset_sha256": EXPECTED_TEXT_POOL_SHA256,
        "pool_size": POOL_SIZE,
        "unique_samples": POOL_SIZE,
        "uses_complete_pool": True,
        "train_steps_per_stage": TRAIN_STEPS_PER_STAGE,
        "batch_size": TRAIN_BATCH_SIZE,
        "stage1_exposures": STAGE_EXPOSURES,
        "stage2_exposures": STAGE_EXPOSURES,
        "stage_sequences_identical": True,
        "total_exposures": TOTAL_EXPOSURES,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if seed not in EXPECTED_EXPOSURE_SHA256:
        mismatches["run_seed"] = (seed, tuple(EXPECTED_EXPOSURE_SHA256))
    elif manifest.get("exposure_indices_sha256") != EXPECTED_EXPOSURE_SHA256[seed]:
        mismatches["exposure_indices_sha256"] = (
            manifest.get("exposure_indices_sha256"),
            EXPECTED_EXPOSURE_SHA256[seed],
        )
    if mismatches:
        raise ValueError(f"FLAP data manifest differs from the main run: {mismatches}")


class RunningFeatureMoments:
    """Numerically stable streaming moments for one text FFN."""

    def __init__(self, device: torch.device, width: int = D_FFN) -> None:
        if width <= 0:
            raise ValueError("feature width must be positive")
        self.width = int(width)
        self.count = 0
        self.mean = torch.zeros(self.width, device=device, dtype=torch.float32)
        self.m2 = torch.zeros(self.width, device=device, dtype=torch.float32)

    @torch.no_grad()
    def add(self, activation: Tensor) -> None:
        if activation.shape[-1] != self.width:
            raise ValueError("activation width differs from the moment accumulator")
        values = activation.detach().reshape(-1, activation.shape[-1]).float()
        batch_count = values.shape[0]
        batch_var, batch_mean = torch.var_mean(values, dim=0, correction=0)
        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_var * batch_count)
            self.count = batch_count
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2.add_(batch_var * batch_count)
        self.m2.add_(delta.square() * (self.count * batch_count / total))
        self.mean.add_(delta * (batch_count / total))
        self.count = total

    def state_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean.detach().cpu(),
            "m2": self.m2.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            not isinstance(state.get("mean"), Tensor)
            or tuple(state["mean"].shape) != (self.width,)
            or not isinstance(state.get("m2"), Tensor)
            or tuple(state["m2"].shape) != (self.width,)
            or int(state.get("count", 0)) < 0
        ):
            raise RuntimeError("invalid FLAP streaming-moment checkpoint")
        self.count = int(state["count"])
        self.mean.copy_(state["mean"].to(self.mean))
        self.m2.copy_(state["m2"].to(self.m2))

    def finish(self) -> tuple[Tensor, Tensor]:
        if self.count <= 1:
            raise RuntimeError("insufficient text-token observations for FLAP")
        return (
            self.mean.detach().cpu(),
            (self.m2 / (self.count - 1)).clamp_min(0).detach().cpu(),
        )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def collect_wifv(
    model: nn.Module,
    tokens: Tensor,
    indices: Tensor,
    manifest: dict[str, Any],
    device: str,
    output_dir: Path,
    log_every: int = 50,
    save_every: int = 200,
    resume: bool = False,
) -> dict[str, Any]:
    """Collect FLAP WIFV statistics over both complete main-stage sequences."""

    validate_data_manifest(manifest)
    if tuple(tokens.shape) != (POOL_SIZE, N_TOKENS):
        raise ValueError("FLAP requires the complete 500k text token cache")
    if indices.numel() != TOTAL_EXPOSURES:
        raise ValueError("FLAP requires both complete main-stage exposure sequences")
    if tensor_sha256(indices) != manifest["exposure_indices_sha256"]:
        raise ValueError("FLAP exposure tensor differs from its data manifest")
    if log_every <= 0 or save_every <= 0:
        raise ValueError("FLAP progress intervals must be positive")

    torch_device = torch.device(device)
    blocks = text_blocks(model)
    moments = [RunningFeatureMoments(torch_device) for _ in blocks]
    progress_path = Path(output_dir) / "flap_calibration_progress.pt"
    next_batch = 0
    elapsed_before = 0.0
    if progress_path.exists():
        if not resume:
            raise RuntimeError(f"progress exists at {progress_path}; pass --resume")
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress.get("data_manifest") != manifest:
            raise RuntimeError("FLAP progress belongs to a different run")
        states = progress.get("moments")
        if not isinstance(states, list) or len(states) != N_LAYERS:
            raise RuntimeError("FLAP progress has invalid layer moments")
        for accumulator, state in zip(moments, states):
            accumulator.load_state_dict(state)
        next_batch = int(progress.get("next_batch", 0))
        elapsed_before = float(progress.get("elapsed_seconds", 0.0))

    handles = []
    for layer, block in enumerate(blocks):

        def hook(_module, inputs, layer_index=layer):
            moments[layer_index].add(inputs[0])

        handles.append(block.mlp.c_proj.register_forward_pre_hook(hook))

    batch_size = int(manifest["batch_size"])
    total_batches = TOTAL_EXPOSURES // batch_size
    expected_observations = next_batch * batch_size * N_TOKENS
    if not 0 <= next_batch <= total_batches or any(
        item.count != expected_observations for item in moments
    ):
        raise RuntimeError("FLAP progress counters are inconsistent")
    started = time.time()
    try:
        for batch_index in range(next_batch, total_batches):
            start = batch_index * batch_size
            selection = indices[start : start + batch_size].to(torch.int64)
            batch = tokens[selection]
            if torch_device.type == "cuda":
                batch = batch.pin_memory()
            batch = batch.to(torch_device, non_blocking=torch_device.type == "cuda")
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=torch_device.type == "cuda",
            ):
                model.encode_text(batch)

            completed = batch_index + 1
            elapsed = elapsed_before + time.time() - started
            if completed % save_every == 0 or completed == total_batches:
                _atomic_torch_save(
                    {
                        "data_manifest": manifest,
                        "moments": [item.state_dict() for item in moments],
                        "next_batch": completed,
                        "elapsed_seconds": elapsed,
                    },
                    progress_path,
                )
            if completed % log_every == 0 or completed == total_batches:
                print(
                    f"batch={completed}/{total_batches} "
                    f"samples={completed * batch_size}/{TOTAL_EXPOSURES} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    means, variances, raw_wifv = [], [], []
    for layer, (block, accumulator) in enumerate(zip(blocks, moments)):
        mean, variance = accumulator.finish()
        norm_sq = block.mlp.c_proj.weight.detach().float().cpu().square().sum(0)
        score = variance * norm_sq
        if not torch.isfinite(score).all() or float(score.std()) == 0.0:
            raise FloatingPointError(f"invalid FLAP WIFV at text layer {layer}")
        means.append(mean)
        variances.append(variance)
        raw_wifv.append(score)
    return {
        "means": torch.stack(means),
        "variances": torch.stack(variances),
        "raw_wifv": torch.stack(raw_wifv),
        "samples_seen": TOTAL_EXPOSURES,
        "tokens_per_layer": [item.count for item in moments],
        "elapsed_seconds": elapsed_before + time.time() - started,
        "complete": True,
    }


def select_channels(
    raw_wifv: Tensor,
    target_reduction: float,
) -> tuple[list[Tensor], Tensor, int]:
    if tuple(raw_wifv.shape) != (N_LAYERS, D_FFN):
        raise ValueError("FLAP WIFV must cover every text FFN channel")
    standardized = (raw_wifv - raw_wifv.mean(dim=1, keepdim=True)) / raw_wifv.std(
        dim=1,
        keepdim=True,
        correction=1,
    )
    if not torch.isfinite(standardized).all():
        raise FloatingPointError("FLAP standardized WIFV contains non-finite values")
    total = standardized.numel()
    remove_count = round(total * target_reduction)
    if not 0 < remove_count < total:
        raise ValueError("FLAP reduction must remove some but not all channels")
    remove_flat = torch.argsort(standardized.reshape(-1))[:remove_count]
    keep_mask = torch.ones(total, dtype=torch.bool)
    keep_mask[remove_flat] = False
    keep_mask = keep_mask.reshape(N_LAYERS, D_FFN)
    kept = [torch.where(keep_mask[layer])[0] for layer in range(N_LAYERS)]
    if any(indices.numel() == 0 for indices in kept):
        raise RuntimeError("FLAP allocation removed an entire text FFN")
    return kept, standardized, remove_count


@torch.no_grad()
def prune_pair(
    c_fc: nn.Linear,
    c_proj: nn.Linear,
    kept_cpu: Tensor,
    mean_cpu: Tensor,
) -> tuple[nn.Linear, nn.Linear, Tensor]:
    device, dtype = c_fc.weight.device, c_fc.weight.dtype
    kept = kept_cpu.to(device=device, dtype=torch.long)
    if kept.unique().numel() != kept.numel():
        raise ValueError("FLAP retained indices contain duplicates")
    remove_mask = torch.ones(D_FFN, device=device, dtype=torch.bool)
    remove_mask[kept] = False
    removed = torch.where(remove_mask)[0]
    compensation = (
        c_proj.weight[:, removed]
        .float()
        .matmul(mean_cpu.to(device=device, dtype=torch.float32)[removed])
    )
    original_bias = (
        c_proj.bias.detach().float()
        if c_proj.bias is not None
        else torch.zeros(D_MODEL, device=device)
    )
    new_fc = nn.Linear(
        D_MODEL,
        kept.numel(),
        bias=c_fc.bias is not None,
        device=device,
        dtype=dtype,
    )
    new_proj = nn.Linear(
        kept.numel(),
        D_MODEL,
        bias=True,
        device=device,
        dtype=dtype,
    )
    new_fc.weight.copy_(c_fc.weight.index_select(0, kept))
    if c_fc.bias is not None:
        new_fc.bias.copy_(c_fc.bias.index_select(0, kept))
    new_proj.weight.copy_(c_proj.weight.index_select(1, kept))
    new_proj.bias.copy_((original_bias + compensation).to(dtype))
    return new_fc, new_proj, compensation.cpu()


@torch.no_grad()
def apply_pruning(
    model: nn.Module,
    means: Tensor,
    kept_indices: Sequence[Tensor],
) -> list[Tensor]:
    if tuple(means.shape) != (N_LAYERS, D_FFN) or len(kept_indices) != N_LAYERS:
        raise ValueError("FLAP pruning state has the wrong shape")
    compensations = []
    for layer, (block, kept) in enumerate(zip(text_blocks(model), kept_indices)):
        block.mlp.c_fc, block.mlp.c_proj, compensation = prune_pair(
            block.mlp.c_fc,
            block.mlp.c_proj,
            kept,
            means[layer],
        )
        compensations.append(compensation)
    return compensations


@torch.no_grad()
def rebuild_pruned_structure(model: nn.Module, kept_indices: Sequence[Tensor]) -> None:
    """Create checkpoint-compatible FFN shapes before loading saved weights."""

    if len(kept_indices) != N_LAYERS:
        raise ValueError(f"expected {N_LAYERS} FLAP retained-index tensors")
    zero_means = torch.zeros((N_LAYERS, D_FFN), dtype=torch.float32)
    apply_pruning(model, zero_means, kept_indices)


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
            f"FLAP text checkpoint mismatch: unexpected={unexpected}, missing={missing}"
        )


def statistics(hidden_sizes: Sequence[int]) -> dict[str, float | int]:
    if len(hidden_sizes) != N_LAYERS or any(not 0 < size <= D_FFN for size in hidden_sizes):
        raise ValueError("invalid FLAP text FFN widths")
    retained = int(sum(hidden_sizes))
    total = N_LAYERS * D_FFN
    removed = total - retained
    ffn_macs = 2.0 * N_TOKENS * D_MODEL * retained / 1e9
    return {
        "hidden_size_sum": retained,
        "removed_channels": removed,
        "active_text_parameters_m": (DENSE_TRANSFORMER_PARAMETERS - removed * (2 * D_MODEL + 1))
        / 1e6,
        "macs_text_g": DENSE_TOTAL_MACS_G - DENSE_FFN_MACS_G + ffn_macs,
        "ffn_macs_text_g": ffn_macs,
        "dense_ffn_macs_text_g": DENSE_FFN_MACS_G,
        "ffn_reduction_percent": 100.0 * removed / total,
    }


def save_final_checkpoint(
    model: nn.Module,
    moment_pack: dict[str, Any],
    manifest: dict[str, Any],
    pretrained_sha256: str,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    validate_data_manifest(manifest)
    if moment_pack.get("complete") is not True:
        raise ValueError("incomplete FLAP moments cannot produce a checkpoint")
    seed = int(manifest["run_seed"])
    target = TARGET_REDUCTIONS[seed]
    kept, standardized, _ = select_channels(moment_pack["raw_wifv"], target)
    compensations = apply_pruning(model, moment_pack["means"], kept)
    hidden_sizes = [int(indices.numel()) for indices in kept]
    summary = statistics(hidden_sizes)

    statistics_path = Path(output_dir) / "flap_statistics.pt"
    _atomic_torch_save(
        {
            **moment_pack,
            "standardized_wifv": standardized,
            "data_manifest": manifest,
        },
        statistics_path,
    )
    checkpoint_path = Path(output_dir) / "flap_text_pruned.pt"
    checkpoint = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "seed": seed,
        "data_seed": DATA_SEED,
        "pretrained_sha256": pretrained_sha256,
        "data_manifest": manifest,
        "paired_budget_target": target,
        "budget_reference_sha256": BUDGET_REFERENCE_SHA256,
        "text_state_dict": text_state_dict(model),
        "kept_indices": kept,
        "hidden_sizes": hidden_sizes,
        "compensation_biases": compensations,
        "statistics": summary,
        "complete": True,
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    validate_checkpoint(checkpoint)
    return checkpoint_path, statistics_path, checkpoint


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("method") != CHECKPOINT_METHOD or checkpoint.get("complete") is not True:
        raise ValueError("not a complete FLAP text checkpoint")
    validate_data_manifest(checkpoint.get("data_manifest", {}))
    seed = checkpoint["data_manifest"]["run_seed"]
    if checkpoint.get("seed") != seed or checkpoint.get("data_seed") != DATA_SEED:
        raise ValueError("FLAP checkpoint seed metadata is inconsistent")
    if not math.isclose(
        float(checkpoint.get("paired_budget_target", -1)),
        TARGET_REDUCTIONS[seed],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("FLAP paired budget differs from the paper run")
    kept = checkpoint.get("kept_indices")
    if not isinstance(kept, list) or len(kept) != N_LAYERS:
        raise ValueError("invalid retained-channel list in FLAP checkpoint")
    hidden_sizes = [int(indices.numel()) for indices in kept]
    if hidden_sizes != checkpoint.get("hidden_sizes"):
        raise ValueError("FLAP hidden sizes disagree with retained indices")
    compensations = checkpoint.get("compensation_biases")
    if (
        not isinstance(compensations, list)
        or len(compensations) != N_LAYERS
        or any(
            not isinstance(item, Tensor) or tuple(item.shape) != (D_MODEL,)
            for item in compensations
        )
    ):
        raise ValueError("FLAP checkpoint has invalid compensation biases")
    calculated = statistics(hidden_sizes)
    recorded = checkpoint.get("statistics", {})
    for key, value in calculated.items():
        if key not in recorded or not math.isclose(
            float(recorded[key]),
            float(value),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"FLAP checkpoint statistic mismatch: {key}")

"""MoPE-CLIP recovery for structurally pruned CLIP text FFNs."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Sequence

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
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_TEXT_POOL_SHA256,
    N_LAYERS,
    N_TOKENS,
    POOL_SIZE,
    PROJECT_ROOT,
    text_blocks,
)


METHOD = "MoPE-CLIP-FFN (Text, paired two-stage recovery)"
CHECKPOINT_METHOD = "MoPE-CLIP Text-FFN"
PAPER = "https://arxiv.org/abs/2403.07839"

GROUP_SIZE = 16
NUM_GROUPS = D_FFN // GROUP_SIZE
STRUCTURE_SEED = 42
TRAIN_STEPS_PER_STAGE = 5_000
TRAIN_BATCH_SIZE = 256
STAGE_EXPOSURES = TRAIN_STEPS_PER_STAGE * TRAIN_BATCH_SIZE
TOTAL_EXPOSURES = 2 * STAGE_EXPOSURES

TARGET_REDUCTIONS = {
    42: 0.4374995800700163,
    123: 0.4535723188148742,
    2026: 0.4624093863898526,
}
TARGET_KEEP_GROUPS = {42: 108, 123: 105, 2026: 103}
EXPECTED_STAGE_EXPOSURE_SHA256 = {
    42: "c5f8db11ce91eb85d283574e54d2ab1b65de4554d355e9f11a6dfc11b62713d7",
    123: "d2a881f29859355fa9e02eb8311b168851876aff869217a5085c6a6e9a24d204",
    2026: "6c60b526d8509aee7a855d8c1e9e793da193acf160565f671975ecdf04273b39",
}

MOPE_DENSE_TEXT_CACHE = (
    PROJECT_ROOT / "data" / "cache" / "sharegpt4v_clip_vitl14_text_features_data_seed42_500k.pt"
)


def tensor_sha256(values: Tensor) -> str:
    array = values.detach().cpu().to(torch.int64).numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def one_stage_exposure_indices(seed: int) -> Tensor:
    """Reproduce one 5,000-step text-main DataLoader sequence."""

    if seed not in TARGET_REDUCTIONS:
        raise ValueError(f"unsupported paper seed: {seed}")
    generator = torch.Generator().manual_seed(seed)
    usable_per_epoch = (POOL_SIZE // TRAIN_BATCH_SIZE) * TRAIN_BATCH_SIZE
    parts = []
    remaining = STAGE_EXPOSURES
    while remaining:
        # DataLoader consumes one generator draw when an iterator is created.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        permutation = torch.randperm(POOL_SIZE, generator=generator)
        take = min(remaining, usable_per_epoch)
        parts.append(permutation[:take].to(torch.int32))
        remaining -= take
    indices = torch.cat(parts)
    if tensor_sha256(indices) != EXPECTED_STAGE_EXPOSURE_SHA256[seed]:
        raise RuntimeError("MoPE exposure order differs from the recorded text main run")
    if torch.unique(indices).numel() != POOL_SIZE:
        raise RuntimeError("MoPE stage does not cover the complete 500k pool")
    return indices


def recovery_data_manifest(seed: int) -> dict[str, Any]:
    """Describe and validate the identical Stage-1/Stage-2 data protocol."""

    indices = one_stage_exposure_indices(seed)
    manifest = {
        "run_seed": int(seed),
        "data_seed": DATA_SEED,
        "dataset_sha256": EXPECTED_TEXT_POOL_SHA256,
        "paired_path_pool_sha256": EXPECTED_PAIRED_PATH_POOL_SHA256,
        "pool_size": POOL_SIZE,
        "unique_samples": int(torch.unique(indices).numel()),
        "uses_complete_pool": True,
        "train_steps_per_stage": TRAIN_STEPS_PER_STAGE,
        "batch_size": TRAIN_BATCH_SIZE,
        "stage1_exposures": STAGE_EXPOSURES,
        "stage2_exposures": STAGE_EXPOSURES,
        "stage_sequences_identical": True,
        "stage_exposure_indices_sha256": tensor_sha256(indices),
        "total_exposures": TOTAL_EXPOSURES,
    }
    validate_recovery_data_manifest(manifest)
    return manifest


def validate_recovery_data_manifest(manifest: dict[str, Any]) -> None:
    seed = manifest.get("run_seed")
    expected = {
        "data_seed": DATA_SEED,
        "dataset_sha256": EXPECTED_TEXT_POOL_SHA256,
        "paired_path_pool_sha256": EXPECTED_PAIRED_PATH_POOL_SHA256,
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
    if seed not in EXPECTED_STAGE_EXPOSURE_SHA256:
        mismatches["run_seed"] = (seed, tuple(EXPECTED_STAGE_EXPOSURE_SHA256))
    elif manifest.get("stage_exposure_indices_sha256") != EXPECTED_STAGE_EXPOSURE_SHA256[seed]:
        mismatches["stage_exposure_indices_sha256"] = (
            manifest.get("stage_exposure_indices_sha256"),
            EXPECTED_STAGE_EXPOSURE_SHA256[seed],
        )
    if mismatches:
        raise ValueError(f"MoPE data manifest differs from the text main run: {mismatches}")


def structure_data_manifest() -> dict[str, Any]:
    """Return the full-pool structure-selection contract (no held-out subset)."""

    return {
        "data_seed": DATA_SEED,
        "dataset_sha256": EXPECTED_TEXT_POOL_SHA256,
        "paired_path_pool_sha256": EXPECTED_PAIRED_PATH_POOL_SHA256,
        "pool_size": POOL_SIZE,
        "unique_samples": POOL_SIZE,
        "selection_samples": POOL_SIZE,
        "uses_complete_main_pool": True,
        "held_out_samples": 0,
    }


def validate_dense_text_cache(payload: dict[str, Any]) -> None:
    features = payload.get("features")
    expected = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "text_pool_sha256": EXPECTED_TEXT_POOL_SHA256,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("MoPE dense-text cache metadata does not match the main pool")
    if (
        not isinstance(features, Tensor)
        or tuple(features.shape) != (POOL_SIZE, D_MODEL)
        or features.dtype != torch.float16
        or not torch.isfinite(features).all()
    ):
        raise RuntimeError("MoPE dense-text cache must contain finite [500000, 768] fp16")


@torch.no_grad()
def prepare_dense_text_cache(
    model: nn.Module,
    tokens: Tensor,
    device: str,
    pretrained_sha256: str,
    cache_path: Path = MOPE_DENSE_TEXT_CACHE,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Cache dense text features for all 500k main-pool captions."""

    cache_path = Path(cache_path)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid MoPE dense-text cache: {cache_path}")
        validate_dense_text_cache(payload)
        return payload
    if tuple(tokens.shape) != (POOL_SIZE, N_TOKENS) or batch_size <= 0:
        raise ValueError("MoPE requires all 500k tokens and a positive batch size")

    outputs = []
    torch_device = torch.device(device)
    for start in range(0, POOL_SIZE, batch_size):
        batch = tokens[start : start + batch_size].to(
            torch_device,
            non_blocking=torch_device.type == "cuda",
        )
        with torch.autocast(
            device_type=torch_device.type,
            dtype=torch.float16,
            enabled=torch_device.type == "cuda",
        ):
            outputs.append(F.normalize(model.encode_text(batch), dim=-1).half().cpu())
        if (start // batch_size + 1) % 100 == 0:
            print(f"dense_text_samples={min(start + batch_size, POOL_SIZE)}/{POOL_SIZE}")
    payload = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "text_pool_sha256": EXPECTED_TEXT_POOL_SHA256,
        "pretrained_sha256": pretrained_sha256,
        "features": torch.cat(outputs),
    }
    validate_dense_text_cache(payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    return payload


def contrastive_loss(image_features: Tensor, text_features: Tensor, scale: Tensor) -> Tensor:
    logits = scale * image_features @ text_features.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def ffn_taylor_scores(model: nn.Module) -> list[Tensor]:
    scores = []
    for layer, block in enumerate(text_blocks(model)):
        fc, proj = block.mlp.c_fc, block.mlp.c_proj
        if fc.weight.grad is None or fc.bias.grad is None or proj.weight.grad is None:
            raise RuntimeError(f"missing MoPE FFN gradient at text layer {layer}")
        incoming = (fc.weight * fc.weight.grad).sum(dim=1) + fc.bias * fc.bias.grad
        outgoing = (proj.weight * proj.weight.grad).sum(dim=0)
        scores.append((incoming.abs() + outgoing.abs()).detach().float().cpu())
    return scores


def rankings_to_groups(scores: Tensor, group_size: int = GROUP_SIZE) -> tuple[Tensor, Tensor]:
    if tuple(scores.shape) != (N_LAYERS, D_FFN):
        raise ValueError(f"MoPE scores must have shape {(N_LAYERS, D_FFN)}")
    if group_size <= 0 or D_FFN % group_size:
        raise ValueError("MoPE group size must divide the FFN width")
    rankings = torch.argsort(scores.float(), dim=1, descending=True, stable=True)
    return rankings, rankings.reshape(N_LAYERS, D_FFN // group_size, group_size)


def register_group_ablation(model: nn.Module, groups: Tensor, group_index: int) -> list[Any]:
    if groups.ndim != 3 or groups.shape[0] != N_LAYERS:
        raise ValueError("invalid MoPE ranked groups")
    if not 0 <= group_index < groups.shape[1]:
        raise ValueError("MoPE group index is out of range")
    handles = []
    for layer, block in enumerate(text_blocks(model)):
        indices_cpu = groups[layer, group_index].clone()

        def hook(
            _module: nn.Module, _inputs: Any, output: Tensor, indices=indices_cpu
        ) -> Tensor:
            masked = output.clone()
            masked.index_fill_(-1, indices.to(output.device), 0)
            return masked

        handles.append(block.mlp.gelu.register_forward_hook(hook))
    return handles


def recall_counts(image_features: Tensor, text_features: Tensor) -> Tensor:
    """Return paired-image retrieval hits at R@1/5/10 for one batch."""

    if image_features.shape != text_features.shape or image_features.ndim != 2:
        raise ValueError("MoPE paired features must have identical [batch, dim] shapes")
    similarity = image_features.float() @ text_features.float().T
    targets = torch.arange(similarity.shape[0])
    order = similarity.argsort(dim=1, descending=True)
    return torch.tensor(
        [int((order[:, :k] == targets[:, None]).any(dim=1).sum()) for k in (1, 5, 10)],
        dtype=torch.int64,
    )


def recall_percentages(counts: Tensor, samples: int) -> dict[str, float]:
    if tuple(counts.shape) != (3,) or samples <= 0:
        raise ValueError("MoPE recall totals are invalid")
    values = 100.0 * counts.double() / samples
    return {
        "r1": float(values[0]),
        "r5": float(values[1]),
        "r10": float(values[2]),
        "mean": float(values.mean()),
    }


def selected_groups(selection: dict[str, Any], seed: int) -> list[int]:
    validate_selection(selection)
    if seed not in TARGET_KEEP_GROUPS:
        raise ValueError(f"unsupported paper seed: {seed}")
    count = TARGET_KEEP_GROUPS[seed]
    return sorted(int(index) for index in selection["group_priority"][:count])


def kept_channel_indices(groups: Tensor, kept_groups: Sequence[int]) -> list[Tensor]:
    if tuple(groups.shape) != (N_LAYERS, NUM_GROUPS, GROUP_SIZE):
        raise ValueError("MoPE groups have the wrong shape")
    kept = sorted(int(index) for index in kept_groups)
    if len(set(kept)) != len(kept) or not kept or kept[-1] >= NUM_GROUPS:
        raise ValueError("MoPE retained group indices are invalid")
    return [groups[layer, kept].reshape(-1).clone() for layer in range(N_LAYERS)]


@torch.no_grad()
def structurally_prune_text_ffn(model: nn.Module, kept_indices: Sequence[Tensor]) -> None:
    if len(kept_indices) != N_LAYERS:
        raise ValueError(f"expected {N_LAYERS} MoPE retained-index tensors")
    for layer, (block, indices_cpu) in enumerate(zip(text_blocks(model), kept_indices)):
        old_fc, old_proj = block.mlp.c_fc, block.mlp.c_proj
        indices = indices_cpu.to(old_fc.weight.device, dtype=torch.long)
        if indices.unique().numel() != indices.numel():
            raise ValueError(f"duplicate MoPE channel at text layer {layer}")
        width = indices.numel()
        new_fc = nn.Linear(
            D_MODEL,
            width,
            bias=old_fc.bias is not None,
            device=old_fc.weight.device,
            dtype=old_fc.weight.dtype,
        )
        new_proj = nn.Linear(
            width,
            D_MODEL,
            bias=old_proj.bias is not None,
            device=old_proj.weight.device,
            dtype=old_proj.weight.dtype,
        )
        new_fc.weight.copy_(old_fc.weight.index_select(0, indices))
        if old_fc.bias is not None:
            new_fc.bias.copy_(old_fc.bias.index_select(0, indices))
        new_proj.weight.copy_(old_proj.weight.index_select(1, indices))
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)
        block.mlp.c_fc, block.mlp.c_proj = new_fc, new_proj


def validate_pruned_text(model: nn.Module, width: int) -> None:
    for layer, block in enumerate(text_blocks(model)):
        if tuple(block.mlp.c_fc.weight.shape) != (width, D_MODEL):
            raise ValueError(f"incorrect MoPE c_fc shape at text layer {layer}")
        if tuple(block.mlp.c_proj.weight.shape) != (D_MODEL, width):
            raise ValueError(f"incorrect MoPE c_proj shape at text layer {layer}")


def text_parameters(model: nn.Module) -> list[nn.Parameter]:
    modules = [model.token_embedding, model.transformer, model.ln_final]
    parameters = [parameter for module in modules for parameter in module.parameters()]
    parameters.append(model.positional_embedding)
    if isinstance(model.text_projection, nn.Module):
        parameters.extend(model.text_projection.parameters())
    elif isinstance(model.text_projection, nn.Parameter):
        parameters.append(model.text_projection)
    unique, seen = [], set()
    for parameter in parameters:
        if id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    return unique


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
            f"MoPE text checkpoint mismatch: unexpected={unexpected}, missing={missing}"
        )


class HiddenCapture:
    """Capture all Transformer-block outputs for hidden-state distillation."""

    def __init__(self, model: nn.Module) -> None:
        self.outputs: list[Tensor] = []
        self.handles = [block.register_forward_hook(self._hook) for block in text_blocks(model)]

    def _hook(self, _module: nn.Module, _inputs: Any, output: Tensor) -> None:
        self.outputs.append(output)

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def soft_cross_entropy(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    teacher_probability = F.softmax(teacher_logits.float(), dim=-1)
    student_log_probability = F.log_softmax(student_logits.float(), dim=-1)
    return -(teacher_probability * student_log_probability).sum(dim=-1).mean()


def cross_modal_losses(
    student_text: Tensor,
    teacher_text: Tensor,
    image_features: Tensor,
    scale: Tensor,
) -> tuple[Tensor, Tensor]:
    labels = torch.arange(student_text.shape[0], device=student_text.device)
    student_i2t = scale * image_features @ student_text.T
    student_t2i = scale * student_text @ image_features.T
    teacher_i2t = scale * image_features @ teacher_text.T
    teacher_t2i = scale * teacher_text @ image_features.T
    itc = 0.5 * (F.cross_entropy(student_i2t, labels) + F.cross_entropy(student_t2i, labels))
    similarity = 0.5 * (
        soft_cross_entropy(student_i2t, teacher_i2t.detach())
        + soft_cross_entropy(student_t2i, teacher_t2i.detach())
    )
    return itc, similarity


def cosine_warmup_lambda(step: int, total_steps: int = 2 * TRAIN_STEPS_PER_STAGE) -> float:
    warmup_steps = total_steps // 10
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def validate_selection(selection: dict[str, Any]) -> None:
    expected = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": "structure_selection",
        "structure_seed": STRUCTURE_SEED,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "group_size": GROUP_SIZE,
        "num_groups": NUM_GROUPS,
        "complete": True,
    }
    mismatches = {
        key: (selection.get(key), value)
        for key, value in expected.items()
        if selection.get(key) != value
    }
    if selection.get("data_manifest") != structure_data_manifest():
        mismatches["data_manifest"] = (
            selection.get("data_manifest"),
            structure_data_manifest(),
        )
    groups = selection.get("groups")
    priority = selection.get("group_priority")
    if not isinstance(groups, Tensor) or tuple(groups.shape) != (
        N_LAYERS,
        NUM_GROUPS,
        GROUP_SIZE,
    ):
        mismatches["groups"] = (getattr(groups, "shape", None), "[12,192,16]")
    if not isinstance(priority, list) or sorted(priority) != list(range(NUM_GROUPS)):
        mismatches["group_priority"] = (priority, "permutation of 0..191")
    if mismatches:
        raise ValueError(f"invalid MoPE structure selection: {mismatches}")


def statistics(retained_width: int) -> dict[str, float | int]:
    if not 0 < retained_width <= D_FFN:
        raise ValueError("invalid MoPE retained FFN width")
    removed_per_layer = D_FFN - retained_width
    removed = N_LAYERS * removed_per_layer
    ffn_macs = N_LAYERS * 2.0 * N_TOKENS * D_MODEL * retained_width / 1e9
    return {
        "retained_width": retained_width,
        "hidden_size_sum": N_LAYERS * retained_width,
        "removed_channels": removed,
        "active_text_parameters_m": (DENSE_TRANSFORMER_PARAMETERS - removed * (2 * D_MODEL + 1))
        / 1e6,
        "macs_text_g": DENSE_TOTAL_MACS_G - DENSE_FFN_MACS_G + ffn_macs,
        "ffn_macs_text_g": ffn_macs,
        "dense_ffn_macs_text_g": DENSE_FFN_MACS_G,
        "ffn_reduction_percent": 100.0 * removed_per_layer / D_FFN,
    }


def validate_final_checkpoint(checkpoint: dict[str, Any]) -> None:
    seed = checkpoint.get("seed")
    if seed not in TARGET_KEEP_GROUPS:
        raise ValueError("MoPE checkpoint has an unsupported paper seed")
    retained_width = TARGET_KEEP_GROUPS[seed] * GROUP_SIZE
    expected = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": 2,
        "stage_step": TRAIN_STEPS_PER_STAGE,
        "global_step": 2 * TRAIN_STEPS_PER_STAGE,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "group_size": GROUP_SIZE,
        "retained_width": retained_width,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    try:
        validate_recovery_data_manifest(checkpoint.get("data_manifest", {}))
    except ValueError as error:
        mismatches["data_manifest"] = (str(error), "valid main two-stage manifest")
    if checkpoint.get("statistics") != statistics(retained_width):
        mismatches["statistics"] = (checkpoint.get("statistics"), statistics(retained_width))
    if "text_state_dict" not in checkpoint:
        mismatches["text_state_dict"] = ("missing", "present")
    if mismatches:
        raise ValueError(f"invalid final MoPE checkpoint: {mismatches}")

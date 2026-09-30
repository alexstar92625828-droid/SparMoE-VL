"""Identity and structure checks for routing-granularity checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from ...common.two_stage import STAGE1_PROTOCOL, protocol_for_stage
from .protocol import (
    DATA_SEED,
    EXPERT_COUNTS,
    MODEL_KEY,
    MODEL_NAME,
    MODALITY,
    RUN_SEED,
    STUDY_NAME,
    TARGET_RATIO,
    TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
    TRAIN_STEPS,
    VISION_LAYERS,
    capacity_factors,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_load(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a checkpoint mapping in {checkpoint_path}")
    return payload


def _controller_parts(
    checkpoint: Mapping[str, Any],
) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
    controller = checkpoint.get("controller")
    hypernetwork = controller.get("hypernetwork") if isinstance(controller, Mapping) else None
    layers = controller.get("layers") if isinstance(controller, Mapping) else None
    if not isinstance(hypernetwork, Mapping) or not isinstance(layers, Mapping):
        raise ValueError("checkpoint has no paper-protocol controller state")
    return hypernetwork, layers


def validate_controller_shapes(
    checkpoint: Mapping[str, Any], expert_count: int, *, stage: int
) -> None:
    """Require complete nested structure and a router only for Stage 2."""

    hypernetwork, layers = _controller_parts(checkpoint)
    latent = hypernetwork.get("z")
    if not isinstance(latent, Tensor) or tuple(latent.shape) != (expert_count, 32):
        raise ValueError(
            f"checkpoint latent codes have shape {getattr(latent, 'shape', None)}; "
            f"expected ({expert_count}, 32)"
        )
    for layer_index in range(VISION_LAYERS):
        prefix = str(layer_index)
        ratio = layers.get(f"{prefix}.full_ratio_logit")
        projection = layers.get(f"{prefix}.proj_mlp_d.2.weight")
        router = layers.get(f"{prefix}.router.weight")
        if not isinstance(ratio, Tensor) or ratio.numel() != 1:
            raise ValueError(f"checkpoint layer {layer_index} has no reference capacity")
        if not isinstance(projection, Tensor) or tuple(projection.shape) != (4_096, 128):
            raise ValueError(f"checkpoint layer {layer_index} has an invalid SPG projection")
        if stage == 1 and router is not None:
            raise ValueError("Stage-1 checkpoint must not contain token-router weights")
        if stage == 2 and (
            not isinstance(router, Tensor) or tuple(router.shape) != (expert_count, 1_024)
        ):
            raise ValueError(f"checkpoint layer {layer_index} has an invalid token router")


def _validate_common(
    checkpoint: Mapping[str, Any], expert_count: int, *, stage: int
) -> dict[str, Any]:
    if checkpoint.get("format_version") != 3:
        raise ValueError("checkpoint must use release format_version=3")
    if checkpoint.get("method") != "sparmoe_vl_routing_granularity":
        raise ValueError("checkpoint method is not the routing-granularity study")
    dataset = checkpoint.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("checkpoint is missing its dataset identity")
    expected = {
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "modality": MODALITY,
        "stage": stage,
        "training_protocol": protocol_for_stage(stage),
        "expert_count": expert_count,
        "training_seed": RUN_SEED,
    }
    for key, wanted in expected.items():
        if checkpoint.get(key) != wanted:
            raise ValueError(f"checkpoint {key}={checkpoint.get(key)!r}; expected {wanted!r}")
    expected_dataset = {
        "data_seed": DATA_SEED,
        "samples": TRAINING_POOL_SIZE,
        "ordered_sha256": TRAINING_POOL_SHA256,
    }
    for key, wanted in expected_dataset.items():
        if dataset.get(key) != wanted:
            raise ValueError(
                f"checkpoint dataset.{key}={dataset.get(key)!r}; expected {wanted!r}"
            )
    levels = tuple(float(value) for value in checkpoint.get("capacity_factors") or ())
    if levels != capacity_factors(expert_count):
        raise ValueError(
            f"checkpoint capacity factors {levels}; expected {capacity_factors(expert_count)}"
        )
    if abs(float(checkpoint.get("target_ratio", -1)) - TARGET_RATIO) > 1e-8:
        raise ValueError("checkpoint target ratio differs from global budget P")
    step = checkpoint.get("step")
    if not isinstance(step, int) or not 0 < step <= TRAIN_STEPS:
        raise ValueError(f"checkpoint step={step!r} is outside the training protocol")
    validate_controller_shapes(checkpoint, expert_count, stage=stage)
    return {
        "format": "release_v3",
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "modality": MODALITY,
        "stage": stage,
        "training_protocol": expected["training_protocol"],
        "expert_count": expert_count,
        "levels": list(levels),
        "target_ratio": TARGET_RATIO,
        "training_seed": RUN_SEED,
        "data_seed": DATA_SEED,
        "pool_size": TRAINING_POOL_SIZE,
        "dataset_sha256": TRAINING_POOL_SHA256,
        "checkpoint_step": step,
    }


def stage1_metadata(
    checkpoint: Mapping[str, Any], expert_count: int, *, checkpoint_sha256: str | None = None
) -> dict[str, Any]:
    """Validate SPG/subspace learning for one expert granularity."""

    if expert_count not in EXPERT_COUNTS:
        raise ValueError(f"expert_count must be one of {EXPERT_COUNTS}")
    metadata = _validate_common(checkpoint, expert_count, stage=1)
    if checkpoint_sha256 is not None:
        metadata["checkpoint_sha256"] = checkpoint_sha256
    return metadata


def checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    expert_count: int,
    *,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a Stage-2 router checkpoint with frozen Stage-1 structure."""

    if expert_count not in EXPERT_COUNTS:
        raise ValueError(f"expert_count must be one of {EXPERT_COUNTS}")
    metadata = _validate_common(checkpoint, expert_count, stage=2)
    stage1 = checkpoint.get("stage1")
    if not isinstance(stage1, Mapping):
        raise ValueError("Stage-2 checkpoint is missing its Stage-1 identity")
    expected_stage1 = {
        "training_protocol": STAGE1_PROTOCOL,
        "expert_count": expert_count,
        "dataset_sha256": TRAINING_POOL_SHA256,
    }
    for key, wanted in expected_stage1.items():
        if stage1.get(key) != wanted:
            raise ValueError(
                f"Stage-2 checkpoint stage1.{key}={stage1.get(key)!r}; expected {wanted!r}"
            )
    if checkpoint_sha256 is not None:
        metadata["checkpoint_sha256"] = checkpoint_sha256
    return metadata


def inspect_checkpoint(path: str | Path, expert_count: int) -> dict[str, Any]:
    checkpoint_path = Path(path)
    digest = file_sha256(checkpoint_path)
    metadata = checkpoint_metadata(
        torch_load(checkpoint_path), expert_count, checkpoint_sha256=digest
    )
    metadata["checkpoint"] = str(checkpoint_path.resolve())
    return metadata

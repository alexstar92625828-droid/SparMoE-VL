"""Checkpoint validation for the strict two-stage capacity study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ...common.two_stage import STAGE1_PROTOCOL, protocol_for_stage
from .protocol import (
    CAPACITY_FACTORS,
    DATA_SEED,
    MODEL_KEY,
    MODEL_NAME,
    POOL_SIZE,
    SEEDS,
    STUDY_NAME,
    TARGET_RATIO,
    TRAIN_STEPS,
    VISION_DATASET_SHA256,
)


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


def checkpoint_metadata(
    checkpoint: Mapping[str, Any], *, expected_stage: int = 2
) -> dict[str, Any]:
    """Reject weights that do not follow the two-stage N=8 protocol."""

    if expected_stage not in (1, 2):
        raise ValueError("expected_stage must be 1 or 2")
    dataset = checkpoint.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("checkpoint is missing dataset identity")
    expected_protocol = protocol_for_stage(expected_stage)
    expected = {
        "format_version": 3,
        "method": "sparmoe_vl_capacity_intervention",
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "training_protocol": expected_protocol,
        "stage": expected_stage,
        "modality": "vision",
    }
    for key, wanted in expected.items():
        if checkpoint.get(key) != wanted:
            raise ValueError(f"checkpoint {key}={checkpoint.get(key)!r}; expected {wanted!r}")
    expected_dataset = {
        "data_seed": DATA_SEED,
        "samples": POOL_SIZE,
        "ordered_sha256": VISION_DATASET_SHA256,
    }
    for key, wanted in expected_dataset.items():
        if dataset.get(key) != wanted:
            raise ValueError(
                f"checkpoint dataset.{key}={dataset.get(key)!r}; expected {wanted!r}"
            )
    seed = int(checkpoint.get("training_seed", -1))
    if seed not in SEEDS:
        raise ValueError(f"checkpoint training seed {seed} is not one of {SEEDS}")
    if abs(float(checkpoint.get("target_ratio", -1)) - TARGET_RATIO) > 1e-8:
        raise ValueError("checkpoint target ratio differs from global budget P")
    levels = tuple(float(value) for value in checkpoint.get("capacity_factors") or ())
    if levels != CAPACITY_FACTORS:
        raise ValueError("checkpoint does not contain the eight registered capacities")
    step = checkpoint.get("step")
    if not isinstance(step, int) or not 0 < step <= TRAIN_STEPS:
        raise ValueError("checkpoint step is outside the registered protocol")
    controller = checkpoint.get("controller")
    if not isinstance(controller, Mapping):
        raise ValueError("checkpoint is missing controller state")
    layers = controller.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError("checkpoint is missing per-layer controller state")
    has_router = any(str(key).endswith("router.weight") for key in layers)
    if expected_stage == 1 and has_router:
        raise ValueError("Stage-1 checkpoint must not contain token-router weights")
    if expected_stage == 2 and not has_router:
        raise ValueError("Stage-2 checkpoint is missing token-router weights")
    if expected_stage == 2:
        stage1 = checkpoint.get("stage1")
        if not isinstance(stage1, Mapping):
            raise ValueError("Stage-2 checkpoint is missing Stage-1 identity")
        if stage1.get("dataset_sha256") != VISION_DATASET_SHA256:
            raise ValueError("Stage 1 and Stage 2 dataset identities differ")
        if stage1.get("training_protocol") != STAGE1_PROTOCOL:
            raise ValueError("Stage-2 checkpoint does not reference a valid Stage 1")
    return {
        "format": "release_v3",
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "protocol": expected_protocol,
        "stage": expected_stage,
        "modality": "vision",
        "training_seed": seed,
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "dataset_sha256": VISION_DATASET_SHA256,
        "target_ratio": TARGET_RATIO,
        "levels": list(CAPACITY_FACTORS),
        "checkpoint_step": step,
    }


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    metadata = checkpoint_metadata(torch_load(checkpoint_path), expected_stage=2)
    metadata["checkpoint"] = str(checkpoint_path.resolve())
    return metadata

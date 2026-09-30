"""Checkpoint validation for the strict two-stage component ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ...common.two_stage import STAGE1_PROTOCOL, protocol_for_stage
from ..vision_budget_sweep import checkpoint_metadata as main_checkpoint_metadata
from .protocol import (
    CAPACITY_FACTORS,
    DATASET_SHA256,
    DATA_SEED,
    MODEL_KEY,
    MODEL_NAME,
    POOL_SIZE,
    REPLACEMENTS,
    TARGET_RATIO,
    TRAIN_STEPS,
    TRAINED_METHODS,
    validate_method_seed,
)


def torch_load(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        value = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"expected a checkpoint mapping in {checkpoint_path}")
    return value


def _state_parts(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    controller = checkpoint.get("controller")
    layers = controller.get("layers") if isinstance(controller, Mapping) else None
    if not isinstance(layers, Mapping):
        raise ValueError("component-ablation checkpoint has no controller state")
    return layers


def checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    method: str,
    *,
    expected_stage: int = 2,
) -> dict[str, Any]:
    if method not in TRAINED_METHODS:
        raise ValueError(f"{method} does not have a separately trained checkpoint")
    if expected_stage not in (1, 2):
        raise ValueError("expected_stage must be 1 or 2")
    dataset = checkpoint.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("checkpoint is missing dataset identity")
    expected_protocol = protocol_for_stage(expected_stage)
    expected = {
        "format_version": 3,
        "method": "sparmoe_vl_component_ablation",
        "variant": method,
        "replacement": REPLACEMENTS[method],
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
        "ordered_sha256": DATASET_SHA256,
    }
    for key, wanted in expected_dataset.items():
        if dataset.get(key) != wanted:
            raise ValueError(
                f"checkpoint dataset.{key}={dataset.get(key)!r}; expected {wanted!r}"
            )
    seed = int(checkpoint.get("training_seed", -1))
    validate_method_seed(method, seed)
    if abs(float(checkpoint.get("target_ratio", -1)) - TARGET_RATIO) > 1e-8:
        raise ValueError("checkpoint target ratio differs from global budget P")
    levels = tuple(float(value) for value in checkpoint.get("capacity_factors") or ())
    if levels != CAPACITY_FACTORS:
        raise ValueError("checkpoint capacity factors differ from the registered study")
    step = checkpoint.get("step")
    if not isinstance(step, int) or not 0 < step <= TRAIN_STEPS:
        raise ValueError("checkpoint step is outside the registered protocol")
    layers = _state_parts(checkpoint)
    has_router = any(str(key).endswith("router.weight") for key in layers)
    if expected_stage == 1 and has_router:
        raise ValueError("Stage-1 checkpoint must not contain token-router weights")
    if expected_stage == 2 and not has_router:
        raise ValueError("Stage-2 checkpoint is missing token-router weights")
    if method == "without_spg":
        missing = [
            layer for layer in range(24) if f"{layer}.independent_mask_logits" not in layers
        ]
        if missing:
            raise ValueError(f"w/o SPG checkpoint is missing masks for layers {missing}")
    if method == "without_layer_adaptive_budget":
        logits = torch.stack(
            [
                torch.as_tensor(layers[f"{layer}.full_ratio_logit"]).reshape(())
                for layer in range(24)
            ]
        )
        if not torch.allclose(torch.sigmoid(logits), torch.full_like(logits, 0.7)):
            raise ValueError("w/o Layer-Adaptive Budget is not fixed at 0.7")
    if expected_stage == 2:
        stage1 = checkpoint.get("stage1")
        if not isinstance(stage1, Mapping):
            raise ValueError("Stage-2 checkpoint is missing Stage-1 identity")
        if stage1.get("dataset_sha256") != DATASET_SHA256:
            raise ValueError("Stage 1 and Stage 2 dataset identities differ")
        if stage1.get("protocol") != STAGE1_PROTOCOL:
            raise ValueError("Stage-2 checkpoint does not reference a valid Stage 1")
    return {
        "format": "release_v3",
        "method": method,
        "replacement": REPLACEMENTS[method],
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "protocol": expected_protocol,
        "stage": expected_stage,
        "modality": "vision",
        "training_seed": seed,
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "dataset_sha256": DATASET_SHA256,
        "target_ratio": TARGET_RATIO,
        "levels": list(CAPACITY_FACTORS),
        "checkpoint_step": step,
    }


def inspect_checkpoint(path: str | Path, method: str) -> dict[str, Any]:
    checkpoint_path = Path(path)
    metadata = checkpoint_metadata(torch_load(checkpoint_path), method)
    metadata["checkpoint"] = str(checkpoint_path.resolve())
    return metadata


def inspect_main_checkpoint(path: str | Path, seed: int) -> dict[str, Any]:
    checkpoint_path = Path(path)
    metadata = main_checkpoint_metadata(torch_load(checkpoint_path))
    if metadata["target_ratio"] != TARGET_RATIO:
        raise ValueError("this study requires the p=0.7 visual main checkpoint")
    if metadata["training_seed"] != int(seed):
        raise ValueError(
            f"main checkpoint seed={metadata['training_seed']}; requested seed={seed}"
        )
    metadata["checkpoint"] = str(checkpoint_path.resolve())
    return metadata

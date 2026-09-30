"""Strict release-v3 checkpoint identities for SigLIP2 transfer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ...common.two_stage import STAGE1_PROTOCOL, protocol_for_stage

from .protocol import (
    CAPACITY_FACTORS,
    DATA_SEED,
    POOL_SIZE,
    SEEDS,
    SigLIP2Spec,
    expected_dataset_sha256,
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
    checkpoint: Mapping[str, Any],
    spec: SigLIP2Spec,
    modality: str,
    stage: int,
) -> dict[str, Any]:
    """Validate one weight against the exact Table-6 model and 500k pool."""

    if modality not in ("vision", "text") or stage not in (1, 2):
        raise ValueError("invalid checkpoint modality or stage")
    if checkpoint.get("method") != "sparmoe_vl_siglip2_transfer":
        raise ValueError("checkpoint is not a paper-protocol SigLIP2 transfer run")
    if checkpoint.get("format_version") != 3:
        raise ValueError("checkpoint must use release format_version=3")
    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, Mapping):
        raise ValueError("checkpoint is missing its training arguments")
    dataset = checkpoint.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("release checkpoint is missing dataset identity")
    metadata = {
        "format": "release_v3",
        "protocol": checkpoint.get("protocol"),
        "stage": checkpoint.get("stage"),
        "modality": checkpoint.get("modality"),
        "model_key": checkpoint.get("model_key"),
        "model_name": checkpoint.get("model_name"),
        "training_seed": checkpoint.get("training_seed"),
        "data_seed": dataset.get("data_seed"),
        "pool_size": dataset.get("samples"),
        "dataset_sha256": dataset.get("ordered_sha256"),
        "target_ratio": checkpoint.get("target_ratio"),
        "levels": checkpoint.get("capacity_factors"),
        "checkpoint_step": checkpoint.get("step"),
    }

    expected_protocol = protocol_for_stage(stage)
    expected = {
        "protocol": expected_protocol,
        "stage": stage,
        "modality": modality,
        "model_key": spec.key,
        "model_name": spec.model_name,
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "dataset_sha256": expected_dataset_sha256(modality),
    }
    for key, wanted in expected.items():
        if metadata.get(key) != wanted:
            raise ValueError(f"checkpoint {key}={metadata.get(key)!r}; expected {wanted!r}")
    recorded_layers = checkpoint.get("sparse_layers")
    if recorded_layers is not None and recorded_layers != list(range(spec.num_layers)):
        raise ValueError("checkpoint sparse layers differ from the full-tower protocol")
    if stage == 1 and checkpoint.get("n_experts", 4) != 4:
        raise ValueError("Stage 1 must learn four nested capacity levels")
    seed = int(metadata["training_seed"])
    if seed not in SEEDS:
        raise ValueError(f"checkpoint training seed {seed} is not one of {SEEDS}")
    target = spec.target_ratio(modality)
    if abs(float(metadata["target_ratio"]) - target) > 1e-8:
        raise ValueError(
            f"checkpoint target ratio={metadata['target_ratio']}; expected {target}"
        )
    if tuple(float(value) for value in metadata["levels"]) != CAPACITY_FACTORS:
        raise ValueError("checkpoint capacity factors differ from the paper protocol")

    expected_args = {
        "steps": 5_000,
        "batch_size": spec.batch_size(modality, stage),
        "max_samples": POOL_SIZE,
        "num_workers": 8 if modality == "vision" else 4,
        "data_seed": DATA_SEED,
        "seed": seed,
        "weight_decay": 0.05,
    }
    for key, wanted in expected_args.items():
        if train_args.get(key) != wanted:
            raise ValueError(
                f"checkpoint train_args.{key}={train_args.get(key)!r}; expected {wanted!r}"
            )
    accumulation = train_args.get("grad_accumulation", 1)
    if modality == "text" and accumulation != spec.text_grad_accumulation:
        raise ValueError("checkpoint gradient accumulation differs from the paper run")
    learning_rate = train_args.get("learning_rate", train_args.get("lr"))
    expected_lr = 1e-3 if stage == 1 else 3e-4
    if learning_rate is None or abs(float(learning_rate) - expected_lr) > 1e-12:
        raise ValueError(f"checkpoint learning rate={learning_rate!r}; expected {expected_lr}")
    temperature = train_args.get("temperature", train_args.get("tau", 0.4))
    if abs(float(temperature) - 0.4) > 1e-12:
        raise ValueError(f"checkpoint temperature={temperature!r}; expected 0.4")
    if stage == 2:
        stage1 = checkpoint.get("stage1")
        if not isinstance(stage1, Mapping):
            raise ValueError("Stage-2 checkpoint is missing Stage-1 identity")
        if stage1.get("protocol") != STAGE1_PROTOCOL:
            raise ValueError("Stage-2 checkpoint does not reference a valid Stage 1")
        if stage1.get("dataset_sha256") != metadata["dataset_sha256"]:
            raise ValueError("Stage 1 and Stage 2 ordered training pools differ")
        if int(stage1.get("training_seed", -1)) != seed:
            raise ValueError("Stage 1 and Stage 2 training seeds differ")
    metadata.update(
        training_seed=seed,
        data_seed=int(metadata["data_seed"]),
        pool_size=int(metadata["pool_size"]),
        target_ratio=float(metadata["target_ratio"]),
        checkpoint_step=int(metadata["checkpoint_step"]),
        levels=list(CAPACITY_FACTORS),
    )
    return metadata


def inspect_checkpoint(
    path: str | Path,
    spec: SigLIP2Spec,
    modality: str,
    stage: int,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    metadata = checkpoint_metadata(torch_load(checkpoint_path), spec, modality, stage)
    metadata["checkpoint"] = str(checkpoint_path.resolve())
    return metadata

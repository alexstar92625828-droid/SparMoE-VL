"""Immutable protocol and checkpoint validation for the LLaVA transfer study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from sparmoe_vl.common.two_stage import protocol_for_stage
from sparmoe_vl.paths import repository_root, workspace_root


STUDY_NAME = "llava_transfer"
MODEL_KEY = "clip_vit_l14_336_llava"
MODEL_NAME = "openai/clip-vit-large-patch14-336"
LLAVA_NAME = "llava-v1.5-7b"
SEEDS = (42, 123, 2026)
DATA_SEED = 42
POOL_SIZE = 500_000
TARGET_RATIO = 0.7
CAPACITY_FACTORS = (0.7, 0.8, 0.9, 1.0)
TRAINING_DATASET_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"

PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
DEFAULT_ANNOTATIONS = RESEARCH_ROOT / "ShareGPT4V" / "annotations" / "sharegpt4v_1246k.json"
DEFAULT_IMAGE_ROOT = RESEARCH_ROOT / "ShareGPT4V" / "images"
DEFAULT_CLIP = RESEARCH_ROOT / "models" / "clip-vit-large-patch14-336"
DEFAULT_LLAVA = RESEARCH_ROOT / "models" / "llava-v1.5-7b-hf-safetensors"
DEFAULT_TOKENIZER = RESEARCH_ROOT / "models" / "llava-v1.5-7b"
DEFAULT_EVAL_ROOT = RESEARCH_ROOT / "data" / "eval"

STAGE_SETTINGS = {
    1: {
        "steps": 5_000,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 0.05,
        "temperature": 0.4,
        "num_workers": 8,
        "gradient_clip_norm": 1.0,
        "best_budget_loss": 0.01,
        "log_every": 100,
        "save_every": 500,
        "loss_weights": {
            "hidden_alignment": 100.0,
            "pooled_alignment": 100.0,
            "global_budget": 50.0,
            "nested_capacity_separation": 1.0,
        },
    },
    2: {
        "steps": 5_000,
        "batch_size": 8,
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "temperature": 0.4,
        "router_warmup": 1_000,
        "random_evals": 3,
        "num_workers": 8,
        "gradient_clip_norm": 1.0,
        "log_every": 100,
        "save_every": 500,
        "loss_weights": {
            "hidden_alignment": 100.0,
            "pooled_alignment": 100.0,
            "routing": 1.0,
        },
    },
}

BENCHMARK_COUNTS = {
    "pope": {
        "random": 2_910,
        "popular": 3_000,
        "adversarial": 3_000,
        "total_questions": 8_910,
        "unique_images": 500,
    },
    "mme_p": {
        "categories": 10,
        "groups": 1_057,
        "questions": 2_114,
    },
    "gqa": {"questions": 12_578},
    "vqav2": {"questions": 214_354, "annotations": 214_354},
}

BENCHMARK_IDENTITIES = {
    "pope_random_sha256": ("c16b65d2df6d70bb0814c8a4375833d39cdb09928fd936d3fc546491efa37f35"),
    "pope_popular_sha256": ("5b16653023bf4f64d57eb0d168259de6627f5cde7538b85ebed3cc33034aa6cc"),
    "pope_adversarial_sha256": (
        "1d8f0906a9b50c05640f6f23abfac97523be2a1bed28f7eadc63b811699b2d26"
    ),
    "mme_p_annotations_sha256": (
        "40d3f3a947dc9f299bc6673a96a268dd9c6613308681a6fb09c2d12634240644"
    ),
    "gqa_questions_sha256": (
        "14039069c0b3c797c7aa9bcd5f4c2aa4b5976e02c0b6773e1a584d942a03a318"
    ),
    "vqav2_questions_sha256": (
        "f34d9c9909dea76700d361baddd63dd47b4494945097459b542d9fe840244811"
    ),
    "vqav2_annotations_sha256": (
        "0564650e712b34c246e4ad142894692bf2e45dee8e9cc6ebb2cedbe85731c4bf"
    ),
}

MME_PERCEPTION_CATEGORIES = (
    "existence",
    "count",
    "position",
    "color",
    "OCR",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "posters",
)

MME_CATEGORY_COUNTS = {
    "existence": {"groups": 30, "questions": 60},
    "count": {"groups": 30, "questions": 60},
    "position": {"groups": 30, "questions": 60},
    "color": {"groups": 30, "questions": 60},
    "OCR": {"groups": 20, "questions": 40},
    "celebrity": {"groups": 170, "questions": 340},
    "scene": {"groups": 200, "questions": 400},
    "landmark": {"groups": 200, "questions": 400},
    "artwork": {"groups": 200, "questions": 400},
    "posters": {"groups": 147, "questions": 294},
}

DENSE_VISUAL_FFN_MACS_G = 116.165443584


def torch_load(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {checkpoint_path}")
    return payload


def _release_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    dataset = checkpoint.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("release checkpoint does not contain its training-data identity")
    return {
        "format": "release_v3",
        "stage": checkpoint.get("stage"),
        "protocol": checkpoint.get("protocol"),
        "training_seed": checkpoint.get("training_seed"),
        "data_seed": dataset.get("data_seed"),
        "pool_size": dataset.get("samples"),
        "dataset_sha256": dataset.get("ordered_sha256"),
        "target_ratio": checkpoint.get("target_ratio"),
        "capacity_factors": checkpoint.get("capacity_factors"),
        "checkpoint_step": checkpoint.get("step"),
        "model_key": checkpoint.get("model_key"),
        "model_name": checkpoint.get("model_name"),
    }


def checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    expected_stage: int | None = None,
) -> dict[str, Any]:
    """Validate a paper-protocol CLIP-336 checkpoint."""

    if checkpoint.get("method") != "sparmoe_vl_two_stage":
        raise ValueError("checkpoint is not a paper-protocol SparMoE-VL run")
    if checkpoint.get("format_version") != 3:
        raise ValueError("checkpoint must use release format_version=3")
    metadata = _release_metadata(checkpoint)
    if checkpoint.get("modality") != "vision":
        raise ValueError("release checkpoint must target the vision tower")
    if expected_stage is not None and metadata["stage"] != expected_stage:
        raise ValueError(
            f"checkpoint stage={metadata['stage']}; expected Stage {expected_stage}"
        )
    integer_checks = {
        "training_seed": SEEDS,
        "data_seed": (DATA_SEED,),
        "pool_size": (POOL_SIZE,),
    }
    for key, choices in integer_checks.items():
        value = metadata.get(key)
        if value is None or int(value) not in choices:
            raise ValueError(f"checkpoint {key}={value!r}; expected one of {choices}")
    exact_checks = {
        "protocol": protocol_for_stage(int(metadata["stage"])),
        "dataset_sha256": TRAINING_DATASET_SHA256,
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
    }
    for key, wanted in exact_checks.items():
        if metadata.get(key) != wanted:
            raise ValueError(f"checkpoint {key}={metadata.get(key)!r}; expected {wanted!r}")
    if abs(float(metadata["target_ratio"]) - TARGET_RATIO) > 1e-8:
        raise ValueError("LLaVA transfer requires the p=0.7 CLIP-336 checkpoint")
    if tuple(float(value) for value in metadata["capacity_factors"]) != CAPACITY_FACTORS:
        raise ValueError("checkpoint capacity factors differ from the paper protocol")
    metadata.update(
        training_seed=int(metadata["training_seed"]),
        data_seed=int(metadata["data_seed"]),
        pool_size=int(metadata["pool_size"]),
        checkpoint_step=int(metadata["checkpoint_step"]),
        target_ratio=float(metadata["target_ratio"]),
        capacity_factors=list(CAPACITY_FACTORS),
    )
    return metadata


def inspect_checkpoint(
    path: str | Path,
    *,
    expected_stage: int | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    metadata = checkpoint_metadata(
        torch_load(checkpoint_path),
        expected_stage=expected_stage,
    )
    metadata["checkpoint"] = str(checkpoint_path.resolve())
    return metadata

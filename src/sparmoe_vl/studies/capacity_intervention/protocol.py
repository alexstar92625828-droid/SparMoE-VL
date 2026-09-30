"""Immutable protocol recovered from the result-generating Table-7 run."""

from __future__ import annotations

from pathlib import Path

from sparmoe_vl.common.two_stage import protocol_for_stage
from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
STUDY_NAME = "capacity_intervention"
MODEL_NAME = "ViT-L-14"
MODEL_KEY = "clip_vit_l14_vision_n8"
SEEDS = (42, 123, 2026)
DATA_SEED = 42
POOL_SIZE = 500_000
TRAIN_STEPS = 5_000
TRAIN_BATCH_SIZE = 24
EVAL_IMAGES = 5_000
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 8
TARGET_RATIO = 0.7
RECOVERY_COSINE_THRESHOLD = 0.95
CAPACITY_FACTORS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
VISION_DATASET_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"
COCO_ANNOTATIONS_SHA256 = "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
TRAIN_ANNOTATIONS = RESEARCH_ROOT / "ShareGPT4V" / "annotations" / "sharegpt4v_1246k.json"
TRAIN_IMAGES = RESEARCH_ROOT / "ShareGPT4V" / "images"
COCO_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "coco" / "annotations" / "captions_val2017.json"
)
COCO_IMAGES = RESEARCH_ROOT / "data" / "eval" / "coco" / "val2017"
CHECKPOINT_ROOT = (
    PROJECT_ROOT / "checkpoints" / "sparmoe_vl_clip_vitl14" / "studies" / STUDY_NAME
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sparmoe_vl_clip_vitl14" / "studies" / STUDY_NAME
ALLOCATIONS = (
    "Self",
    "Uniform",
    "Shuffled",
    "Layer-Uniform",
    "Layer-Shuffled",
)
TOKEN_ALLOCATIONS = ("Self", "Uniform", "Shuffled")
LAYER_ALLOCATIONS = ("Layer-Self", "Layer-Uniform", "Layer-Shuffled")
METRICS = (
    "Activated FFN MACs (G)",
    "Cosine ↑",
    "NRE ↓",
    "Recovery Rate ↑",
)


def stage_checkpoint(seed: int, stage: int) -> Path:
    if seed not in SEEDS or stage not in (1, 2):
        raise ValueError("invalid seed or training stage")
    return CHECKPOINT_ROOT / f"seed_{seed}" / f"stage{stage}" / "best.pt"


def training_manifest(seed: int, stage: int) -> dict:
    """Return the fixed two-stage run identity for one training seed."""

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    return {
        "study": STUDY_NAME,
        "paper_scope": "Table 7",
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "modality": "vision",
        "training_protocol": protocol_for_stage(stage),
        "stage": stage,
        "training_seed": seed,
        "data_seed": DATA_SEED,
        "samples": POOL_SIZE,
        "expected_ordered_sha256": VISION_DATASET_SHA256,
        "steps": TRAIN_STEPS,
        "batch_size": TRAIN_BATCH_SIZE,
        "target_ratio": TARGET_RATIO,
        "capacity_factors": list(CAPACITY_FACTORS),
        "learning_rate": 1e-3 if stage == 1 else 3e-4,
        "weight_decay": 0.05,
        "router_temperature": 0.4,
        "objective": (
            ["representation_preservation", "global_budget_p", "nested_capacity_separation"]
            if stage == 1
            else [
                "representation_preservation",
                "token_routing",
            ]
        ),
        "trainable": (
            ["spg", "reference_capacities", "nested_channel_subspaces"]
            if stage == 1
            else ["token_router"]
        ),
        "frozen": (
            ["dense_backbone", "token_router"]
            if stage == 1
            else [
                "dense_backbone",
                "spg",
                "reference_capacities",
                "nested_channel_subspaces",
            ]
        ),
        "router_warmup_steps": 1_000 if stage == 2 else None,
    }

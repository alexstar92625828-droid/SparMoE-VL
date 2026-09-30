"""Immutable protocol recovered from the result-generating granularity study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sparmoe_vl.common.two_stage import STAGE2_PROTOCOL, TWO_STAGE_PROTOCOL
from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
STUDY_NAME = "routing_granularity_geometry_preservation"
PROTOCOL = "capacity_matched_routing_granularity_seed42_v1"
PAPER_SCOPE = "Figure 6"
MODEL_NAME = "ViT-L-14"
MODEL_KEY = "clip_vit_l14_vision"
MODALITY = "vision"
RUN_SEED = 42
DATA_SEED = 42
TRAINING_POOL_SIZE = 500_000
TRAINING_POOL_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"
TRAIN_STEPS = 5_000
STAGE1_BATCH_SIZE = 32
STAGE2_BATCH_SIZE = 24
NUM_WORKERS = 8
TARGET_RATIO = 0.7
EXPERT_COUNTS = (4, 6, 8, 10)
CAPACITY_FACTORS = {
    4: (0.7, 0.8, 0.9, 1.0),
    6: (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    8: (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    10: (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
}
TRAINING_PROTOCOLS = {count: STAGE2_PROTOCOL for count in EXPERT_COUNTS}

PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
PRETRAINED_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
TRAIN_ANNOTATIONS = RESEARCH_ROOT / "ShareGPT4V" / "annotations" / "sharegpt4v_1246k.json"
TRAIN_IMAGES = RESEARCH_ROOT / "ShareGPT4V" / "images"
COCO_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "coco" / "annotations" / "captions_val2017.json"
)
COCO_ANNOTATIONS_SHA256 = "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
COCO_IMAGES = RESEARCH_ROOT / "data" / "eval" / "coco" / "val2017"
COCO_IMAGE_COUNT = 5_000
COCO_CAPTION_COUNT = 25_014
COCO_IMAGE_ORDER_SHA256 = "7c80db1cddca30f33714d8c2bcfba14f77cf99c63240476087f4e011d995e488"
COCO_CAPTION_ORDER_SHA256 = "1940704f11cbb20fc751ff751cdd7c1eaf4adcad3e189efe6a69cdfabd3b5a48"
COCO_CAPTION_IMAGE_INDEX_SHA256 = (
    "8fdf80ac08536b3f88d6a2e43219a18ba97ee226455251ea8548424cec961964"
)
IMAGE_BATCH_SIZE = 64
TEXT_BATCH_SIZE = 256

VISION_LAYERS = 24
PATCH_TOKENS = 256
MODEL_DIM = 1_024
FFN_DIM = 4_096
DENSE_FFN_MACS_G = 51.740934144
DENSE_TOTAL_MACS_G = 81.012768768

CHECKPOINT_ROOT = (
    PROJECT_ROOT / "checkpoints" / "sparmoe_vl_clip_vitl14" / "studies" / STUDY_NAME
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figures" / STUDY_NAME


def capacity_factors(expert_count: int) -> tuple[float, ...]:
    """Return the paper's registered nested capacity factors for one N."""

    try:
        return CAPACITY_FACTORS[int(expert_count)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"expert_count must be one of {EXPERT_COUNTS}") from error


def training_protocol(expert_count: int) -> str:
    try:
        return TRAINING_PROTOCOLS[int(expert_count)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"expert_count must be one of {EXPERT_COUNTS}") from error


def checkpoint_path(expert_count: int) -> Path:
    capacity_factors(expert_count)
    return CHECKPOINT_ROOT / f"n{expert_count}" / "stage2" / "best.pt"


def stage1_checkpoint_path(expert_count: int) -> Path:
    capacity_factors(expert_count)
    return CHECKPOINT_ROOT / f"n{expert_count}" / "stage1" / "best.pt"


def protocol_manifest() -> dict[str, Any]:
    """Return only registered protocol values, never measured outcomes."""

    return {
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "modality": MODALITY,
        "run_seed": RUN_SEED,
        "training_data": {
            "source": "ShareGPT4V",
            "data_seed": DATA_SEED,
            "samples": TRAINING_POOL_SIZE,
            "ordered_sha256": TRAINING_POOL_SHA256,
        },
        "expert_counts": list(EXPERT_COUNTS),
        "capacity_factors": {
            str(count): list(capacity_factors(count)) for count in EXPERT_COUNTS
        },
        "training_protocols": {str(count): training_protocol(count) for count in EXPERT_COUNTS},
        "training": {
            "two_stage_protocol": TWO_STAGE_PROTOCOL,
            "steps": TRAIN_STEPS,
            "stage1_batch_size": STAGE1_BATCH_SIZE,
            "stage2_batch_size": STAGE2_BATCH_SIZE,
            "target_ratio": TARGET_RATIO,
            "stage1_learning_rate": 1e-3,
            "stage2_learning_rate": 3e-4,
            "weight_decay": 0.05,
            "router_temperature": 0.4,
            "router_warmup_steps": 1_000,
            "stage1_objective": [
                "representation_preservation",
                "global_budget_p",
                "nested_capacity_separation",
            ],
            "stage2_objective": [
                "representation_preservation",
                "token_routing",
            ],
            "stage2_frozen": [
                "backbone",
                "spg",
                "reference_capacities",
                "nested_channel_subspaces",
            ],
            "stage2_trainable": ["token_router"],
        },
        "evaluation": {
            "dataset": "COCO-val2017",
            "images": COCO_IMAGE_COUNT,
            "captions": COCO_CAPTION_COUNT,
            "image_batch_size": IMAGE_BATCH_SIZE,
            "text_batch_size": TEXT_BATCH_SIZE,
            "routing": ["learned", "capacity_matched_shuffle"],
            "control_invariant": ("exact expert-count multiset per layer and inference batch"),
        },
    }

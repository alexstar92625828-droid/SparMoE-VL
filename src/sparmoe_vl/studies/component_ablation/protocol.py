"""Immutable protocol recovered from the result-generating Table-8 runs."""

from __future__ import annotations

from pathlib import Path

from sparmoe_vl.common.two_stage import protocol_for_stage
from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
STUDY_NAME = "component_ablation"
MODEL_NAME = "ViT-L-14"
MODEL_KEY = "clip_vit_l14_vision_component_ablation"
DATA_SEED = 42
POOL_SIZE = 500_000
DATASET_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"
TARGET_RATIO = 0.7
CAPACITY_FACTORS = (0.7, 0.8, 0.9, 1.0)
TRAIN_STEPS = 5_000
TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 128
NUM_WORKERS = 8
STAGE1_LEARNING_RATE = 1e-3
STAGE2_LEARNING_RATE = 3e-4
MASK_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
ROUTER_TEMPERATURE = 0.4
ROUTER_WARMUP = 1_000
BEST_BUDGET_LOSS = 0.01
LOG_EVERY = 100
EVALUATION_SEED = 42

TRAINED_METHODS = (
    "without_spg",
    "without_layer_adaptive_budget",
    "without_geometry_preservation",
)
METHODS = (
    "without_spg",
    "without_token_router",
    "without_layer_adaptive_budget",
    "without_geometry_preservation",
    "sparmoe_vl",
)
METHOD_LABELS = {
    "without_spg": "w/o SPG",
    "without_token_router": "w/o Token Router",
    "without_layer_adaptive_budget": "w/o Layer-Adaptive Budget",
    "without_geometry_preservation": "w/o Geometry Preservation",
    "sparmoe_vl": "SparMoE-VL",
}
COMPONENT_FLAGS = {
    "without_spg": ("×", "✓", "✓", "✓"),
    "without_token_router": ("✓", "×", "✓", "✓"),
    "without_layer_adaptive_budget": ("✓", "✓", "×", "✓"),
    "without_geometry_preservation": ("✓", "✓", "✓", "×"),
    "sparmoe_vl": ("✓", "✓", "✓", "✓"),
}
REPLACEMENTS = {
    "without_spg": "per_layer_per_expert_independent_learnable_soft_masks",
    "without_token_router": "uniform_random_per_patch_token",
    "without_layer_adaptive_budget": "fixed_full_ratio_0.7_in_every_layer",
    "without_geometry_preservation": "clip_image_text_contrastive_supervision",
    "sparmoe_vl": "none",
}

# The geometry-preservation ablation uses its registered third seed; the other
# rows use the standard main-experiment seed set.
STANDARD_SEEDS = (42, 123, 2026)
PAPER_SEEDS = {
    "without_spg": STANDARD_SEEDS,
    "without_token_router": STANDARD_SEEDS,
    "without_layer_adaptive_budget": STANDARD_SEEDS,
    "without_geometry_preservation": (42, 123, 3407),
    "sparmoe_vl": STANDARD_SEEDS,
}
ALL_TRAINING_SEEDS = (42, 123, 2026, 3407)

METRICS = (
    "ffn_macs_v_g",
    "coco_i2t_r1",
    "coco_t2i_r1",
    "flickr_i2t_r1",
    "flickr_t2i_r1",
    "retention_percent",
)
EVALUATION_COUNTS = {
    "coco": {"images": 5_000, "texts": 25_014},
    "flickr30k": {"images": 1_000, "texts": 5_000},
}
EVALUATION_SHA256 = {
    "coco": "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9",
    "flickr30k": "395990db603ab8bafd5c7ab2746b22058bb1e75b78b3eb56ad755931364ac137",
}

PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
TRAIN_ANNOTATIONS = RESEARCH_ROOT / "ShareGPT4V" / "annotations" / "sharegpt4v_1246k.json"
TRAIN_IMAGES = RESEARCH_ROOT / "ShareGPT4V" / "images"
COCO_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "coco" / "annotations" / "captions_val2017.json"
)
COCO_IMAGES = RESEARCH_ROOT / "data" / "eval" / "coco" / "val2017"
FLICKR_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "flickr30k" / "flickr_annotations_30k.csv"
)
FLICKR_IMAGES = RESEARCH_ROOT / "data" / "eval" / "flickr30k" / "flickr30k-images"
CHECKPOINT_ROOT = (
    PROJECT_ROOT / "checkpoints" / "sparmoe_vl_clip_vitl14" / "studies" / STUDY_NAME
)
MAIN_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "main" / "vision"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sparmoe_vl_clip_vitl14" / "studies" / STUDY_NAME


def validate_method_seed(method: str, seed: int) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown Table-8 method: {method}")
    if int(seed) not in PAPER_SEEDS[method]:
        raise ValueError(f"method={method} uses paper seeds {PAPER_SEEDS[method]}, got {seed}")


def stage_checkpoint(method: str, seed: int, stage: int) -> Path:
    validate_method_seed(method, seed)
    if method not in TRAINED_METHODS or stage not in (1, 2):
        raise ValueError("invalid trained method or stage")
    return CHECKPOINT_ROOT / method / f"seed_{seed}" / f"stage{stage}" / "best.pt"


def training_manifest(method: str, seed: int, stage: int) -> dict:
    if method not in TRAINED_METHODS:
        raise ValueError(f"{method} reuses main weights and has no ablation training run")
    validate_method_seed(method, seed)
    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    return {
        "study": STUDY_NAME,
        "paper_scope": "Table 8",
        "method": method,
        "label": METHOD_LABELS[method],
        "replacement": REPLACEMENTS[method],
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "modality": "vision",
        "training_protocol": protocol_for_stage(stage),
        "stage": stage,
        "training_seed": int(seed),
        "data_seed": DATA_SEED,
        "samples": POOL_SIZE,
        "expected_ordered_sha256": DATASET_SHA256,
        "steps": TRAIN_STEPS,
        "batch_size": TRAIN_BATCH_SIZE,
        "target_ratio": TARGET_RATIO,
        "capacity_factors": list(CAPACITY_FACTORS),
        "learning_rate": (STAGE1_LEARNING_RATE if stage == 1 else STAGE2_LEARNING_RATE),
        "mask_learning_rate": (
            MASK_LEARNING_RATE if method == "without_spg" and stage == 1 else None
        ),
        "weight_decay": WEIGHT_DECAY,
        "router_temperature": ROUTER_TEMPERATURE,
        "router_warmup_steps": ROUTER_WARMUP if stage == 2 else None,
        "objective": (
            ["representation_objective", "global_budget_p", "nested_capacity_separation"]
            if stage == 1
            else [
                "representation_objective",
                "token_routing",
            ]
        ),
        "trainable": (
            ["spg_or_registered_replacement", "reference_capacities"]
            if stage == 1
            else ["token_router"]
        ),
        "frozen": (
            ["dense_backbone", "token_router"]
            if stage == 1
            else [
                "dense_backbone",
                "spg_or_registered_replacement",
                "reference_capacities",
                "nested_channel_subspaces",
            ]
        ),
    }

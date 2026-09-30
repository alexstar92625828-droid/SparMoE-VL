"""Immutable protocol for the layer-wise representation consistency study."""

from __future__ import annotations

from typing import Any

from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
STUDY_NAME = "layerwise_representation_consistency"
PROTOCOL = "clip_vitl14_layerwise_representation_consistency_seed42_v1"
PAPER_SCOPE = "Figure 7"

MODEL_NAME = "ViT-L-14"
MODEL_KEY = "clip_vit_l14_vision"
RUN_SEED = 42
DATA_SEED = 42
TRAINING_POOL_SIZE = 500_000
TRAINING_POOL_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"
TARGET_RATIO = 0.7
CAPACITY_FACTORS = (0.7, 0.8, 0.9, 1.0)
ROUTING_MODE = "learned_argmax"

NUM_LAYERS = 24
MODEL_DIM = 1_024
TOKENS_PER_IMAGE = 257
PATCHES_PER_IMAGE = TOKENS_PER_IMAGE - 1
COCO_IMAGES_TOTAL = 5_000
BATCH_SIZE = 4
NUM_WORKERS = 0
CPU_THREADS = 32
CHECKPOINT_EVERY = 5
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
HISTOGRAM_RANGE = (-0.30, 1.00)
HISTOGRAM_BINS = 156

PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
PRETRAINED_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "sparmoe_vl_clip_vitl14"
    / "vision"
    / "seed_42"
    / "stage2"
    / "best.pt"
)
HISTORICAL_CHECKPOINT_SHA256 = (
    "d07652461df13d4189c524c10631bcde69a51f242d4b33dd214c46b0eef38450"
)
COCO_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "coco" / "annotations" / "captions_val2017.json"
)
COCO_ANNOTATIONS_SHA256 = "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
COCO_IMAGES = RESEARCH_ROOT / "data" / "eval" / "coco" / "val2017"
COCO_MANIFEST_SHA256 = "09b3bbeda289610ec9fd4e4b5e6da32ec04f98a9a4111e99790de863be0f8f9e"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figures" / STUDY_NAME


def protocol_manifest() -> dict[str, Any]:
    """Return protocol facts only; measured values never live in source files."""

    return {
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "checkpoint_role": "vision_main_stage2_seed42",
        "run_seed": RUN_SEED,
        "training_data_seed": DATA_SEED,
        "training_pool_size": TRAINING_POOL_SIZE,
        "training_pool_sha256": TRAINING_POOL_SHA256,
        "target_ratio": TARGET_RATIO,
        "capacity_factors": list(CAPACITY_FACTORS),
        "routing": ROUTING_MODE,
        "representation_point": "post_transformer_block",
        "dataset": "COCO-val2017",
        "images": COCO_IMAGES_TOTAL,
        "image_order": "ascending_file_name",
        "num_layers": NUM_LAYERS,
        "tokens_per_image": TOKENS_PER_IMAGE,
        "patches_per_image": PATCHES_PER_IMAGE,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "metrics": {
            "local": "full_patch_token_cosine_distribution",
            "global": ["cls_token_cosine", "centered_linear_cka_over_cls"],
        },
        "cka_estimator": "exact_centered_linear_cka",
        "tf32": False,
    }

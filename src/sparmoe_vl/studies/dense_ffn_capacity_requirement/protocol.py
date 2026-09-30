"""Fixed protocol for the frozen Dense CLIP analysis in Figure 3."""

from __future__ import annotations

from typing import Any

from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
STUDY_NAME = "dense_ffn_capacity_requirement"
PROTOCOL = "dense_only_required_capacity_seed42_v1"
PAPER_SCOPE = "Figure 3"
MODEL_NAME = "ViT-L-14"
MODEL_KEY = "clip_vit_l14_dense_vision"
SEED = 42
NUM_LAYERS = 24
FFN_DIM = 4_096
PATCH_TOKENS = 256
COCO_IMAGES_TOTAL = 5_000
CALIBRATION_IMAGES = 1_000
EVALUATION_IMAGES = 4_000
BATCH_SIZE = 16
CAPACITY_LEVELS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
COSINE_THRESHOLD = 0.95
NRE_THRESHOLD = 0.20
CHANNEL_RANKING = "mean_squared_gelu_activation_times_squared_c_proj_column_norm"
COMPARISON = "local_dense_ffn_output_excluding_cls_token"
SPLIT_METHOD = "torch_randperm_over_canonical_coco_image_order"

PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
PRETRAINED_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
COCO_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "coco" / "annotations" / "captions_val2017.json"
)
COCO_ANNOTATIONS_SHA256 = "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
COCO_IMAGES = RESEARCH_ROOT / "data" / "eval" / "coco" / "val2017"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figures" / STUDY_NAME


def protocol_manifest() -> dict[str, Any]:
    return {
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "backbone": "frozen_dense_clip",
        "seed": SEED,
        "dataset": "COCO-val2017",
        "total_images": COCO_IMAGES_TOTAL,
        "calibration_images": CALIBRATION_IMAGES,
        "evaluation_images": EVALUATION_IMAGES,
        "split_method": SPLIT_METHOD,
        "num_layers": NUM_LAYERS,
        "patch_tokens_per_image": PATCH_TOKENS,
        "ffn_dim": FFN_DIM,
        "capacity_levels": list(CAPACITY_LEVELS),
        "cosine_threshold": COSINE_THRESHOLD,
        "nre_threshold": NRE_THRESHOLD,
        "channel_ranking": CHANNEL_RANKING,
        "comparison": COMPARISON,
        "batch_size": BATCH_SIZE,
        "tf32": False,
    }

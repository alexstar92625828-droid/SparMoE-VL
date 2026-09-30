"""Immutable protocol for cross-modal similarity structure preservation."""

from __future__ import annotations

from typing import Any

from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
STUDY_NAME = "cross_modal_similarity_preservation"
PROTOCOL = "clip_vitl14_cross_modal_similarity_preservation_seed42_v1"
PAPER_SCOPE = "Figure 8"

MODEL_NAME = "ViT-L-14"
MODEL_KEY = "clip_vit_l14_vision"
RUN_SEED = 42
DATA_SEED = 42
TRAINING_POOL_SIZE = 500_000
TRAINING_POOL_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"
TARGET_RATIO = 0.7
CAPACITY_FACTORS = (0.7, 0.8, 0.9, 1.0)

IMAGE_COUNT = 5_000
CAPTION_COUNT = 25_014
PAIRWISE_SIMILARITY_COUNT = IMAGE_COUNT * CAPTION_COUNT
MODEL_DIM = 1_024
OUTPUT_DIM = 768
LAYER_COUNT = 24
TEXT_BATCH_SIZE = 256
PROJECTION_BATCH_SIZE = 512
MATRIX_BLOCK_SIZE = 128
SAMPLE_SEED = 42
SAMPLE_COUNT = 256
DISPLAY_GROUPS = 64
GROUP_SIZE = SAMPLE_COUNT // DISPLAY_GROUPS

PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
PRETRAINED_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
COCO_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "coco" / "annotations" / "captions_val2017.json"
)
COCO_ANNOTATIONS_SHA256 = "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
COCO_IMAGES = RESEARCH_ROOT / "data" / "eval" / "coco" / "val2017"
COCO_IMAGE_ORDER_SHA256 = "09b3bbeda289610ec9fd4e4b5e6da32ec04f98a9a4111e99790de863be0f8f9e"
COCO_CAPTION_ORDER_SHA256 = "3a4b987ec0b120b2391c36530bd20a71b3e9a538baf2183faec716b80b63570e"
COCO_CAPTION_IMAGE_INDEX_SHA256 = (
    "90c7a3e1df88d4f2355a9c147a0a9948fa2e6a1f6550be3093688c62da66a74d"
)
COCO_CAPTION_ANNOTATION_ID_SHA256 = (
    "4825294f9c36b54d9501e8f548ecc9fe307750b8de26e007305d3c5c1e6aa1a0"
)
COCO_FIRST_CAPTION_INDEX_SHA256 = (
    "029bb127c6392c7751e00ffb02fcaf28c0ca8d2795eb31b857b03df2108c6253"
)
UNORDERED_SAMPLE_INDEX_SHA256 = (
    "46d904ebb27ce288eafcf17929d888e536b14ec1b8c971766c6e3eb77a7e11c3"
)

LAYERWISE_STUDY = "layerwise_representation_consistency"
LAYERWISE_PROTOCOL = "clip_vitl14_layerwise_representation_consistency_seed42_v1"
LAYERWISE_OUTPUT = PROJECT_ROOT / "outputs" / "figures" / LAYERWISE_STUDY
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figures" / STUDY_NAME


def protocol_manifest() -> dict[str, Any]:
    """Return only registered protocol facts, never measured similarities."""

    return {
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "run_seed": RUN_SEED,
        "visual_checkpoint_role": "vision_main_stage2_seed42",
        "text_encoder": "frozen_dense_clip",
        "training_data": {
            "source": "ShareGPT4V",
            "data_seed": DATA_SEED,
            "samples": TRAINING_POOL_SIZE,
            "ordered_sha256": TRAINING_POOL_SHA256,
        },
        "evaluation": {
            "dataset": "COCO-val2017",
            "images": IMAGE_COUNT,
            "captions": CAPTION_COUNT,
            "pairwise_similarities": PAIRWISE_SIMILARITY_COUNT,
            "image_order": "ascending_file_name",
            "caption_order": "annotation_file_order",
            "matrix_statistics": ["pearson", "matrix_cosine", "mae"],
        },
        "image_feature_source": {
            "study": LAYERWISE_STUDY,
            "protocol": LAYERWISE_PROTOCOL,
            "layer": LAYER_COUNT,
            "representation": "projected_final_cls",
        },
        "visualization": {
            "selection": "uniform_without_replacement_independent_of_scores",
            "sample_seed": SAMPLE_SEED,
            "matched_pairs": SAMPLE_COUNT,
            "ordering": "average_linkage_dense_joint_image_text_embedding",
            "display_groups": DISPLAY_GROUPS,
            "pairs_per_group": GROUP_SIZE,
            "aggregation": "non_overlapping_arithmetic_mean",
            "shared_color_range": True,
        },
    }

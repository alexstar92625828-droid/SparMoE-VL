"""Fixed protocol for the paper's input-dependent routing visualization."""

from __future__ import annotations

from typing import Any

from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
STUDY_NAME = "input_dependent_token_routing"
PROTOCOL = "clip_vitl14_input_dependent_token_routing_seed42_v1"
PAPER_SCOPE = "Figure 5"
MODEL_NAME = "ViT-L-14"
RUN_SEED = 42
DATA_SEED = 42
TRAINING_POOL_SIZE = 500_000
CAPACITY_FACTORS = (0.7, 0.8, 0.9, 1.0)
NUM_EXPERTS = len(CAPACITY_FACTORS)
ROUTING_MODE = "learned_argmax"

VISION_TARGET_RATIO = 0.7
VISION_TRAINING_POOL_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"
VISION_LAYERS = (6, 12, 18)
PATCH_GRID_SIZE = 16
PATCHES_PER_IMAGE = PATCH_GRID_SIZE**2
OVERLAY_ALPHA = 0.62

TEXT_TARGET_RATIO = 0.6
TEXT_TRAINING_POOL_SHA256 = "c76621edddeeb546f6fa798552b2ae11eb409c63b520080d42a8a37b66f31e9d"
TEXT_LAYER = 12
TEXT_INPUT = (
    "A crowded city street scene with several people walking near storefronts, "
    "parked cars, bicycles, traffic signs, glass windows, colorful buildings, "
    "and outdoor advertisements while pedestrians move through the urban area "
    "under bright daylight, with reflections on the windows, objects arranged "
    "along the sidewalk, and multiple visual details that describe the complex "
    "relationship between people, vehicles, architecture, and public space."
)
TEXT_INPUT_SHA256 = "7e1698fc87ce138984bcb50277ea50a5b296d1a72a481ed4ed2eb572c00017ce"

PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
PRETRAINED_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
VISION_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "sparmoe_vl_clip_vitl14"
    / "vision"
    / "seed_42"
    / "stage2"
    / "best.pt"
)
TEXT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "sparmoe_vl_clip_vitl14"
    / "text"
    / "seed_42"
    / "stage2"
    / "best.pt"
)
HISTORICAL_VISION_CHECKPOINT_SHA256 = (
    "d07652461df13d4189c524c10631bcde69a51f242d4b33dd214c46b0eef38450"
)
HISTORICAL_TEXT_CHECKPOINT_SHA256 = (
    "00c56fcde5db2a30b852c77cb54546db0197268f1eef627a3e0f8dfbbf52bd17"
)

IMAGE_ROOT = RESEARCH_ROOT / "data" / "eval" / "coco" / "val2017"
IMAGE_IDENTITIES = (
    (
        "000000256407.jpg",
        "e9b63060910cbd8ff9d9f420204ec691d7e46d2034ea24857f6e4dbf1966f535",
    ),
    (
        "000000269682.jpg",
        "c2198ff1b869a69eb7a41fa81ca66e251771eab91822cb262a2dd682fa375275",
    ),
    (
        "000000515828.jpg",
        "cffa35700a8255de38930c3bf75055c83c6b09f20b949af0852f3099222b03ca",
    ),
    (
        "000000440336.jpg",
        "e5c2df473a26427ae57950acec86d1e4d3a49cdf1a18d427cd1a354465408f00",
    ),
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "figures" / STUDY_NAME


def protocol_manifest() -> dict[str, Any]:
    """Return inputs and model identities without any measured routing result."""

    return {
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "routing": ROUTING_MODE,
        "run_seed": RUN_SEED,
        "training_data_seed": DATA_SEED,
        "training_pool_size": TRAINING_POOL_SIZE,
        "capacity_factors": list(CAPACITY_FACTORS),
        "vision": {
            "checkpoint_role": "vision_main_stage2_seed42",
            "training_pool_sha256": VISION_TRAINING_POOL_SHA256,
            "target_ratio": VISION_TARGET_RATIO,
            "layers_one_based": list(VISION_LAYERS),
            "patch_grid": [PATCH_GRID_SIZE, PATCH_GRID_SIZE],
            "images": [
                {"file_name": file_name, "sha256": digest}
                for file_name, digest in IMAGE_IDENTITIES
            ],
            "overlay_alpha": OVERLAY_ALPHA,
        },
        "text": {
            "checkpoint_role": "text_main_stage2_seed42",
            "training_pool_sha256": TEXT_TRAINING_POOL_SHA256,
            "target_ratio": TEXT_TARGET_RATIO,
            "layer_one_based": TEXT_LAYER,
            "input": TEXT_INPUT,
            "input_sha256": TEXT_INPUT_SHA256,
        },
    }

"""Immutable protocol for the CLIP rows of the architecture-transfer study.

Only CLIP ViT-B/16 and ViT-B/32 belong here. CLIP ViT-L/14 supplies the
paper's main experiments and is intentionally not duplicated as an
architecture-transfer run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
SEEDS = (42, 123, 2026)
DATA_SEED = 42
POOL_SIZE = 500_000
CAPACITY_FACTORS = (0.7, 0.8, 0.9, 1.0)
VISION_TARGET_RATIO = 0.7
TEXT_TARGET_RATIO = 0.6
VISION_DATASET_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"
TEXT_DATASET_SHA256 = "c76621edddeeb546f6fa798552b2ae11eb409c63b520080d42a8a37b66f31e9d"
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


@dataclass(frozen=True)
class CLIPSpec:
    """Architecture and result-generating batch settings for one CLIP version."""

    name: str
    key: str
    display_name: str
    open_clip_name: str
    pretrained: str
    image_size: int
    patch_size: int
    num_patches: int
    num_layers: int
    vision_heads: int
    vision_dim: int
    vision_ffn_dim: int
    output_dim: int
    context_length: int
    text_heads: int
    text_dim: int
    text_ffn_dim: int
    vision_stage1_batch: int
    vision_stage2_batch: int
    text_batch: int = 256
    text_grad_accumulation: int = 1

    def batch_size(self, modality: Literal["vision", "text"], stage: int) -> int:
        if modality == "vision":
            return self.vision_stage1_batch if stage == 1 else self.vision_stage2_batch
        return self.text_batch

    def target_ratio(self, modality: Literal["vision", "text"]) -> float:
        return VISION_TARGET_RATIO if modality == "vision" else TEXT_TARGET_RATIO

    def default_pretrained(self) -> str:
        if self.pretrained == "openai":
            return self.pretrained
        return str(RESEARCH_ROOT / self.pretrained)


MODEL_SPECS = {
    "vit_b16": CLIPSpec(
        name="vit_b16",
        key="clip_vit_b16",
        display_name="CLIP ViT-B/16",
        open_clip_name="ViT-B-16",
        pretrained="openai",
        image_size=224,
        patch_size=16,
        num_patches=196,
        num_layers=12,
        vision_heads=12,
        vision_dim=768,
        vision_ffn_dim=3072,
        output_dim=512,
        context_length=77,
        text_heads=8,
        text_dim=512,
        text_ffn_dim=2048,
        vision_stage1_batch=128,
        vision_stage2_batch=96,
    ),
    "vit_b32": CLIPSpec(
        name="vit_b32",
        key="clip_vit_b32",
        display_name="CLIP ViT-B/32",
        open_clip_name="ViT-B-32",
        pretrained="models/ViT-B-32.pt",
        image_size=224,
        patch_size=32,
        num_patches=49,
        num_layers=12,
        vision_heads=12,
        vision_dim=768,
        vision_ffn_dim=3072,
        output_dim=512,
        context_length=77,
        text_heads=8,
        text_dim=512,
        text_ffn_dim=2048,
        vision_stage1_batch=256,
        vision_stage2_batch=192,
    ),
}


def get_spec(name: str) -> CLIPSpec:
    """Resolve one Table-6 CLIP version; ViT-L/14 is deliberately absent."""

    try:
        return MODEL_SPECS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown CLIP transfer version {name!r}; choose from {tuple(MODEL_SPECS)}"
        ) from error


def expected_dataset_sha256(modality: str) -> str:
    if modality == "vision":
        return VISION_DATASET_SHA256
    if modality == "text":
        return TEXT_DATASET_SHA256
    raise ValueError("modality must be 'vision' or 'text'")

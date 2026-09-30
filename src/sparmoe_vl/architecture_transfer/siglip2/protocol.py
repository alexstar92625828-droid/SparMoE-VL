"""Immutable protocol for the SigLIP2 rows of the architecture-transfer study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
TOKENIZER_JSON_SHA256 = "2b35cc56fa15e8edef437bbd3943a7bce4441d5d765d5d35b3d1770ba0d22b44"
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
class SigLIP2Spec:
    """Architecture and result-generating batch settings for one version."""

    name: str
    key: str
    display_name: str
    model_name: str
    pretrained_relpath: str
    tokenizer_relpath: str
    context_length: int
    image_size: int
    patch_size: int
    num_patches: int
    num_layers: int
    num_heads: int
    model_dim: int
    ffn_dim: int
    vision_stage1_batch: int
    vision_stage2_batch: int
    text_micro_batch: int
    text_grad_accumulation: int

    @property
    def pretrained(self) -> Path:
        return RESEARCH_ROOT / self.pretrained_relpath

    @property
    def tokenizer(self) -> Path:
        return RESEARCH_ROOT / self.tokenizer_relpath

    def batch_size(self, modality: Literal["vision", "text"], stage: int) -> int:
        if modality == "vision":
            return self.vision_stage1_batch if stage == 1 else self.vision_stage2_batch
        return self.text_micro_batch

    def grad_accumulation(self, modality: Literal["vision", "text"]) -> int:
        return 1 if modality == "vision" else self.text_grad_accumulation

    def target_ratio(self, modality: Literal["vision", "text"]) -> float:
        return VISION_TARGET_RATIO if modality == "vision" else TEXT_TARGET_RATIO


MODEL_SPECS = {
    "vit_b16": SigLIP2Spec(
        name="vit_b16",
        key="siglip2_vit_b16",
        display_name="SigLIP2 ViT-B/16",
        model_name="ViT-B-16-SigLIP2",
        pretrained_relpath=(
            "models/siglip2_b16/models--timm--ViT-B-16-SigLIP2/"
            "snapshots/eee10eff6dd8cabae2d7f379d4e8cfcd352030aa/"
            "open_clip_model.safetensors"
        ),
        tokenizer_relpath="models/siglip2_tokenizer",
        context_length=64,
        image_size=224,
        patch_size=16,
        num_patches=196,
        num_layers=12,
        num_heads=12,
        model_dim=768,
        ffn_dim=3072,
        vision_stage1_batch=128,
        vision_stage2_batch=96,
        text_micro_batch=256,
        text_grad_accumulation=1,
    ),
    "vit_l16": SigLIP2Spec(
        name="vit_l16",
        key="siglip2_vit_l16",
        display_name="SigLIP2 ViT-L/16",
        model_name="ViT-L-16-SigLIP2-256",
        pretrained_relpath="models/siglip2_l16/open_clip_model.safetensors",
        tokenizer_relpath="models/siglip2_tokenizer",
        context_length=64,
        image_size=256,
        patch_size=16,
        num_patches=256,
        num_layers=24,
        num_heads=16,
        model_dim=1024,
        ffn_dim=4096,
        vision_stage1_batch=32,
        vision_stage2_batch=24,
        text_micro_batch=32,
        text_grad_accumulation=4,
    ),
}


def get_spec(name: str) -> SigLIP2Spec:
    try:
        return MODEL_SPECS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown SigLIP2 transfer version {name!r}; choose from {tuple(MODEL_SPECS)}"
        ) from error


def expected_dataset_sha256(modality: str) -> str:
    if modality == "vision":
        return VISION_DATASET_SHA256
    if modality == "text":
        return TEXT_DATASET_SHA256
    raise ValueError("modality must be 'vision' or 'text'")

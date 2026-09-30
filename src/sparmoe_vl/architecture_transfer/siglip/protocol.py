"""Immutable protocol for the six SigLIP experiments in the paper.

The paper evaluates three SigLIP backbones with SparMoE-VL applied to either
the image tower or the text tower.  Values in this registry were recovered
from the result-generating three-seed launchers and checkpoints, not from the
older exploratory scripts that remain in the research workspace.
"""

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
class SigLIPSpec:
    """Architecture and result-generating batch settings for one backbone."""

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


MODEL_SPECS = {
    "vit_b16": SigLIPSpec(
        key="siglip1_vit_b16",
        display_name="SigLIP ViT-B/16",
        model_name="ViT-B-16-SigLIP",
        pretrained_relpath="models/siglip1_b16/open_clip_model.safetensors",
        tokenizer_relpath="models/siglip1_b16_tokenizer",
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
    "vit_l16": SigLIPSpec(
        key="siglip1_vit_l16",
        display_name="SigLIP ViT-L/16",
        model_name="ViT-L-16-SigLIP-256",
        pretrained_relpath="models/siglip1_l16/open_clip_model.safetensors",
        tokenizer_relpath="models/siglip1_l16_tokenizer",
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
    "so400m14": SigLIPSpec(
        key="siglip_vit_so400m14",
        display_name="SigLIP ViT-SO400M/14",
        model_name="ViT-SO400M-14-SigLIP",
        pretrained_relpath="models/siglip1_so400m/open_clip_model.safetensors",
        tokenizer_relpath="models/siglip1_so400m_tokenizer",
        context_length=16,
        image_size=224,
        patch_size=14,
        num_patches=256,
        num_layers=27,
        num_heads=16,
        model_dim=1152,
        ffn_dim=4304,
        vision_stage1_batch=16,
        vision_stage2_batch=12,
        text_micro_batch=128,
        text_grad_accumulation=2,
    ),
}


def get_spec(name: str) -> SigLIPSpec:
    """Resolve a public short name and reject unregistered architectures."""

    try:
        return MODEL_SPECS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown SigLIP model {name!r}; choose from {tuple(MODEL_SPECS)}"
        ) from error


def expected_training_identity(modality: str) -> dict[str, int | str]:
    if modality not in ("vision", "text"):
        raise ValueError("modality must be 'vision' or 'text'")
    return {
        "data_seed": DATA_SEED,
        "samples": POOL_SIZE,
        "ordered_sha256": (
            VISION_DATASET_SHA256 if modality == "vision" else TEXT_DATASET_SHA256
        ),
    }

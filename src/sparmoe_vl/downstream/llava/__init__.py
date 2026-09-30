"""LLaVA-v1.5 transfer experiment for the CLIP ViT-L/14-336 vision tower."""

from .protocol import (
    CAPACITY_FACTORS,
    DATA_SEED,
    MODEL_KEY,
    SEEDS,
    STUDY_NAME,
    TARGET_RATIO,
    TRAINING_DATASET_SHA256,
)

__all__ = [
    "CAPACITY_FACTORS",
    "DATA_SEED",
    "MODEL_KEY",
    "SEEDS",
    "STUDY_NAME",
    "TARGET_RATIO",
    "TRAINING_DATASET_SHA256",
]

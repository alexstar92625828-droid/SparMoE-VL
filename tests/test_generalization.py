import pytest
import torch

from sparmoe_vl.common.macs import clip_vitl14_text_routed_ffn_macs
from sparmoe_vl.studies.generalization.cache import classification_accuracy
from sparmoe_vl.studies.generalization.evaluate import parse_args
from sparmoe_vl.studies.generalization.protocol import (
    CAPACITY_FACTORS,
    DATASET_COUNTS,
    EVALUATION_IDENTITIES,
    POOL_SIZE,
    SEEDS,
    TEXT_DATASET_SHA256,
    VISION_DATASET_SHA256,
    checkpoint_metadata,
)


def release_checkpoint(modality: str) -> dict:
    return {
        "format_version": 3,
        "method": "sparmoe_vl_two_stage",
        "protocol": "frozen_spg_token_router_stage2",
        "stage": 2,
        "modality": modality,
        "model_name": "ViT-L-14",
        "step": 800,
        "training_seed": 2026,
        "target_ratio": 0.7 if modality == "vision" else 0.6,
        "capacity_factors": list(CAPACITY_FACTORS),
        "dataset": {
            "data_seed": 42,
            "samples": POOL_SIZE,
            "ordered_sha256": (
                VISION_DATASET_SHA256 if modality == "vision" else TEXT_DATASET_SHA256
            ),
        },
    }


@pytest.mark.parametrize("modality", ["vision", "text"])
def test_table4_accepts_release_main_checkpoints(modality: str) -> None:
    metadata = checkpoint_metadata(release_checkpoint(modality), modality)
    assert metadata["training_seed"] in SEEDS
    assert metadata["data_seed"] == 42
    assert metadata["pool_size"] == 500_000


def test_table4_rejects_wrong_text_training_pool() -> None:
    checkpoint = release_checkpoint("text")
    checkpoint["dataset"]["ordered_sha256"] = VISION_DATASET_SHA256
    with pytest.raises(ValueError, match="dataset_sha256"):
        checkpoint_metadata(checkpoint, "text")


def test_table4_dataset_contract_is_complete() -> None:
    assert DATASET_COUNTS == {
        "coco": {"images": 5_000, "texts": 25_014},
        "flickr30k": {"images": 1_000, "texts": 5_000},
        "cifar100": {"images": 10_000, "classes": 100},
        "imagenet1k": {"images": 50_000, "classes": 1_000},
        "food101": {"images": 25_250, "classes": 101},
    }
    assert len(EVALUATION_IDENTITIES) == 8
    assert all(len(value) == 64 for value in EVALUATION_IDENTITIES.values())


def test_text_routed_macs_keep_first_position_dense() -> None:
    ratios = torch.full((12,), 0.5, dtype=torch.float64)
    macs = clip_vitl14_text_routed_ffn_macs(ratios)
    dense_ffn = 12 * 2 * 77 * 768 * 3072 / 1e9
    sparse_ffn = 12 * (1 + 76 * 0.5) * 2 * 768 * 3072 / 1e9
    assert macs.dense_ffn_g == pytest.approx(dense_ffn)
    assert macs.sparse_ffn_g == pytest.approx(sparse_ffn)


def test_classification_accuracy_uses_all_examples() -> None:
    images = torch.eye(3)
    prototypes = torch.eye(3)
    labels = torch.tensor([0, 1, 2])
    assert classification_accuracy(images, prototypes, labels) == 100.0


def test_modality_specific_evaluator_hides_internal_modality_flag(tmp_path) -> None:
    args = parse_args(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--cache",
            str(tmp_path / "cache.pt"),
            "--output",
            str(tmp_path / "evaluation.json"),
        ],
        fixed_modality="vision",
    )
    assert args.modality == "vision"

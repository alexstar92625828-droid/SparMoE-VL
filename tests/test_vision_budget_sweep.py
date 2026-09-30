from argparse import Namespace

import pytest

from sparmoe_vl.studies.vision_budget_sweep import (
    BUDGET_POINTS,
    CAPACITY_FACTORS,
    DATASET_SHA256,
    POOL_SIZE,
    SEEDS,
    budget_tag,
    checkpoint_metadata,
    protocol_manifest,
    validate_training_protocol,
)


def training_args(stage: int = 1, target_ratio: float = 0.4) -> Namespace:
    return Namespace(
        modality="vision",
        stage=stage,
        target_ratio=target_ratio,
        seed=42,
        data_seed=42,
        max_samples=500_000,
        steps=5_000,
        batch_size=32 if stage == 1 else 24,
        learning_rate=1e-3 if stage == 1 else 3e-4,
        weight_decay=0.05,
        num_workers=8,
        temperature=0.4,
        router_warmup=1_000,
        random_evals=3,
        best_budget_loss=0.01,
        log_every=100,
        save_every=500,
    )


def release_checkpoint(
    target_ratio: float = 0.4,
    study: object = "vision_budget_sweep",
) -> dict:
    return {
        "format_version": 3,
        "method": "sparmoe_vl_two_stage",
        "protocol": "frozen_spg_token_router_stage2",
        "stage": 2,
        "modality": "vision",
        "model_name": "ViT-L-14",
        "step": 4_600,
        "training_seed": 42,
        "target_ratio": target_ratio,
        "capacity_factors": list(CAPACITY_FACTORS),
        "study": study,
        "dataset": {
            "data_seed": 42,
            "samples": POOL_SIZE,
            "ordered_sha256": DATASET_SHA256,
        },
    }


def test_manifest_registers_exact_same_500k_pool_for_both_stages() -> None:
    for point in BUDGET_POINTS:
        for seed in SEEDS:
            manifest = protocol_manifest(point, seed)
            assert manifest["data"] == {
                "dataset": "ShareGPT4V",
                "data_seed": 42,
                "candidate_pool_size": 500_000,
                "ordered_sha256": DATASET_SHA256,
                "same_candidate_pool_in_both_stages": True,
            }
            assert manifest["stage1"]["optimizer_sample_exposures"] == 160_000
            assert manifest["stage2"]["optimizer_sample_exposures"] == 120_000
            assert manifest["stage2"]["loss_weights"] == {
                "representation": 100.0,
                "routing": 1.0,
            }
            assert "best_budget_loss" not in manifest["stage2"]


def test_budget_tags_are_stable() -> None:
    assert [budget_tag(point) for point in BUDGET_POINTS] == [
        "p04",
        "p05",
        "p06",
        "p07",
        "p08",
    ]


def test_training_accepts_exact_paper_protocol() -> None:
    validate_training_protocol(training_args(stage=1))
    validate_training_protocol(training_args(stage=2))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_samples", 50_000),
        ("data_seed", 123),
        ("steps", 4_999),
        ("batch_size", 16),
        ("num_workers", 4),
        ("temperature", 0.5),
        ("best_budget_loss", 0.1),
        ("log_every", 50),
    ],
)
def test_training_rejects_protocol_drift(field: str, value: object) -> None:
    args = training_args(stage=1)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_training_protocol(args)


def test_p07_is_reused_instead_of_retrained() -> None:
    with pytest.raises(ValueError, match="trainable sweep"):
        validate_training_protocol(training_args(target_ratio=0.7))
    metadata = checkpoint_metadata(release_checkpoint(target_ratio=0.7, study=None))
    assert metadata["reuses_visual_main_experiment"] is True


def test_release_checkpoint_identity_is_normalized() -> None:
    metadata = checkpoint_metadata(release_checkpoint())
    assert metadata["format"] == "release_v3"
    assert metadata["target_ratio"] == 0.4
    assert metadata["training_seed"] == 42
    assert metadata["pool_size"] == 500_000


def test_checkpoint_rejects_wrong_data_fingerprint() -> None:
    checkpoint = release_checkpoint()
    checkpoint["dataset"]["ordered_sha256"] = "wrong"
    with pytest.raises(ValueError, match="500k visual pool"):
        checkpoint_metadata(checkpoint)


def test_release_checkpoint_keeps_main_and_study_sources_distinct() -> None:
    sweep = checkpoint_metadata(release_checkpoint())
    main = checkpoint_metadata(release_checkpoint(target_ratio=0.7, study=None))
    assert sweep["reuses_visual_main_experiment"] is False
    assert main["reuses_visual_main_experiment"] is True
    with pytest.raises(ValueError, match="visual main"):
        checkpoint_metadata(release_checkpoint(target_ratio=0.7))

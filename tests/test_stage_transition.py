import pytest

from sparmoe_vl.common.training import validate_stage1_checkpoint


def checkpoint() -> dict:
    return {
        "format_version": 3,
        "method": "sparmoe_vl_two_stage",
        "protocol": "spg_global_budget_stage1",
        "stage": 1,
        "modality": "vision",
        "model_name": "ViT-L-14",
        "sparse_layers": [0, 1],
        "target_ratio": 0.7,
        "training_seed": 42,
        "dataset": {
            "source": "ShareGPT4V",
            "data_seed": 42,
            "samples": 500_000,
            "ordered_sha256": "a" * 64,
        },
    }


def validate(payload: dict, dataset: dict) -> None:
    validate_stage1_checkpoint(
        payload,
        modality="vision",
        model_name="ViT-L-14",
        sparse_layers=[0, 1],
        target_ratio=0.7,
        training_seed=42,
        dataset=dataset,
    )


def test_matching_stage_transition() -> None:
    payload = checkpoint()
    validate(payload, payload["dataset"])


def test_different_stage_data_is_rejected() -> None:
    payload = checkpoint()
    changed = dict(payload["dataset"], ordered_sha256="b" * 64)
    with pytest.raises(ValueError, match="subsets differ"):
        validate(payload, changed)

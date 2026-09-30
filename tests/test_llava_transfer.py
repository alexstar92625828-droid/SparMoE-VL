"""Protocol tests for the CLIP-336 to LLaVA transfer study."""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from sparmoe_vl.downstream.llava.evaluation.summarize import _sparse_metrics
from sparmoe_vl.downstream.llava.metrics import (
    mme_category_metrics,
    normalize_gqa_answer,
    normalize_vqa_answer,
    parse_yes_no,
    pope_metrics,
    vqav2_score,
)
from sparmoe_vl.downstream.llava.protocol import (
    CAPACITY_FACTORS,
    DATA_SEED,
    DENSE_VISUAL_FFN_MACS_G,
    MODEL_KEY,
    MODEL_NAME,
    POOL_SIZE,
    SEEDS,
    STAGE_SETTINGS,
    TARGET_RATIO,
    TRAINING_DATASET_SHA256,
    checkpoint_metadata,
)
from sparmoe_vl.downstream.llava.runtime import build_batch_inputs, build_input_ids
from sparmoe_vl.downstream.llava.training import validate_training_protocol


def test_release_checkpoint_identity_is_strict() -> None:
    checkpoint = {
        "format_version": 3,
        "method": "sparmoe_vl_two_stage",
        "stage": 2,
        "protocol": "frozen_spg_token_router_stage2",
        "modality": "vision",
        "training_seed": 123,
        "dataset": {
            "data_seed": DATA_SEED,
            "samples": POOL_SIZE,
            "ordered_sha256": TRAINING_DATASET_SHA256,
        },
        "target_ratio": TARGET_RATIO,
        "capacity_factors": list(CAPACITY_FACTORS),
        "step": 2_200,
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
    }
    metadata = checkpoint_metadata(checkpoint, expected_stage=2)
    assert metadata["format"] == "release_v3"
    assert metadata["training_seed"] == 123
    checkpoint["dataset"]["ordered_sha256"] = "wrong"
    with pytest.raises(ValueError, match="dataset_sha256"):
        checkpoint_metadata(checkpoint, expected_stage=2)


def test_training_cli_cannot_silently_change_paper_protocol() -> None:
    settings = STAGE_SETTINGS[2]
    assert settings["loss_weights"] == {
        "hidden_alignment": 100.0,
        "pooled_alignment": 100.0,
        "routing": 1.0,
    }
    arguments = Namespace(
        data_seed=DATA_SEED,
        max_samples=POOL_SIZE,
        target_ratio=TARGET_RATIO,
        steps=settings["steps"],
        batch_size=settings["batch_size"],
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        temperature=settings["temperature"],
        num_workers=settings["num_workers"],
        log_every=settings["log_every"],
        save_every=settings["save_every"],
        router_warmup=settings["router_warmup"],
        random_evals=settings["random_evals"],
    )
    validate_training_protocol(arguments, stage=2)
    arguments.batch_size = 32
    with pytest.raises(ValueError, match="batch_size"):
        validate_training_protocol(arguments, stage=2)


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool):
        del text
        return Namespace(input_ids=[1, 2] if add_special_tokens else [3, 4, 5])


def test_llava_prompt_contains_exactly_576_image_tokens() -> None:
    tokenizer = _Tokenizer()
    sequence = build_input_ids(tokenizer, "question", 32_000, 576)
    assert sequence.shape == (581,)
    assert int((sequence == 32_000).sum()) == 576
    input_ids, attention_mask = build_batch_inputs(
        tokenizer, ["short", "longer question"], 32_000, 576
    )
    assert input_ids.shape == attention_mask.shape == (2, 581)
    assert torch.all(attention_mask == 1)


def test_answer_parsing_and_paper_metrics() -> None:
    assert parse_yes_no("Yes, it is.") == "yes"
    assert parse_yes_no("I think no.") == "no"
    assert parse_yes_no("unclear") == "unknown"
    pope = pope_metrics(
        [
            {"label": "yes", "prediction": "yes"},
            {"label": "no", "prediction": "yes"},
            {"label": "yes", "prediction": "unknown"},
        ]
    )
    assert pope["yes_ratio"] == pytest.approx(2 / 3)
    assert pope["f1"] == pytest.approx(0.5)
    mme = mme_category_metrics(
        [
            {"group_id": "a", "label": "yes", "prediction": "yes"},
            {"group_id": "a", "label": "no", "prediction": "no"},
            {"group_id": "b", "label": "yes", "prediction": "no"},
            {"group_id": "b", "label": "no", "prediction": "no"},
        ]
    )
    assert mme["accuracy"] == pytest.approx(0.75)
    assert mme["accuracy_plus"] == pytest.approx(0.5)
    assert mme["score"] == pytest.approx(125.0)
    assert normalize_gqa_answer("The answer is: Blue.") == "blue"
    assert normalize_vqa_answer("The two cats.") == "2 cats"
    assert vqav2_score("two", ["2", "two", "2", "three"]) == pytest.approx(1.0)


def test_macs_and_retention_formulas_match_table_semantics() -> None:
    expected_dense = 24 * 577 * 2 * 1024 * 4096 / 1e9
    assert DENSE_VISUAL_FFN_MACS_G == pytest.approx(expected_dense)
    dense = {
        "visual_ffn_macs_g": expected_dense,
        "pope_avg_f1": 0.8,
        "pope_yes_ratio": 0.4,
        "mme_p_score": 1_500.0,
        "gqa_acc_pct": 60.0,
        "vqav2_acc_pct": 75.0,
    }
    sparse_results = {
        "visual_ffn_macs": {"metrics": {"sparse_visual_ffn_macs_g": 80.0}},
        "pope": {"metrics": {"pope_avg_f1": 0.792, "pope_yes_ratio": 0.39}},
        "mme_p": {"metrics": {"mme_p_score": 1_350.0}},
        "gqa": {"metrics": {"gqa_acc": 0.594}},
        "vqav2": {"metrics": {"vqav2_acc": 0.735}},
    }
    row = _sparse_metrics(sparse_results, dense)
    assert row["pope_f1_retention_pct"] == pytest.approx(99.0)
    assert row["pope_yes_ratio_delta"] == pytest.approx(-0.01)
    assert row["mme_p_retention_pct"] == pytest.approx(90.0)
    assert row["gqa_retention_pct"] == pytest.approx(99.0)
    assert row["vqav2_retention_pct"] == pytest.approx(98.0)


def test_seed_set_is_the_paper_seed_set() -> None:
    assert SEEDS == (42, 123, 2026)

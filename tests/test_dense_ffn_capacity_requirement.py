import copy
import json
from pathlib import Path

import pytest
import torch

from sparmoe_vl.studies.dense_ffn_capacity_requirement.analysis import (
    build_split,
    measure_batch_requirements,
    split_identity,
)
from sparmoe_vl.studies.dense_ffn_capacity_requirement.merge import (
    merge_shards,
    normalize_shard,
    validate_merged,
)
from sparmoe_vl.studies.dense_ffn_capacity_requirement.plotting import (
    load_plot_data,
)
from sparmoe_vl.studies.dense_ffn_capacity_requirement.protocol import (
    CALIBRATION_IMAGES,
    CAPACITY_LEVELS,
    CHANNEL_RANKING,
    COCO_ANNOTATIONS_SHA256,
    COMPARISON,
    COSINE_THRESHOLD,
    EVALUATION_IMAGES,
    FFN_DIM,
    MODEL_KEY,
    MODEL_NAME,
    NRE_THRESHOLD,
    NUM_LAYERS,
    PAPER_SCOPE,
    PATCH_TOKENS,
    PRETRAINED_SHA256,
    PROTOCOL,
    SEED,
    SPLIT_METHOD,
    STUDY_NAME,
)


OBSERVATIONS = EVALUATION_IMAGES * PATCH_TOKENS


def layer_record() -> dict:
    counts = [OBSERVATIONS // len(CAPACITY_LEVELS)] * len(CAPACITY_LEVELS)
    proportions = [count / OBSERVATIONS for count in counts]
    mean = sum(level * count for level, count in zip(CAPACITY_LEVELS, counts)) / OBSERVATIONS
    return {
        "observations": OBSERVATIONS,
        "required_capacity_counts": counts,
        "required_capacity_proportions": proportions,
        "layer_mean": mean,
        "cosine_pass_counts": [0] * len(CAPACITY_LEVELS),
        "nre_pass_counts": [0] * len(CAPACITY_LEVELS),
        "joint_pass_counts": [0] * len(CAPACITY_LEVELS),
        "mean_cosine_by_capacity": [0.0] * len(CAPACITY_LEVELS),
        "mean_nre_by_capacity": [0.0] * len(CAPACITY_LEVELS),
    }


def shard(layer_start: int, layer_end: int) -> dict:
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "backbone": "frozen_dense_clip",
        "pretrained_sha256": PRETRAINED_SHA256,
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "seed": SEED,
        "split_method": SPLIT_METHOD,
        "calibration_images": CALIBRATION_IMAGES,
        "evaluation_images": EVALUATION_IMAGES,
        "calibration_split_sha256": "1" * 64,
        "evaluation_split_sha256": "2" * 64,
        "split_disjoint": True,
        "patch_tokens_per_image": PATCH_TOKENS,
        "ffn_dim": FFN_DIM,
        "capacity_levels": list(CAPACITY_LEVELS),
        "cosine_threshold": COSINE_THRESHOLD,
        "nre_threshold": NRE_THRESHOLD,
        "channel_ranking": CHANNEL_RANKING,
        "comparison": COMPARISON,
        "batch_first_verified": True,
        "tf32": False,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "ranking_file": "generated_rankings.pt",
        "layers": {str(layer): layer_record() for layer in range(layer_start, layer_end + 1)},
    }


def test_seeded_coco_split_is_disjoint_deterministic_and_exhaustive() -> None:
    first = build_split(5_000, SEED, CALIBRATION_IMAGES, EVALUATION_IMAGES)
    second = build_split(5_000, SEED, CALIBRATION_IMAGES, EVALUATION_IMAGES)
    calibration, evaluation = first
    assert first == second
    assert len(calibration) == CALIBRATION_IMAGES
    assert len(evaluation) == EVALUATION_IMAGES
    assert set(calibration).isdisjoint(evaluation)
    assert set(calibration + evaluation) == set(range(5_000))


def test_strict_split_rejects_a_partial_dataset() -> None:
    with pytest.raises(ValueError, match="consume every COCO image"):
        build_split(5_000, SEED, 100, 100)


def test_split_identity_does_not_depend_on_the_local_data_root() -> None:
    indices = [2, 0, 1]
    first = [Path("/first/root") / f"{index}.jpg" for index in range(3)]
    second = [Path("/another/root") / f"{index}.jpg" for index in range(3)]
    assert split_identity(indices, first) == split_identity(indices, second)


def test_minimum_capacity_requires_both_reconstruction_criteria() -> None:
    ffn_dim = 10
    hidden = torch.stack((torch.eye(ffn_dim)[0], torch.eye(ffn_dim)[-1]))
    projection = torch.eye(ffn_dim)
    bias = torch.zeros(ffn_dim)
    order = torch.arange(ffn_dim)
    assignments, statistics = measure_batch_requirements(
        hidden,
        projection,
        bias,
        order,
        cosine_threshold=0.95,
        nre_threshold=0.20,
    )
    assert assignments.tolist() == [0, len(CAPACITY_LEVELS) - 1]
    assert statistics["joint_pass_counts"][0].item() == 1
    assert statistics["joint_pass_counts"][-1].item() == len(hidden)


def test_two_shards_merge_into_one_complete_layer_set() -> None:
    merged = merge_shards(
        [shard(1, 12), shard(13, 24)],
        ["layers_01_12.json", "layers_13_24.json"],
    )
    validate_merged(merged, "merged")
    assert list(merged["layers"]) == [str(layer) for layer in range(1, 25)]


def test_merger_rejects_overlap_and_protocol_drift() -> None:
    with pytest.raises(ValueError, match="duplicate layer"):
        merge_shards(
            [shard(1, 12), shard(12, 24)],
            ["first.json", "overlap.json"],
        )
    changed = shard(13, 24)
    changed["evaluation_split_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="evaluation_split_sha256"):
        merge_shards([shard(1, 12), changed], ["first.json", "changed.json"])


def test_historical_schema_is_normalized_without_changing_the_method() -> None:
    current = shard(1, 12)
    historical = {
        **current,
        "model": current["model_name"],
        "levels": current["capacity_levels"],
        "channel_ranking": "mean squared GELU activation times squared c_proj column norm",
        "comparison": "local Dense FFN output, excluding CLS token",
    }
    for field in (
        "study",
        "paper_scope",
        "model_name",
        "model_key",
        "backbone",
        "ffn_dim",
        "capacity_levels",
    ):
        historical.pop(field, None)
    normalized = normalize_shard(historical, "historical.json")
    assert normalized["source_format"] == "historical"
    assert normalized["channel_ranking"] == CHANNEL_RANKING
    assert normalized["comparison"] == COMPARISON


def test_plot_loader_requires_a_fully_validated_merged_analysis(
    tmp_path: Path,
) -> None:
    merged = merge_shards(
        [shard(1, 12), shard(13, 24)],
        ["layers_01_12.json", "layers_13_24.json"],
    )
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(merged), encoding="utf-8")
    levels, proportions, means = load_plot_data(path)
    assert levels.shape == (len(CAPACITY_LEVELS),)
    assert proportions.shape == (NUM_LAYERS, len(CAPACITY_LEVELS))
    assert means.shape == (NUM_LAYERS,)

    changed = copy.deepcopy(merged)
    changed["cosine_threshold"] = 0.0
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="cosine_threshold"):
        load_plot_data(path)

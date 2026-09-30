import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sparmoe_vl.studies.layerwise_capacity_allocation.analysis import (
    build_payload,
    capacity_statistics,
    load_coco_image_paths,
    partial_manifest,
    restore_partial,
    save_partial,
    validate_analysis,
)
from sparmoe_vl.studies.layerwise_capacity_allocation.plotting import (
    load_plot_data,
)
from sparmoe_vl.studies.layerwise_capacity_allocation.protocol import (
    CAPACITY_FACTORS,
    COCO_ANNOTATIONS_SHA256,
    COCO_IMAGES_TOTAL,
    COCO_MANIFEST_SHA256,
    DATA_SEED,
    MODEL_KEY,
    MODEL_NAME,
    NUM_EXPERTS,
    NUM_LAYERS,
    PAPER_SCOPE,
    PATCHES_PER_IMAGE,
    PROTOCOL,
    ROUTING_MODE,
    RUN_SEED,
    STUDY_NAME,
    TARGET_RATIO,
    TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
)


def checkpoint_metadata() -> dict:
    return {
        "format": "release_v2",
        "protocol": "frozen_spg_token_router_stage2",
        "target_ratio": TARGET_RATIO,
        "training_seed": RUN_SEED,
        "data_seed": DATA_SEED,
        "pool_size": TRAINING_POOL_SIZE,
        "dataset_sha256": TRAINING_POOL_SHA256,
        "checkpoint_step": 1,
        "capacity_factors": list(CAPACITY_FACTORS),
        "reuses_visual_main_experiment": True,
        "checkpoint": "generated.pt",
        "checkpoint_sha256": "1" * 64,
    }


def complete_analysis() -> dict:
    tokens = COCO_IMAGES_TOTAL * PATCHES_PER_IMAGE
    counts = torch.zeros(NUM_LAYERS, NUM_EXPERTS, dtype=torch.int64)
    counts[:, 0] = tokens
    paths = [Path(f"{index:012d}.jpg") for index in range(COCO_IMAGES_TOTAL)]
    payload = build_payload(
        paths,
        checkpoint_metadata(),
        counts,
        torch.full((NUM_LAYERS,), 0.5),
        elapsed=0.0,
        partial_smoke=False,
    )
    payload["dataset_manifest_sha256"] = COCO_MANIFEST_SHA256
    return payload


def test_coco_image_order_is_ascending_by_image_id(tmp_path: Path) -> None:
    annotations = tmp_path / "captions.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 20, "file_name": "twenty.jpg"},
                    {"id": 3, "file_name": "three.jpg"},
                    {"id": 11, "file_name": "eleven.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )
    paths = load_coco_image_paths(annotations, tmp_path / "images")
    assert [path.name for path in paths] == ["three.jpg", "eleven.jpg", "twenty.jpg"]


def test_coco_image_order_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    annotations = tmp_path / "captions.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "first.jpg"},
                    {"id": 1, "file_name": "second.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identifiers must be unique"):
        load_coco_image_paths(annotations, tmp_path / "images")


def test_capacity_statistics_use_layer_budget_and_expert_usage() -> None:
    base = np.full(NUM_LAYERS, 0.5)
    usage = np.full((NUM_LAYERS, NUM_EXPERTS), 1.0 / NUM_EXPERTS)
    capacities, activated = capacity_statistics(base, usage)
    assert capacities.shape == (NUM_LAYERS, NUM_EXPERTS)
    assert capacities[0].tolist() == pytest.approx(
        [0.5 * factor for factor in CAPACITY_FACTORS]
    )
    assert activated[0] == pytest.approx(capacities[0].mean())


def test_partial_counts_are_bound_to_data_checkpoint_and_progress(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.partial.npz"
    processed = 2
    counts = torch.zeros(NUM_LAYERS, NUM_EXPERTS, dtype=torch.int64)
    counts[:, 0] = processed * PATCHES_PER_IMAGE
    manifest = partial_manifest("a" * 64, "b" * 64, 10)
    save_partial(path, counts, processed, manifest)
    restored, restored_count = restore_partial(path, manifest, 10)
    assert torch.equal(restored, counts)
    assert restored_count == processed
    with pytest.raises(ValueError, match="different protocol"):
        restore_partial(path, partial_manifest("a" * 64, "c" * 64, 10), 10)


def test_analysis_validator_checks_count_and_capacity_conservation() -> None:
    payload = complete_analysis()
    validate_analysis(payload, "analysis")
    payload["counts_layer_by_expert"][0][0] -= 1
    with pytest.raises(ValueError, match="conserve patch tokens"):
        validate_analysis(payload, "analysis")


def test_plot_loader_accepts_only_valid_complete_analysis(tmp_path: Path) -> None:
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(complete_analysis()), encoding="utf-8")
    data = load_plot_data(path)
    assert data["usage"].shape == (NUM_LAYERS, NUM_EXPERTS)
    assert data["capacities"].shape == (NUM_LAYERS, NUM_EXPERTS)


def test_public_defaults_keep_generated_artifacts_under_outputs() -> None:
    from sparmoe_vl.studies.layerwise_capacity_allocation.analysis import parse_args

    args = parse_args([])
    assert "outputs/figures/layerwise_capacity_allocation" in str(args.output)
    assert args.device == "cpu"
    assert args.batch_size == 8
    assert args.num_workers == 0


def test_check_payload_contains_no_experimental_measurements() -> None:
    from sparmoe_vl.studies.layerwise_capacity_allocation.protocol import (
        protocol_manifest,
    )

    manifest = protocol_manifest()
    assert manifest["study"] == STUDY_NAME
    assert manifest["paper_scope"] == PAPER_SCOPE
    assert manifest["model_name"] == MODEL_NAME
    assert manifest["model_key"] == MODEL_KEY
    assert manifest["routing"] == ROUTING_MODE
    assert "counts_layer_by_expert" not in manifest
    assert "proportions_layer_by_expert" not in manifest
    assert manifest["images"] == COCO_IMAGES_TOTAL
    assert manifest["patches_per_image"] == PATCHES_PER_IMAGE
    assert manifest["training_pool_sha256"] == TRAINING_POOL_SHA256
    assert COCO_ANNOTATIONS_SHA256


def test_parser_rejects_unexpected_positional_arguments() -> None:
    from sparmoe_vl.studies.layerwise_capacity_allocation.analysis import parse_args

    with pytest.raises(SystemExit):
        parse_args(["unexpected"])


def test_build_payload_marks_smoke_results_as_non_paper_protocol() -> None:
    counts = torch.zeros(NUM_LAYERS, NUM_EXPERTS, dtype=torch.int64)
    counts[:, 0] = PATCHES_PER_IMAGE
    payload = build_payload(
        [Path("one.jpg")],
        checkpoint_metadata(),
        counts,
        torch.full((NUM_LAYERS,), 0.5),
        elapsed=0.0,
        partial_smoke=True,
    )
    assert payload["protocol"] == f"{PROTOCOL}_smoke"

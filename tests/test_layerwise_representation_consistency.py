import json
from pathlib import Path

import numpy as np
import pytest

from sparmoe_vl.studies.layerwise_representation_consistency.analysis import (
    CACHE_NAMES,
    cache_shapes,
    centered_linear_cka,
    load_coco_image_paths,
    state_manifest,
    validate_analysis,
    validate_progress,
)
from sparmoe_vl.studies.layerwise_representation_consistency.plotting import (
    layer_histograms,
    load_patch_cache,
)
from sparmoe_vl.studies.layerwise_representation_consistency.protocol import (
    CAPACITY_FACTORS,
    COCO_ANNOTATIONS_SHA256,
    COCO_IMAGES_TOTAL,
    COCO_MANIFEST_SHA256,
    DATA_SEED,
    HISTOGRAM_BINS,
    MODEL_KEY,
    MODEL_NAME,
    NUM_LAYERS,
    PAPER_SCOPE,
    PATCHES_PER_IMAGE,
    PROTOCOL,
    RUN_SEED,
    STUDY_NAME,
    TARGET_RATIO,
    TOKENS_PER_IMAGE,
    TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
    protocol_manifest,
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
    zeros = [0.0] * NUM_LAYERS
    halves = [0.5] * NUM_LAYERS
    distribution = {
        "mean": list(halves),
        "std": list(zeros),
        "q10": list(halves),
        "q25": list(halves),
        "median": list(halves),
        "q75": list(halves),
        "q90": list(halves),
    }
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "checkpoint_metadata": checkpoint_metadata(),
        "routing": "learned_argmax",
        "representation_point": "post_transformer_block",
        "dataset": "COCO-val2017",
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "dataset_manifest_sha256": COCO_MANIFEST_SHA256,
        "image_order": "ascending_file_name",
        "image_count": COCO_IMAGES_TOTAL,
        "num_layers": NUM_LAYERS,
        "tokens_per_image": TOKENS_PER_IMAGE,
        "patches_per_image": PATCHES_PER_IMAGE,
        "all_tokens_per_layer": COCO_IMAGES_TOTAL * TOKENS_PER_IMAGE,
        "patch_tokens_per_layer": COCO_IMAGES_TOTAL * PATCHES_PER_IMAGE,
        "cka_estimator": "exact_centered_linear_cka_over_all_cls_states",
        "layers_one_based": list(range(1, NUM_LAYERS + 1)),
        "cls_linear_cka": [1.0] * NUM_LAYERS,
        "all_token_cosine": list(halves),
        "cls_token_cosine": list(halves),
        "patch_token_cosine": distribution,
        "cache": {
            key: {
                "file": CACHE_NAMES[key],
                "dtype": "float32",
                "shape": list(shape),
            }
            for key, shape in cache_shapes(COCO_IMAGES_TOTAL).items()
        },
        "tf32": False,
    }


def test_coco_order_matches_result_generating_file_name_sort(tmp_path: Path) -> None:
    annotations = tmp_path / "captions.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 20, "file_name": "000020.jpg"},
                    {"id": 3, "file_name": "000003.jpg"},
                    {"id": 11, "file_name": "000011.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )
    paths = load_coco_image_paths(annotations, tmp_path / "images")
    assert [path.name for path in paths] == ["000003.jpg", "000011.jpg", "000020.jpg"]


def test_exact_centered_linear_cka_is_invariant_to_orthogonal_rotation() -> None:
    rng = np.random.default_rng(42)
    dense = rng.normal(size=(64, 8)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    sparse = (dense @ q.astype(np.float32)).astype(np.float32)
    assert centered_linear_cka(dense, sparse) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="share a two-dimensional shape"):
        centered_linear_cka(dense, sparse[:-1])


def test_progress_is_bound_to_protocol_and_conserves_token_counts() -> None:
    processed = 3
    progress = {
        "state_manifest": "a" * 64,
        "image_count": 10,
        "processed": processed,
        "all_token_cosine_sums": [1.0] * NUM_LAYERS,
        "cls_token_cosine_sums": [1.0] * NUM_LAYERS,
        "all_token_counts": [processed * TOKENS_PER_IMAGE] * NUM_LAYERS,
        "cls_token_counts": [processed] * NUM_LAYERS,
    }
    values = validate_progress(progress, "a" * 64, 10)
    assert values[-1] == processed
    progress["all_token_counts"][0] -= 1
    with pytest.raises(ValueError, match="token counts disagree"):
        validate_progress(progress, "a" * 64, 10)
    assert state_manifest("b" * 64, "c" * 64, 10) != state_manifest("b" * 64, "d" * 64, 10)


def test_histogram_conserves_every_patch_token() -> None:
    values = np.linspace(-0.2, 1.0, 2 * NUM_LAYERS * PATCHES_PER_IMAGE).reshape(
        2, NUM_LAYERS, PATCHES_PER_IMAGE
    )
    edges, fractions = layer_histograms(values)
    assert len(edges) == HISTOGRAM_BINS + 1
    assert fractions.shape == (HISTOGRAM_BINS, NUM_LAYERS)
    assert fractions.sum(axis=0) == pytest.approx(np.full(NUM_LAYERS, 100.0))
    values[0, 0, 0] = -0.5
    with pytest.raises(ValueError, match="registered plot range"):
        layer_histograms(values)


def test_analysis_validator_rejects_partial_and_malformed_metrics() -> None:
    payload = complete_analysis()
    validate_analysis(payload, "analysis")
    payload["protocol"] = f"{PROTOCOL}_smoke"
    with pytest.raises(ValueError, match="protocol"):
        validate_analysis(payload, "analysis")
    payload["image_count"] = 2
    payload["dataset_manifest_sha256"] = "a" * 64
    payload["all_tokens_per_layer"] = 2 * TOKENS_PER_IMAGE
    payload["patch_tokens_per_layer"] = 2 * PATCHES_PER_IMAGE
    for key, shape in cache_shapes(2).items():
        payload["cache"][key]["shape"] = list(shape)
    validate_analysis(payload, "analysis", allow_partial_smoke=True)
    payload = complete_analysis()
    payload["patch_token_cosine"]["q25"][0] = 0.8
    with pytest.raises(ValueError, match="quantiles"):
        validate_analysis(payload, "analysis")


def test_patch_cache_loader_checks_exact_contract(tmp_path: Path) -> None:
    payload = complete_analysis()
    shape = (2, NUM_LAYERS, PATCHES_PER_IMAGE)
    payload["cache"]["patch_cosine"]["shape"] = list(shape)
    analysis = tmp_path / "analysis.json"
    cache = np.memmap(
        tmp_path / CACHE_NAMES["patch_cosine"], dtype=np.float32, mode="w+", shape=shape
    )
    cache[:] = 0.5
    cache.flush()
    loaded = load_patch_cache(payload, analysis)
    assert loaded.shape == shape
    (tmp_path / CACHE_NAMES["patch_cosine"]).write_bytes(b"short")
    with pytest.raises(ValueError, match="unexpected size"):
        load_patch_cache(payload, analysis)


def test_protocol_manifest_contains_no_measured_results() -> None:
    manifest = protocol_manifest()
    assert manifest["training_pool_size"] == 500_000
    assert manifest["training_pool_sha256"] == TRAINING_POOL_SHA256
    assert manifest["images"] == 5_000
    assert manifest["metrics"]["local"] == "full_patch_token_cosine_distribution"
    assert "cls_linear_cka" not in manifest
    assert "patch_token_cosine" not in manifest


def test_public_directory_is_semantic_and_contains_no_results() -> None:
    project = Path(__file__).resolve().parents[1]
    directory = project / "experiments" / "figures" / STUDY_NAME
    assert {path.name for path in directory.iterdir() if path.is_file()} == {
        "config.yaml",
        "analyze.py",
        "plot.py",
        "run_study.sh",
    }
    assert not any(
        path.suffix in {".json", ".csv", ".pt", ".mmap"} for path in directory.iterdir()
    )
    assert all("figure" not in path.stem.lower() for path in directory.iterdir())


def test_cli_defaults_keep_generated_artifacts_under_outputs() -> None:
    from sparmoe_vl.studies.layerwise_representation_consistency.analysis import parse_args

    args = parse_args([])
    assert args.batch_size == 4
    assert args.num_workers == 0
    assert args.max_images == 5_000
    assert "outputs/figures/layerwise_representation_consistency" in str(args.output_dir)


def test_parser_rejects_unexpected_positional_arguments() -> None:
    from sparmoe_vl.studies.layerwise_representation_consistency.analysis import parse_args

    with pytest.raises(SystemExit):
        parse_args(["unexpected"])

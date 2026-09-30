from pathlib import Path

import numpy as np
import pytest
import torch

from sparmoe_vl.studies.cross_modal_similarity_preservation.analysis import (
    FEATURE_FILE,
    SAMPLE_FILE,
    SAMPLE_MANIFEST_FILE,
    PaperCOCO,
    build_visualization,
    corpus_identity,
    fixed_unordered_sample,
    full_matrix_statistics,
    subset_coco,
    validate_analysis,
    validate_features,
    validate_visualization,
)
from sparmoe_vl.studies.cross_modal_similarity_preservation.plotting import (
    aggregate_square_matrix,
)
from sparmoe_vl.studies.cross_modal_similarity_preservation.protocol import (
    CAPACITY_FACTORS,
    CAPTION_COUNT,
    COCO_ANNOTATIONS_SHA256,
    COCO_CAPTION_ANNOTATION_ID_SHA256,
    COCO_CAPTION_IMAGE_INDEX_SHA256,
    COCO_CAPTION_ORDER_SHA256,
    COCO_FIRST_CAPTION_INDEX_SHA256,
    COCO_IMAGE_ORDER_SHA256,
    DATA_SEED,
    DISPLAY_GROUPS,
    IMAGE_COUNT,
    LAYER_COUNT,
    LAYERWISE_PROTOCOL,
    LAYERWISE_STUDY,
    MODEL_KEY,
    MODEL_NAME,
    OUTPUT_DIM,
    PAIRWISE_SIMILARITY_COUNT,
    PAPER_SCOPE,
    PRETRAINED_SHA256,
    PROTOCOL,
    RUN_SEED,
    SAMPLE_COUNT,
    SAMPLE_SEED,
    STUDY_NAME,
    TARGET_RATIO,
    TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
    UNORDERED_SAMPLE_INDEX_SHA256,
    protocol_manifest,
)


def toy_corpus(image_count: int = 4) -> PaperCOCO:
    captions = tuple(f"caption {index}" for index in range(image_count * 2))
    mapping = np.repeat(np.arange(image_count), 2).astype(np.int64)
    return PaperCOCO(
        image_paths=tuple(Path(f"{index:012d}.jpg") for index in range(image_count)),
        image_ids=tuple(range(100, 100 + image_count)),
        file_names=tuple(f"{index:012d}.jpg" for index in range(image_count)),
        captions=captions,
        caption_image_indices=mapping,
        caption_annotation_ids=tuple(range(1_000, 1_000 + len(captions))),
        first_caption_indices=np.arange(0, len(captions), 2, dtype=np.int64),
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


def smoke_analysis(corpus: PaperCOCO, sample_count: int = 2) -> dict:
    identities = corpus_identity(corpus)
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": f"{PROTOCOL}_smoke",
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "visual_checkpoint_metadata": checkpoint_metadata(),
        "text_encoder": "frozen_dense_clip",
        "pretrained_sha256": PRETRAINED_SHA256,
        "evaluation": {
            "dataset": "COCO-val2017",
            "annotation_sha256": COCO_ANNOTATIONS_SHA256,
            "images": len(corpus.image_paths),
            "captions": len(corpus.captions),
            **identities,
        },
        "image_feature_source": {
            "study": LAYERWISE_STUDY,
            "protocol": f"{LAYERWISE_PROTOCOL}_smoke",
            "analysis_sha256": "a" * 64,
            "layer": LAYER_COUNT,
            "representation": "projected_final_cls",
        },
        "statistics": {
            "pairwise_similarity_count": len(corpus.image_paths) * len(corpus.captions),
            "pearson": 0.5,
            "matrix_cosine": 0.5,
            "mae": 0.1,
            "rmse": 0.2,
        },
        "visualization": {
            "sample_seed": SAMPLE_SEED,
            "sample_count": sample_count,
            "unordered_selection_sha256": "b" * 64,
            "ordering": "average_linkage_dense_joint_image_text_embedding",
            "sample_file": SAMPLE_FILE,
            "manifest_file": SAMPLE_MANIFEST_FILE,
        },
        "feature_file": FEATURE_FILE,
    }


def normalized(values: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(values.float(), dim=-1)


def test_protocol_registers_full_matrix_and_dense_text_tower() -> None:
    manifest = protocol_manifest()
    assert manifest["training_data"]["samples"] == 500_000
    assert manifest["training_data"]["ordered_sha256"] == TRAINING_POOL_SHA256
    assert manifest["text_encoder"] == "frozen_dense_clip"
    assert manifest["evaluation"]["images"] == 5_000
    assert manifest["evaluation"]["captions"] == 25_014
    assert manifest["evaluation"]["pairwise_similarities"] == PAIRWISE_SIMILARITY_COUNT
    assert manifest["visualization"]["matched_pairs"] == 256
    assert manifest["visualization"]["display_groups"] == 64
    assert "pearson" not in manifest
    assert "matrix_cosine" not in manifest
    assert "mae" not in manifest


def test_registered_coco_identities_cover_every_ordering_relation() -> None:
    assert IMAGE_COUNT == 5_000
    assert CAPTION_COUNT == 25_014
    for digest in (
        COCO_IMAGE_ORDER_SHA256,
        COCO_CAPTION_ORDER_SHA256,
        COCO_CAPTION_IMAGE_INDEX_SHA256,
        COCO_CAPTION_ANNOTATION_ID_SHA256,
        COCO_FIRST_CAPTION_INDEX_SHA256,
    ):
        assert len(digest) == 64


def test_smoke_subset_keeps_only_selected_images_and_their_captions() -> None:
    corpus = toy_corpus(4)
    subset = subset_coco(corpus, 2)
    assert subset.file_names == corpus.file_names[:2]
    assert subset.captions == corpus.captions[:4]
    assert subset.caption_image_indices.tolist() == [0, 0, 1, 1]
    assert subset.first_caption_indices.tolist() == [0, 2]
    with pytest.raises(ValueError, match="outside the COCO corpus"):
        subset_coco(corpus, 0)


def test_full_matrix_statistics_match_direct_flattened_calculation() -> None:
    generator = torch.Generator().manual_seed(42)
    dense = normalized(torch.randn(5, 6, generator=generator))
    sparse = normalized(dense + 0.1 * torch.randn(5, 6, generator=generator))
    texts = normalized(torch.randn(9, 6, generator=generator))
    result = full_matrix_statistics(
        dense,
        sparse,
        texts,
        device=torch.device("cpu"),
        block_size=2,
    )
    dense_matrix = (dense @ texts.T).double().numpy().reshape(-1)
    sparse_matrix = (sparse @ texts.T).double().numpy().reshape(-1)
    assert result["pairwise_similarity_count"] == 45
    assert result["pearson"] == pytest.approx(np.corrcoef(dense_matrix, sparse_matrix)[0, 1])
    assert result["matrix_cosine"] == pytest.approx(
        np.dot(dense_matrix, sparse_matrix)
        / (np.linalg.norm(dense_matrix) * np.linalg.norm(sparse_matrix))
    )
    difference = sparse_matrix - dense_matrix
    assert result["mae"] == pytest.approx(np.abs(difference).mean())
    assert result["rmse"] == pytest.approx(np.sqrt(np.square(difference).mean()))


def test_fixed_sample_is_independent_of_features_and_hash_locked() -> None:
    selected = fixed_unordered_sample(IMAGE_COUNT, SAMPLE_COUNT)
    assert selected.shape == (SAMPLE_COUNT,)
    assert len(np.unique(selected)) == SAMPLE_COUNT
    from sparmoe_vl.studies.cross_modal_similarity_preservation.analysis import (
        sequence_sha256,
    )

    assert sequence_sha256(selected.tolist()) == UNORDERED_SAMPLE_INDEX_SHA256


def test_visualization_uses_first_captions_and_one_shared_order(tmp_path: Path) -> None:
    corpus = toy_corpus(4)
    generator = torch.Generator().manual_seed(7)
    features = {
        "protocol": f"{PROTOCOL}_smoke",
        "dense_image_features": normalized(torch.randn(4, 8, generator=generator)),
        "sparse_image_features": normalized(torch.randn(4, 8, generator=generator)),
        "text_features": normalized(torch.randn(8, 8, generator=generator)),
    }
    sample, manifest = build_visualization(features, corpus, 4, tmp_path)
    validate_visualization(
        sample,
        manifest,
        corpus,
        4,
        allow_partial_smoke=True,
    )
    assert sample["dense"].shape == (4, 4)
    assert np.array_equal(
        sample["caption_indices"],
        corpus.first_caption_indices[sample["image_indices"]],
    )
    changed = dict(sample)
    changed["caption_indices"] = sample["caption_indices"].copy()
    changed["caption_indices"][0] += 1
    with pytest.raises(ValueError, match="first caption"):
        validate_visualization(
            changed,
            manifest,
            corpus,
            4,
            allow_partial_smoke=True,
        )


def test_group_aggregation_averages_nonoverlapping_cells() -> None:
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    grouped = aggregate_square_matrix(matrix, 2)
    expected = np.asarray(
        [
            [matrix[:2, :2].mean(), matrix[:2, 2:].mean()],
            [matrix[2:, :2].mean(), matrix[2:, 2:].mean()],
        ]
    )
    assert np.array_equal(grouped, expected)
    with pytest.raises(ValueError, match="not divisible"):
        aggregate_square_matrix(matrix, 3)


def test_feature_validator_binds_dense_text_and_layerwise_source() -> None:
    corpus = toy_corpus(2)
    generator = torch.Generator().manual_seed(3)
    features = {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": f"{PROTOCOL}_smoke",
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "image_count": 2,
        "caption_count": 4,
        "feature_dimension": OUTPUT_DIM,
        "text_encoder": "frozen_dense_clip",
        "pretrained_sha256": PRETRAINED_SHA256,
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        **corpus_identity(corpus),
        "layerwise_source": {
            "study": LAYERWISE_STUDY,
            "protocol": f"{LAYERWISE_PROTOCOL}_smoke",
            "analysis_sha256": "a" * 64,
        },
        "dense_image_features": normalized(torch.randn(2, OUTPUT_DIM, generator=generator)),
        "sparse_image_features": normalized(torch.randn(2, OUTPUT_DIM, generator=generator)),
        "text_features": normalized(torch.randn(4, OUTPUT_DIM, generator=generator)),
        "caption_image_indices": torch.tensor([0, 0, 1, 1]),
        "first_caption_indices": torch.tensor([0, 2]),
        "image_ids": list(corpus.image_ids),
        "image_files": list(corpus.file_names),
    }
    validate_features(
        features,
        corpus,
        "a" * 64,
        allow_partial_smoke=True,
    )
    features["text_encoder"] = "sparse_text"
    with pytest.raises(ValueError, match="text_encoder"):
        validate_features(
            features,
            corpus,
            "a" * 64,
            allow_partial_smoke=True,
        )


def test_analysis_validator_rejects_partial_by_default_and_bad_counts() -> None:
    corpus = toy_corpus(2)
    payload = smoke_analysis(corpus)
    validate_analysis(payload, "analysis", allow_partial_smoke=True)
    with pytest.raises(ValueError, match="protocol"):
        validate_analysis(payload, "analysis")
    payload["statistics"]["pairwise_similarity_count"] -= 1
    with pytest.raises(ValueError, match="count is incomplete"):
        validate_analysis(payload, "analysis", allow_partial_smoke=True)


def test_public_directory_has_semantic_names_and_no_generated_results() -> None:
    project = Path(__file__).resolve().parents[1]
    directory = project / "experiments" / "figures" / STUDY_NAME
    assert {path.name for path in directory.iterdir() if path.is_file()} == {
        "config.yaml",
        "analyze.py",
        "plot.py",
        "run_study.sh",
    }
    assert not any(
        path.suffix in {".json", ".csv", ".pt", ".npz", ".pdf", ".png"}
        for path in directory.iterdir()
    )
    assert all("figure" not in path.stem.lower() for path in directory.iterdir())


def test_cli_defaults_match_result_generating_protocol() -> None:
    from sparmoe_vl.studies.cross_modal_similarity_preservation.analysis import (
        parse_args,
    )

    args = parse_args([])
    assert args.device == "cuda:0"
    assert args.max_images == IMAGE_COUNT
    assert args.sample_count == SAMPLE_COUNT
    assert args.sample_seed == SAMPLE_SEED
    assert "outputs/figures/layerwise_representation_consistency" in str(
        args.layerwise_analysis
    )
    assert "outputs/figures/cross_modal_similarity_preservation" in str(args.output_dir)


def test_parser_rejects_unexpected_positional_arguments() -> None:
    from sparmoe_vl.studies.cross_modal_similarity_preservation.analysis import (
        parse_args,
    )

    with pytest.raises(SystemExit):
        parse_args(["unexpected"])


def test_config_explicitly_has_no_sparse_text_checkpoint() -> None:
    project = Path(__file__).resolve().parents[1]
    text = (project / "experiments" / "figures" / STUDY_NAME / "config.yaml").read_text(
        encoding="utf-8"
    )
    assert "source: frozen_dense_clip" in text
    assert "sparse_checkpoint: null" in text
    assert f"display_groups: {DISPLAY_GROUPS}" in text

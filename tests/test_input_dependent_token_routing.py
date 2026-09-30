import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from sparmoe_vl.studies.input_dependent_token_routing import analysis
from sparmoe_vl.studies.input_dependent_token_routing.analysis import (
    build_payload,
    extract_text_routes,
    extract_vision_routes,
    merge_word_groups,
    validate_analysis,
)
from sparmoe_vl.studies.input_dependent_token_routing.plotting import load_plot_data
from sparmoe_vl.studies.input_dependent_token_routing.protocol import (
    CAPACITY_FACTORS,
    DATA_SEED,
    IMAGE_IDENTITIES,
    NUM_EXPERTS,
    PAPER_SCOPE,
    PATCHES_PER_IMAGE,
    PATCH_GRID_SIZE,
    PROTOCOL,
    ROUTING_MODE,
    RUN_SEED,
    STUDY_NAME,
    TEXT_INPUT,
    TEXT_INPUT_SHA256,
    TEXT_LAYER,
    TEXT_TARGET_RATIO,
    TEXT_TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
    VISION_LAYERS,
    VISION_TARGET_RATIO,
    VISION_TRAINING_POOL_SHA256,
    protocol_manifest,
)


def checkpoint_metadata(modality: str) -> dict:
    target = VISION_TARGET_RATIO if modality == "vision" else TEXT_TARGET_RATIO
    pool_sha = (
        VISION_TRAINING_POOL_SHA256 if modality == "vision" else TEXT_TRAINING_POOL_SHA256
    )
    return {
        "format": "release_v2",
        "modality": modality,
        "target_ratio": target,
        "training_seed": RUN_SEED,
        "data_seed": DATA_SEED,
        "pool_size": TRAINING_POOL_SIZE,
        "dataset_sha256": pool_sha,
        "capacity_factors": list(CAPACITY_FACTORS),
        "checkpoint_step": 1,
        "checkpoint_sha256": ("1" if modality == "vision" else "2") * 64,
    }


def synthetic_routes() -> tuple[dict, dict]:
    assignments = {
        str(layer): np.zeros(
            (len(IMAGE_IDENTITIES), PATCH_GRID_SIZE, PATCH_GRID_SIZE),
            dtype=np.int64,
        ).tolist()
        for layer in VISION_LAYERS
    }
    vision = {
        "layers_one_based": list(VISION_LAYERS),
        "patch_grid": [PATCH_GRID_SIZE, PATCH_GRID_SIZE],
        "assignments_by_layer": assignments,
        "expert_usage_by_layer_and_image": {
            str(layer): [[1.0, 0.0, 0.0, 0.0] for _ in IMAGE_IDENTITIES]
            for layer in VISION_LAYERS
        },
        "actual_retention_ratios_by_layer": {
            str(layer): [0.4, 0.5, 0.6, 0.7] for layer in VISION_LAYERS
        },
    }
    text = {
        "layer_one_based": TEXT_LAYER,
        "eot_position": 3,
        "content_bpe_token_count_before_punctuation_filter": 2,
        "expert_usage_counts": [1, 1, 0, 0],
        "actual_retention_ratios": [0.4, 0.5, 0.6, 0.7],
        "words": [
            {
                "label": "token",
                "positions": [1, 2],
                "piece_labels": ["to", "ken"],
                "piece_routes": [0, 1],
            }
        ],
    }
    return vision, text


def complete_analysis() -> dict:
    vision, text = synthetic_routes()
    return build_payload(
        {
            "vision": checkpoint_metadata("vision"),
            "text": checkpoint_metadata("text"),
        },
        vision,
        text,
    )


class FakeTokenizer:
    pieces = {
        10: "hello</w>",
        11: ",</w>",
        12: "multi",
        13: "piece</w>",
    }

    def decode(self, values):
        return self.pieces.get(int(values[0]), "")


def test_registered_inputs_have_fixed_order_and_text_digest() -> None:
    manifest = protocol_manifest()
    assert manifest["study"] == STUDY_NAME
    assert manifest["paper_scope"] == PAPER_SCOPE
    assert manifest["routing"] == ROUTING_MODE
    assert [item["file_name"] for item in manifest["vision"]["images"]] == [
        value[0] for value in IMAGE_IDENTITIES
    ]
    assert hashlib.sha256(TEXT_INPUT.encode("utf-8")).hexdigest() == TEXT_INPUT_SHA256
    assert "assignments_by_layer" not in json.dumps(manifest)


def test_fixed_image_validator_rejects_changed_or_reordered_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    Image.new("RGB", (8, 8), "red").save(first)
    Image.new("RGB", (8, 8), "blue").save(second)
    first_sha = analysis.file_sha256(first)
    second_sha = analysis.file_sha256(second)
    monkeypatch.setattr(
        analysis,
        "IMAGE_IDENTITIES",
        ((first.name, first_sha), (second.name, second_sha)),
    )
    assert analysis.validate_fixed_inputs(tmp_path) == (first, second)
    first.write_bytes(second.read_bytes())
    with pytest.raises(ValueError, match="has changed"):
        analysis.validate_fixed_inputs(tmp_path)


def test_word_merging_and_text_route_alignment_preserve_bpe_positions() -> None:
    tokenizer = FakeTokenizer()
    token_ids = torch.tensor([[1, 10, 11, 12, 13, 99, 0]])
    groups = merge_word_groups(tokenizer, token_ids[0], 5)
    assert groups == [
        {"label": "hello", "positions": [1]},
        {"label": ",", "positions": [2]},
        {"label": "multipiece", "positions": [3, 4]},
    ]
    route_ids = torch.tensor([0, 3, 1, 2, 0, 0])
    layer = SimpleNamespace(
        transformer_layer=TEXT_LAYER - 1,
        routing=SimpleNamespace(gates=F.one_hot(route_ids, NUM_EXPERTS).float()),
    )
    output = SimpleNamespace(
        layers=(layer,),
        eos_positions=torch.tensor([5]),
        retention_ratios=torch.tensor([[0.4, 0.5, 0.6, 0.7]]),
    )
    result = extract_text_routes(output, tokenizer, token_ids)
    assert [word["label"] for word in result["words"]] == ["hello", "multipiece"]
    assert result["words"][1]["piece_routes"] == [1, 2]
    assert result["expert_usage_counts"] == [1, 1, 1, 0]


def test_visual_route_extraction_preserves_image_and_patch_axes() -> None:
    batch_size = len(IMAGE_IDENTITIES)
    layers = []
    for layer_number in VISION_LAYERS:
        route_ids = torch.arange(batch_size * PATCHES_PER_IMAGE) % NUM_EXPERTS
        layers.append(
            SimpleNamespace(
                transformer_layer=layer_number - 1,
                routing=SimpleNamespace(gates=F.one_hot(route_ids, NUM_EXPERTS).float()),
            )
        )
    output = SimpleNamespace(
        layers=tuple(layers),
        retention_ratios=torch.tensor([[0.4, 0.5, 0.6, 0.7] for _ in VISION_LAYERS]),
    )
    result = extract_vision_routes(output, batch_size)
    for layer in VISION_LAYERS:
        routes = np.asarray(result["assignments_by_layer"][str(layer)])
        assert routes.shape == (batch_size, PATCH_GRID_SIZE, PATCH_GRID_SIZE)
        assert np.allclose(
            result["expert_usage_by_layer_and_image"][str(layer)],
            np.full((batch_size, NUM_EXPERTS), 0.25),
        )


def test_analysis_validator_binds_routes_to_fixed_inputs() -> None:
    payload = complete_analysis()
    validate_analysis(payload, "analysis")
    payload["inputs"]["vision"]["images"] = list(
        reversed(payload["inputs"]["vision"]["images"])
    )
    with pytest.raises(ValueError, match="selection or order"):
        validate_analysis(payload, "analysis")


def test_analysis_validator_recomputes_usage_from_assignments() -> None:
    payload = complete_analysis()
    payload["vision"]["assignments_by_layer"][str(VISION_LAYERS[0])][0][0][0] = 1
    with pytest.raises(ValueError, match="usage disagrees"):
        validate_analysis(payload, "analysis")


def test_plot_loader_accepts_only_valid_analysis(tmp_path: Path) -> None:
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(complete_analysis()), encoding="utf-8")
    payload = load_plot_data(path)
    assert payload["protocol"] == PROTOCOL


def test_public_defaults_use_semantic_output_names() -> None:
    args = analysis.parse_args([])
    assert "outputs/figures/input_dependent_token_routing" in str(args.output)
    experiment = Path("experiments/figures/input_dependent_token_routing")
    assert all("figure" not in path.name.lower() for path in experiment.iterdir())

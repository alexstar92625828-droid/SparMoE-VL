import json
from pathlib import Path

import pytest
import torch
from torch import nn

from sparmoe_vl.architecture_transfer.clip.model import NestedCLIPFFN
from sparmoe_vl.common.two_stage import STAGE1_PROTOCOL
from sparmoe_vl.studies.routing_granularity_geometry_preservation import checkpoints
from sparmoe_vl.studies.routing_granularity_geometry_preservation.checkpoints import (
    checkpoint_metadata,
)
from sparmoe_vl.studies.routing_granularity_geometry_preservation.evaluation import (
    anti_matched_assignments,
    vision_macs,
)
from sparmoe_vl.studies.routing_granularity_geometry_preservation.plotting import (
    load_plot_data,
    validate_analysis,
)
from sparmoe_vl.studies.routing_granularity_geometry_preservation.protocol import (
    CAPACITY_FACTORS,
    DENSE_FFN_MACS_G,
    DENSE_TOTAL_MACS_G,
    EXPERT_COUNTS,
    COCO_ANNOTATIONS_SHA256,
    COCO_CAPTION_IMAGE_INDEX_SHA256,
    COCO_CAPTION_ORDER_SHA256,
    COCO_IMAGE_ORDER_SHA256,
    MODEL_KEY,
    MODEL_NAME,
    PROTOCOL,
    RUN_SEED,
    STUDY_NAME,
    TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
    capacity_factors,
    protocol_manifest,
)
from sparmoe_vl.studies.routing_granularity_geometry_preservation.training import (
    parse_args as parse_training_args,
)


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(4, 8)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(8, 4)

    def forward(self, inputs):
        return self.c_proj(self.gelu(self.c_fc(inputs)))


def release_checkpoint(expert_count: int) -> dict:
    return {
        "format_version": 3,
        "method": "sparmoe_vl_routing_granularity",
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "modality": "vision",
        "stage": 2,
        "training_protocol": "frozen_spg_token_router_stage2",
        "expert_count": expert_count,
        "capacity_factors": list(capacity_factors(expert_count)),
        "target_ratio": 0.7,
        "training_seed": RUN_SEED,
        "step": 100,
        "dataset": {
            "data_seed": 42,
            "samples": TRAINING_POOL_SIZE,
            "ordered_sha256": TRAINING_POOL_SHA256,
        },
        "train_args": {
            "steps": 5_000,
            "batch_size": 24,
            "max_samples": 500_000,
            "num_workers": 8,
            "seed": 42,
            "data_seed": 42,
            "weight_decay": 0.05,
            "router_warmup": 1_000,
            "log_every": 100,
            "random_evals": 3,
            "learning_rate": 3e-4,
            "temperature": 0.4,
        },
        "stage1": {
            "training_protocol": STAGE1_PROTOCOL,
            "expert_count": expert_count,
            "dataset_sha256": TRAINING_POOL_SHA256,
        },
    }


def analysis_payload() -> dict:
    experiments = {}
    for count in EXPERT_COUNTS:
        route = {
            "dense_cosine": 0.9,
            "mean_r1_retention": 95.0,
            "vision_ffn_macs_g": 32.0 + count / 10,
        }
        experiments[str(count)] = {
            "checkpoint": {
                "expert_count": count,
                "dataset_sha256": TRAINING_POOL_SHA256,
                "training_protocol": "frozen_spg_token_router_stage2",
            },
            "capacity_factors": list(capacity_factors(count)),
            "capacity_match_verification": {
                "per_layer_per_batch_multiset_checks": 1_896,
                "matched_checks": 1_896,
                "ffn_macs_absolute_difference_g": 0.0,
            },
            "learned": dict(route),
            "capacity_matched_shuffle": dict(route),
        }
    return {
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "run_seed": RUN_SEED,
        "control": {"name": "capacity_matched_shuffle"},
        "evaluation": {
            "dataset": "COCO-val2017",
            "images": 5_000,
            "captions": 25_014,
            "annotation_sha256": COCO_ANNOTATIONS_SHA256,
            "image_order_sha256": COCO_IMAGE_ORDER_SHA256,
            "caption_order_sha256": COCO_CAPTION_ORDER_SHA256,
            "caption_image_index_sha256": COCO_CAPTION_IMAGE_INDEX_SHA256,
        },
        "dense": {"vision_ffn_macs_g": DENSE_FFN_MACS_G},
        "experiments": experiments,
    }


def test_protocol_registers_all_granularities_on_the_main_500k_pool() -> None:
    manifest = protocol_manifest()
    assert manifest["expert_counts"] == [4, 6, 8, 10]
    assert manifest["training_data"]["samples"] == 500_000
    assert manifest["training_data"]["ordered_sha256"] == TRAINING_POOL_SHA256
    assert CAPACITY_FACTORS[4] == (0.7, 0.8, 0.9, 1.0)
    assert CAPACITY_FACTORS[10] == tuple(value / 10 for value in range(1, 11))
    assert manifest["training_protocols"]["4"] == "frozen_spg_token_router_stage2"
    assert manifest["training_protocols"]["8"] == "frozen_spg_token_router_stage2"
    assert manifest["training"]["stage2_objective"] == [
        "representation_preservation",
        "token_routing",
    ]
    assert manifest["training"]["stage2_trainable"] == ["token_router"]
    assert "spg" in manifest["training"]["stage2_frozen"]


def test_capacity_matched_reassignment_reverses_relation_and_preserves_counts() -> None:
    learned = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3])
    shuffled = anti_matched_assignments(learned)
    assert torch.equal(torch.bincount(learned), torch.bincount(shuffled))
    assert shuffled.tolist() == [3, 3, 2, 1, 1, 1, 0, 0]


def test_forced_router_replays_registered_expert_ids_exactly() -> None:
    layer = NestedCLIPFFN(ToyMLP(), 4, 8, (0.5, 0.75, 1.0), 0.7)
    states = torch.randn(1, 7, 4)
    embedding = torch.randn(3, 128)
    expected = torch.tensor([2, 0, 1, 2, 1, 0])
    layer.routing_mode = "forced"
    layer.forced_expert_ids = expected
    _, auxiliary = layer(states, embedding, 0.4, None)
    assert torch.equal(auxiliary["G"].argmax(-1), expected)
    layer.forced_expert_ids = expected[:-1]
    with pytest.raises(ValueError, match="one expert id"):
        layer(states, embedding, 0.4, None)


def test_paper_macs_match_dense_convention_at_full_width() -> None:
    macs = vision_macs([1.0] * 24)
    assert macs["ffn"] == pytest.approx(DENSE_FFN_MACS_G)
    assert macs["total"] == pytest.approx(DENSE_TOTAL_MACS_G)
    with pytest.raises(ValueError, match="24 layer ratios"):
        vision_macs([1.0] * 23)


def test_release_checkpoint_requires_exact_data_and_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checkpoints, "validate_controller_shapes", lambda *_, **__: None)
    metadata = checkpoint_metadata(release_checkpoint(6), 6)
    assert metadata["pool_size"] == 500_000
    assert metadata["levels"] == list(capacity_factors(6))
    changed = release_checkpoint(6)
    changed["dataset"]["samples"] = 50_000
    with pytest.raises(ValueError, match="dataset.samples"):
        checkpoint_metadata(changed, 6)


def test_release_checkpoint_requires_stage1_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checkpoints, "validate_controller_shapes", lambda *_, **__: None)
    checkpoint = release_checkpoint(8)
    del checkpoint["stage1"]
    with pytest.raises(ValueError, match="Stage-1 identity"):
        checkpoint_metadata(checkpoint, 8)


def test_plot_loader_rejects_partial_or_nonmatched_results(tmp_path: Path) -> None:
    payload = analysis_payload()
    validate_analysis(payload, "synthetic")
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    values = load_plot_data(path)
    assert values["experts"].tolist() == [4.0, 6.0, 8.0, 10.0]
    payload["experiments"]["4"]["capacity_match_verification"][
        "ffn_macs_absolute_difference_g"
    ] = 1.0
    with pytest.raises(ValueError, match="not compute matched"):
        validate_analysis(payload, "synthetic")


def test_public_training_entrypoints_lock_seed_data_and_batch_protocol() -> None:
    stage1 = parse_training_args(["--expert-count", "4"], phase="stage1")
    assert stage1.expert_count == 4
    assert stage1.batch_size == 32
    assert stage1.learning_rate == pytest.approx(1e-3)
    stage2 = parse_training_args(["--expert-count", "10"], phase="stage2")
    assert stage2.batch_size == 24
    assert stage2.max_samples == 500_000
    assert stage2.stage1_checkpoint is not None


def test_public_directory_has_semantic_names_and_no_measured_results() -> None:
    project = Path(__file__).resolve().parents[1]
    directory = project / "experiments" / "figures" / STUDY_NAME
    assert directory.is_dir()
    assert {path.name for path in directory.iterdir() if path.is_file()} == {
        "config.yaml",
        "train_stage1.py",
        "train_stage2.py",
        "evaluate.py",
        "plot.py",
        "run_training.sh",
        "run_study.sh",
    }
    files = [path for path in directory.iterdir() if path.is_file()]
    assert not any(path.suffix in {".json", ".csv", ".pt"} for path in files)
    assert all("figure" not in path.stem.lower() for path in files)

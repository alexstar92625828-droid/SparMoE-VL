import json
from pathlib import Path

import pytest
import torch
from torch import nn

from sparmoe_vl.architecture_transfer.clip.model import HyperNetwork, NestedCLIPFFN
from sparmoe_vl.common.two_stage import STAGE1_PROTOCOL, STAGE2_PROTOCOL
from sparmoe_vl.studies.capacity_intervention.checkpoints import checkpoint_metadata
from sparmoe_vl.studies.capacity_intervention.evaluation import activated_ffn_macs
from sparmoe_vl.studies.capacity_intervention.model import set_layer_allocation
from sparmoe_vl.studies.capacity_intervention.protocol import (
    ALLOCATIONS,
    CAPACITY_FACTORS,
    METRICS,
    MODEL_KEY,
    MODEL_NAME,
    STUDY_NAME,
    VISION_DATASET_SHA256,
    training_manifest,
)
from sparmoe_vl.studies.capacity_intervention.summary import summarize


def release_checkpoint(seed: int = 42) -> dict:
    return {
        "format_version": 3,
        "method": "sparmoe_vl_capacity_intervention",
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "step": 4_600,
        "training_seed": seed,
        "dataset": {
            "data_seed": 42,
            "samples": 500_000,
            "ordered_sha256": VISION_DATASET_SHA256,
        },
        "target_ratio": 0.7,
        "capacity_factors": list(CAPACITY_FACTORS),
        "training_protocol": STAGE2_PROTOCOL,
        "stage": 2,
        "modality": "vision",
        "controller": {"layers": {"0.router.weight": torch.empty(len(CAPACITY_FACTORS), 8)}},
        "stage1": {
            "training_protocol": STAGE1_PROTOCOL,
            "dataset_sha256": VISION_DATASET_SHA256,
        },
    }


def test_checkpoint_requires_the_exact_dense_to_n8_protocol():
    metadata = checkpoint_metadata(release_checkpoint())
    assert metadata["format"] == "release_v3"
    assert metadata["pool_size"] == 500_000
    assert metadata["levels"] == list(CAPACITY_FACTORS)
    changed = release_checkpoint()
    changed["capacity_factors"] = [0.7, 0.8, 0.9, 1.0]
    with pytest.raises(ValueError, match="eight registered capacities"):
        checkpoint_metadata(changed)


def test_hypernetwork_supports_all_eight_table7_capacities():
    network = HyperNetwork(8, 24)
    assert network().shape == (24, 8, 128)


def test_stage2_inherits_structure_and_trains_only_router() -> None:
    stage1 = training_manifest(42, 1)
    stage2 = training_manifest(42, 2)
    assert "global_budget_p" in stage1["objective"]
    assert "global_budget_p" not in stage2["objective"]
    assert stage1["trainable"] == [
        "spg",
        "reference_capacities",
        "nested_channel_subspaces",
    ]
    assert stage2["trainable"] == ["token_router"]
    assert "spg" in stage2["frozen"]
    assert "nested_channel_subspaces" in stage2["frozen"]


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(8, 16)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(16, 8)

    def forward(self, inputs):
        return self.c_proj(self.gelu(self.c_fc(inputs)))


def test_token_interventions_preserve_counts_and_uniformize_width():
    layer = NestedCLIPFFN(ToyMLP(), 8, 16, CAPACITY_FACTORS, 0.7)
    with torch.no_grad():
        layer.router.weight.copy_(torch.eye(8))
    layer.eval()
    states = torch.cat((torch.zeros(1, 1, 8), 5 * torch.eye(8).unsqueeze(0)), dim=1)
    embedding = torch.randn(8, 128)

    layer.routing_mode = "learned"
    _, learned = layer(states, embedding, 0.4, None)
    learned_ids = learned["G"].argmax(-1)
    assert torch.equal(learned_ids, torch.arange(8))

    layer.routing_mode = "shuffled"
    _, shuffled = layer(states, embedding, 0.4, None)
    shuffled_ids = shuffled["G"].argmax(-1)
    assert torch.equal(shuffled_ids, torch.arange(7, -1, -1))
    assert torch.equal(torch.sort(shuffled_ids).values, torch.sort(learned_ids).values)

    expected_width = int(torch.ceil(learned["masks"].float().mean(-1).mean() * 16))
    layer.routing_mode = "uniform"
    _, uniform = layer(states, embedding, 0.4, None)
    assert torch.equal(uniform["G"].argmax(-1), torch.zeros(8, dtype=torch.long))
    assert torch.equal(uniform["masks"], uniform["masks"][:1].expand_as(uniform["masks"]))
    assert int(uniform["masks"][0].sum()) == expected_width


class FakeInterventionModel:
    def __init__(self) -> None:
        self.layers = nn.ModuleDict({str(index): nn.Module() for index in range(3)})
        for index, layer in enumerate(self.layers.values(), start=1):
            layer.register_parameter(
                "full_ratio_logit",
                nn.Parameter(torch.tensor(float(index))),
            )


def test_layer_interventions_use_mean_and_descending_reassignment():
    model = FakeInterventionModel()
    original = torch.tensor([1.0, 2.0, 3.0])
    set_layer_allocation(model, "uniform", original)
    assert [float(layer.full_ratio_logit) for layer in model.layers.values()] == [2.0] * 3
    set_layer_allocation(model, "shuffled", original)
    assert [float(layer.full_ratio_logit) for layer in model.layers.values()] == [3.0, 2.0, 1.0]


def test_activated_macs_include_one_dense_cls_per_layer():
    expected = (24 + 256 * 24) * 2 * 1024 * 4096 / 1e9
    assert activated_ffn_macs([1.0] * 24) == pytest.approx(expected)
    with pytest.raises(ValueError, match="24 layer ratios"):
        activated_ffn_macs([1.0] * 23)


def result(seed: int, value: float) -> dict:
    completed = {
        allocation: {
            "Allocation": allocation,
            **{metric: value for metric in METRICS},
        }
        for allocation in ALLOCATIONS
    }
    return {
        "checkpoint_step": seed,
        "run_seed": seed,
        "data_seed": 42,
        "dataset_sha256": VISION_DATASET_SHA256,
        "images": 5_000,
        "completed": completed,
    }


def test_summary_uses_three_training_seeds_and_sample_sd(tmp_path: Path):
    paths = []
    for seed, value in zip((42, 123, 2026), (1.0, 2.0, 3.0)):
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(result(seed, value)), encoding="utf-8")
        paths.append(path)
    summary = summarize(paths)
    for table in ("token_allocation", "layer_allocation"):
        for allocation in summary[table].values():
            for metric in METRICS:
                assert allocation[metric]["mean"] == pytest.approx(2.0)
                assert allocation[metric]["sample_sd"] == pytest.approx(1.0)

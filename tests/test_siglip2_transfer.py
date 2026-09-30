import json
from pathlib import Path

import pytest
import torch
from torch import nn

from sparmoe_vl.architecture_transfer.siglip.model import SigLIPSparMoE
from sparmoe_vl.architecture_transfer.siglip2.checkpoints import checkpoint_metadata
from sparmoe_vl.architecture_transfer.siglip2.evaluation import dense_macs, sparse_macs
from sparmoe_vl.architecture_transfer.siglip2.model import (
    NestedFFN,
    SigLIP2SparMoE,
)
from sparmoe_vl.architecture_transfer.siglip2.protocol import (
    CAPACITY_FACTORS,
    MODEL_SPECS,
    TEXT_DATASET_SHA256,
    VISION_DATASET_SHA256,
    get_spec,
)
from sparmoe_vl.architecture_transfer.siglip2.summary import METRICS, summarize
from sparmoe_vl.common.two_stage import STAGE1_PROTOCOL, protocol_for_stage


EXPECTED_DENSE_MACS = {
    ("vit_b16", "vision"): (17.586487296, 11.098128384),
    ("vit_l16", "vision"): (81.000398848, 51.539607552),
    ("vit_b16", "text"): (5.511315456, 3.623878656),
    ("vit_l16", "text"): (19.528679424, 12.884901888),
}


@pytest.mark.parametrize("model,modality", EXPECTED_DENSE_MACS)
def test_dense_macs_match_result_generating_evaluators(model, modality):
    expected_total, expected_ffn = EXPECTED_DENSE_MACS[(model, modality)]
    actual = dense_macs(model, modality)
    assert actual["total_g"] == pytest.approx(expected_total)
    assert actual["ffn_g"] == pytest.approx(expected_ffn)
    all_dense = sparse_macs(model, modality, [1.0] * get_spec(model).num_layers)
    assert all_dense["ffn_g"] == pytest.approx(expected_ffn)
    assert all_dense["reduction_pct"] == pytest.approx(0.0)


def test_only_paper_siglip2_versions_are_registered():
    assert tuple(MODEL_SPECS) == ("vit_b16", "vit_l16")
    with pytest.raises(ValueError, match="unknown SigLIP2 transfer version"):
        get_spec("so400m14")


def test_registered_batches_are_result_generating_settings():
    b16 = get_spec("vit_b16")
    l16 = get_spec("vit_l16")
    assert (b16.vision_stage1_batch, b16.vision_stage2_batch) == (128, 96)
    assert (l16.vision_stage1_batch, l16.vision_stage2_batch) == (32, 24)
    assert (b16.text_micro_batch, b16.text_grad_accumulation) == (256, 1)
    assert (l16.text_micro_batch, l16.text_grad_accumulation) == (32, 4)


@pytest.mark.parametrize("model", MODEL_SPECS)
@pytest.mark.parametrize("modality,target", (("vision", 0.7), ("text", 0.6)))
@pytest.mark.parametrize("stage", (1, 2))
def test_release_checkpoint_requires_exact_shared_500k_pool(model, modality, target, stage):
    spec = get_spec(model)
    sha = VISION_DATASET_SHA256 if modality == "vision" else TEXT_DATASET_SHA256
    seed = 42
    checkpoint = {
        "format_version": 3,
        "method": "sparmoe_vl_siglip2_transfer",
        "step": 100,
        "training_seed": seed,
        "dataset": {
            "data_seed": 42,
            "samples": 500_000,
            "ordered_sha256": sha,
        },
        "model_key": spec.key,
        "model_name": spec.model_name,
        "target_ratio": target,
        "protocol": protocol_for_stage(stage),
        "stage": stage,
        "modality": modality,
        "moe_layers": list(range(spec.num_layers)),
        "n_experts": 4,
        "capacity_factors": list(CAPACITY_FACTORS),
        "train_args": {
            "steps": 5_000,
            "batch_size": spec.batch_size(modality, stage),
            "grad_accumulation": spec.grad_accumulation(modality),
            "lr": 1e-3 if stage == 1 else 3e-4,
            "weight_decay": 0.05,
            "tau": 0.4,
            "max_samples": 500_000,
            "num_workers": 8 if modality == "vision" else 4,
            "seed": seed,
            "data_seed": 42,
        },
    }
    if stage == 2:
        checkpoint["stage1"] = {
            "protocol": STAGE1_PROTOCOL,
            "dataset_sha256": sha,
            "training_seed": seed,
        }
    metadata = checkpoint_metadata(checkpoint, spec, modality, stage)
    assert metadata["pool_size"] == 500_000
    assert metadata["dataset_sha256"] == sha


class ToyVisionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(8, 4)

    def forward(self, inputs):
        return self.fc2(self.act(self.fc1(inputs)))


def test_siglip2_vision_routes_every_patch_without_a_cls_exception():
    torch.manual_seed(11)
    layer = NestedFFN(ToyVisionMLP(), model_dim=4, ffn_dim=8, levels=CAPACITY_FACTORS)
    inputs = torch.randn(2, 5, 4)
    embedding = torch.randn(4, 128)
    _, auxiliary = layer.vision_forward(inputs, embedding, tau=0.4)
    assert auxiliary["G"].shape[0] == 2 * 5


def test_siglip2_uses_the_verified_siglip_family_tower_contract():
    assert SigLIP2SparMoE is SigLIPSparMoE


def _result(seed: int, value: float) -> dict:
    spec = get_spec("vit_b16")
    return {
        "checkpoint_step": seed,
        "run_seed": seed,
        "data_seed": 42,
        "dataset_sha256": VISION_DATASET_SHA256,
        "protocol": "frozen_spg_token_router_stage2",
        "model_key": spec.key,
        "model_name": spec.model_name,
        "modality": "vision",
        "dense": {"total_macs_g": 1.0, "ffn_macs_g": 0.5},
        "sparse": {metric: value for metric in METRICS},
    }


def test_three_seed_summary_uses_sample_standard_deviation(tmp_path: Path):
    paths = []
    for seed, value in zip((42, 123, 2026), (1.0, 2.0, 3.0)):
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(_result(seed, value)), encoding="utf-8")
        paths.append(path)
    result = summarize(paths, "vit_b16", "vision")
    for metric in METRICS:
        assert result["sparse"][metric]["mean"] == pytest.approx(2.0)
        assert result["sparse"][metric]["std"] == pytest.approx(1.0)

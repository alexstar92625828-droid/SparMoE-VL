import json
from pathlib import Path

import pytest
import torch
from torch import nn

from sparmoe_vl.architecture_transfer.siglip.checkpoints import checkpoint_metadata
from sparmoe_vl.architecture_transfer.siglip.evaluation import dense_macs, sparse_macs
from sparmoe_vl.architecture_transfer.siglip.model import NestedFFN
from sparmoe_vl.architecture_transfer.siglip.protocol import (
    CAPACITY_FACTORS,
    MODEL_SPECS,
    PROJECT_ROOT,
    TEXT_DATASET_SHA256,
    VISION_DATASET_SHA256,
    get_spec,
)
from sparmoe_vl.architecture_transfer.siglip.summary import METRICS, summarize
from sparmoe_vl.common.two_stage import STAGE1_PROTOCOL, STAGE2_PROTOCOL


EXPECTED_DENSE_MACS = {
    ("vit_b16", "vision"): (17.586487296, 11.098128384),
    ("vit_l16", "vision"): (81.000398848, 51.539607552),
    ("so400m14", "vision"): (109.824049152, 68.542267392),
    ("vit_b16", "text"): (5.511315456, 3.623878656),
    ("vit_l16", "text"): (19.528679424, 12.884901888),
    ("so400m14", "text"): (6.593052672, 4.283891712),
}


@pytest.mark.parametrize("model,modality", EXPECTED_DENSE_MACS)
def test_dense_macs_match_result_generating_evaluators(model, modality):
    total, ffn = EXPECTED_DENSE_MACS[(model, modality)]
    actual = dense_macs(model, modality)
    assert actual["total_g"] == pytest.approx(total)
    assert actual["ffn_g"] == pytest.approx(ffn)
    all_dense = sparse_macs(
        model,
        modality,
        [1.0] * get_spec(model).num_layers,
    )
    assert all_dense["ffn_g"] == pytest.approx(ffn)
    assert all_dense["reduction_pct"] == pytest.approx(0.0)


def test_registered_batch_sizes_are_the_actual_three_seed_settings():
    b16, l16, so400m = (MODEL_SPECS[name] for name in ("vit_b16", "vit_l16", "so400m14"))
    assert (b16.vision_stage1_batch, b16.vision_stage2_batch) == (128, 96)
    assert (l16.vision_stage1_batch, l16.vision_stage2_batch) == (32, 24)
    assert (so400m.vision_stage1_batch, so400m.vision_stage2_batch) == (16, 12)
    assert (b16.text_micro_batch, b16.text_grad_accumulation) == (256, 1)
    assert (l16.text_micro_batch, l16.text_grad_accumulation) == (32, 4)
    assert (so400m.text_micro_batch, so400m.text_grad_accumulation) == (128, 2)
    assert so400m.image_size == 224
    assert so400m.num_patches == (so400m.image_size // so400m.patch_size) ** 2


def test_each_paper_version_has_an_independent_experiment_directory():
    experiment_root = PROJECT_ROOT / "experiments" / "architecture_transfer" / "siglip"
    expected_files = {
        "config.yaml",
        "train_stage1.py",
        "train_stage2.py",
        "evaluate.py",
        "summarize.py",
        "run_three_seeds.sh",
    }
    assert {path.name for path in experiment_root.iterdir() if path.is_dir()} == set(
        MODEL_SPECS
    )
    assert not [path for path in experiment_root.iterdir() if path.is_file()]
    for model in MODEL_SPECS:
        assert {
            path.name for path in (experiment_root / model).iterdir() if path.is_file()
        } == expected_files


@pytest.mark.parametrize("model", MODEL_SPECS)
@pytest.mark.parametrize("modality,target", (("vision", 0.7), ("text", 0.6)))
def test_release_checkpoint_identity(model, modality, target):
    spec = get_spec(model)
    expected_sha = VISION_DATASET_SHA256 if modality == "vision" else TEXT_DATASET_SHA256
    checkpoint = {
        "format_version": 3,
        "method": "sparmoe_vl_siglip_transfer",
        "step": 100,
        "training_seed": 42,
        "dataset": {
            "data_seed": 42,
            "samples": 500_000,
            "ordered_sha256": expected_sha,
        },
        "model_key": spec.key,
        "model_name": spec.model_name,
        "target_ratio": target,
        "capacity_factors": list(CAPACITY_FACTORS),
        "protocol": STAGE2_PROTOCOL,
        "stage": 2,
        "modality": modality,
        "train_args": {
            "steps": 5_000,
            "batch_size": spec.batch_size(modality, 2),
            "grad_accumulation": spec.grad_accumulation(modality),
            "lr": 3e-4,
            "weight_decay": 0.05,
            "tau": 0.4,
            "max_samples": 500_000,
            "num_workers": 8 if modality == "vision" else 4,
            "seed": 42,
            "data_seed": 42,
        },
        "stage1": {
            "protocol": STAGE1_PROTOCOL,
            "dataset_sha256": expected_sha,
            "training_seed": 42,
        },
    }
    metadata = checkpoint_metadata(checkpoint, spec, modality, 2)
    assert metadata["pool_size"] == 500_000
    assert metadata["dataset_sha256"] == expected_sha


class ToyTextMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(4, 8)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(8, 4)

    def forward(self, inputs):
        return self.c_proj(self.gelu(self.c_fc(inputs)))


def test_siglip_text_keeps_the_last_pooled_position_dense():
    torch.manual_seed(11)
    original = ToyTextMLP()
    layer = NestedFFN(original, model_dim=4, ffn_dim=8, levels=CAPACITY_FACTORS)
    layer.eval()
    inputs = torch.randn(2, 5, 4)
    embedding = torch.randn(4, 128)
    output, auxiliary = layer.text_forward(inputs, embedding, tau=0.4)
    assert torch.equal(output[:, -1:], original(inputs[:, -1:]))
    assert auxiliary["G"].shape[0] == 2 * (5 - 1)


def _result(seed, value):
    sparse = {metric: value for metric in METRICS}
    return {
        "checkpoint_step": seed,
        "run_seed": seed,
        "data_seed": 42,
        "dataset_sha256": VISION_DATASET_SHA256,
        "protocol": "frozen_spg_token_router_stage2",
        "model_key": get_spec("vit_b16").key,
        "model_name": get_spec("vit_b16").model_name,
        "modality": "vision",
        "dense": {"total_macs_g": 1.0, "ffn_macs_g": 0.5},
        "sparse": sparse,
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

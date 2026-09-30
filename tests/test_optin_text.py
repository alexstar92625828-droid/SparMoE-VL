import torch
from torch import nn

from sparmoe_vl.baselines.text.common import (
    EXPECTED_PAIRED_PATH_POOL_SHA256,
    EXPECTED_TEXT_POOL_SHA256,
    POOL_SIZE,
)
from sparmoe_vl.baselines.text.optin import (
    clip_output_kl,
    manifold_sampler_seed,
    official_manifold_cost,
    structurally_prune_pair,
    validate_data_manifest,
)


def test_optin_data_manifest_rejects_subsets() -> None:
    manifest = {
        "data_seed": 42,
        "pool_size": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "uses_complete_pool": True,
        "text_pool_sha256": EXPECTED_TEXT_POOL_SHA256,
        "paired_path_pool_sha256": EXPECTED_PAIRED_PATH_POOL_SHA256,
        "processing_seed": 42,
        "processing_order_sha256": "test-order",
        "batch_size": 32,
        "batches": 15_625,
    }
    validate_data_manifest(manifest)
    manifest["selected_samples"] = 32
    try:
        validate_data_manifest(manifest)
    except ValueError as error:
        assert "differs from the main pool" in str(error)
    else:
        raise AssertionError("a 32-sample OPTIN manifest was accepted")


def test_optin_costs_are_zero_for_identical_representations() -> None:
    trajectory = torch.randn(2, 4, 3)
    sampler = torch.arange(8)
    logits = torch.randn(2, 2)
    assert official_manifold_cost(trajectory, trajectory, sampler).item() == 0
    assert clip_output_kl(logits, logits).item() < 1e-6


def test_optin_sampler_includes_batch_identity() -> None:
    first = manifold_sampler_seed(42, 0, 1, 2, 3)
    assert first == manifold_sampler_seed(42, 0, 1, 2, 3)
    assert first != manifold_sampler_seed(42, 1, 1, 2, 3)


def test_optin_structural_pruning_slices_both_projections() -> None:
    c_fc = nn.Linear(3, 4, bias=True)
    c_proj = nn.Linear(4, 2, bias=True)
    kept = torch.tensor([0, 2])
    old_fc = c_fc.weight.detach().clone()
    old_proj = c_proj.weight.detach().clone()
    new_fc, new_proj = structurally_prune_pair(c_fc, c_proj, kept)
    assert tuple(new_fc.weight.shape) == (2, 3)
    assert tuple(new_proj.weight.shape) == (2, 2)
    assert torch.equal(new_fc.weight, old_fc[kept])
    assert torch.equal(new_proj.weight, old_proj[:, kept])

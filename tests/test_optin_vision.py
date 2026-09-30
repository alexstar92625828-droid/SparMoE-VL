from pathlib import Path

import torch
from torch import nn

from sparmoe_vl.baselines.vision.common import (
    D_FFN,
    EXPECTED_IMAGE_POOL_SHA256,
    EXPECTED_PROCESSING_ORDER_SHA256,
    EXPECTED_VISION_CAPTION_POOL_SHA256,
    N_LAYERS,
    N_TOKENS,
    POOL_SIZE,
)
from sparmoe_vl.baselines.vision.optin import (
    CHECKPOINT_METHOD,
    TOTAL_CANDIDATES,
    candidate_partition,
    clip_output_kl_batched,
    combine_official_scores,
    ffn_statistics,
    full_calibration_manifest,
    manifold_sampler_seed,
    merge_score_shards,
    official_manifold_cost_batched,
    save_score_shard,
    structurally_prune_pair,
    validate_checkpoint,
    validate_data_manifest,
)


def complete_manifest(seed: int = 42) -> dict:
    return {
        "data_seed": 42,
        "pool_size": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "paired_caption_sha256": EXPECTED_VISION_CAPTION_POOL_SHA256,
        "processing_seed": seed,
        "processing_order_sha256": EXPECTED_PROCESSING_ORDER_SHA256[seed],
        "batch_size": 32,
        "batches": 15_625,
        "selection": "all visual-main samples; processing seed changes order only",
    }


def test_visual_optin_uses_500k_not_one_batch() -> None:
    order, manifest = full_calibration_manifest(42, 32)
    assert order.numel() == 500_000
    assert torch.unique(order).numel() == 500_000
    assert manifest["selected_samples"] == 500_000
    assert manifest["batch_size"] == 32
    assert manifest["batches"] == 15_625
    validate_data_manifest(manifest)


def test_visual_optin_rejects_32_as_the_total_sample_count() -> None:
    manifest = complete_manifest()
    manifest["selected_samples"] = 32
    try:
        validate_data_manifest(manifest)
    except ValueError as error:
        assert "differs from the main experiment" in str(error)
    else:
        raise AssertionError("visual OPTIN accepted a one-batch calibration subset")


def test_visual_optin_candidate_partitions_are_exact() -> None:
    ranges = [candidate_partition(index, 7) for index in range(7)]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == TOTAL_CANDIDATES
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert sum(end - start for start, end in ranges) == N_LAYERS * D_FFN


def test_visual_optin_sampler_is_reproducible_per_full_pool_batch() -> None:
    first = manifold_sampler_seed(42, 0, 3, 17, 8)
    assert first == manifold_sampler_seed(42, 0, 3, 17, 8)
    assert first != manifold_sampler_seed(42, 1, 3, 17, 8)
    assert first != manifold_sampler_seed(123, 0, 3, 17, 8)


def test_visual_optin_losses_are_zero_for_identical_outputs() -> None:
    generator = torch.Generator().manual_seed(9)
    teacher = torch.randn(3, N_TOKENS, 8, generator=generator)
    students = teacher.unsqueeze(0).clone()
    samplers = torch.arange(768).unsqueeze(0)
    manifold = official_manifold_cost_batched(teacher, students, samplers)
    logits = torch.randn(3, 3, generator=generator)
    kl = clip_output_kl_batched(logits, logits.unsqueeze(0))
    assert torch.allclose(manifold, torch.zeros_like(manifold), atol=1e-8)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_visual_optin_preserves_official_final_layer_decay() -> None:
    mmd = torch.ones((N_LAYERS, D_FFN), dtype=torch.float64)
    kl = torch.zeros_like(mmd)
    combined, scalings = combine_official_scores(mmd, kl)
    assert torch.all(combined[:-1] == 1)
    assert torch.allclose(combined[-1], torch.full_like(combined[-1], 0.0001))
    assert torch.all(scalings == 1)


def test_visual_optin_structural_pruning_slices_both_projections() -> None:
    fc = nn.Linear(3, 5)
    proj = nn.Linear(5, 3)
    kept = torch.tensor([1, 4])
    expected_fc = fc.weight[kept].clone()
    expected_proj = proj.weight[:, kept].clone()
    new_fc, new_proj = structurally_prune_pair(fc, proj, kept)
    assert tuple(new_fc.weight.shape) == (2, 3)
    assert tuple(new_proj.weight.shape) == (3, 2)
    assert torch.equal(new_fc.weight, expected_fc)
    assert torch.equal(new_proj.weight, expected_proj)


def test_visual_optin_merge_requires_full_candidate_coverage(tmp_path: Path) -> None:
    manifest = complete_manifest()
    split = TOTAL_CANDIDATES // 2
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    save_score_shard(
        first,
        torch.ones(split),
        torch.ones(split),
        manifest,
        0,
        split,
    )
    save_score_shard(
        second,
        torch.ones(TOTAL_CANDIDATES - split),
        torch.ones(TOTAL_CANDIDATES - split),
        manifest,
        split,
        TOTAL_CANDIDATES,
    )
    raw_mmd, raw_kl, merged_manifest = merge_score_shards((second, first))
    assert tuple(raw_mmd.shape) == (N_LAYERS, D_FFN)
    assert tuple(raw_kl.shape) == (N_LAYERS, D_FFN)
    assert merged_manifest == manifest
    try:
        merge_score_shards((first,))
    except ValueError as error:
        assert "coverage" in str(error)
    else:
        raise AssertionError("visual OPTIN accepted incomplete candidate coverage")


def test_visual_optin_checkpoint_requires_complete_main_pool() -> None:
    hidden_sizes = [D_FFN] * N_LAYERS
    checkpoint = {
        "method": CHECKPOINT_METHOD,
        "complete": True,
        "data_manifest": complete_manifest(),
        "kept_indices": [torch.arange(D_FFN) for _ in range(N_LAYERS)],
        "hidden_sizes": hidden_sizes,
        "statistics": ffn_statistics(hidden_sizes),
        "visual_state_dict": {},
    }
    validate_checkpoint(checkpoint)
    checkpoint["data_manifest"]["selected_samples"] = 32
    try:
        validate_checkpoint(checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError("OPTIN checkpoint accepted an incomplete visual pool")

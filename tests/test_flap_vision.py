import torch
from torch import nn

from sparmoe_vl.baselines.vision.common import (
    D_FFN,
    EXPECTED_IMAGE_POOL_SHA256,
    EXPECTED_PROCESSING_ORDER_SHA256,
    N_LAYERS,
    N_TOKENS,
    POOL_SIZE,
)
from sparmoe_vl.baselines.vision.flap import (
    CHECKPOINT_METHOD,
    TARGET_FFN_REDUCTION,
    RunningFeatureMoments,
    ffn_statistics,
    full_calibration_manifest,
    prune_linear_pair_with_compensation,
    select_channels,
    validate_checkpoint,
    validate_data_manifest,
)


def complete_manifest(seed: int = 42) -> dict:
    return {
        "data_seed": 42,
        "pool_size": POOL_SIZE,
        "unique_samples": POOL_SIZE,
        "selected_samples": POOL_SIZE,
        "calibration_exposures": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "processing_seed": seed,
        "processing_order_sha256": EXPECTED_PROCESSING_ORDER_SHA256[seed],
        "batch_size": 128,
        "batches": 3_907,
        "final_batch_size": 32,
        "image_tokens_per_layer": POOL_SIZE * N_TOKENS,
        "selection": "every visual-main image exactly once in first-epoch order",
    }


def retained_indices_for_target() -> list[torch.Tensor]:
    remove_count = round(N_LAYERS * D_FFN * TARGET_FFN_REDUCTION)
    base, remainder = divmod(remove_count, N_LAYERS)
    return [torch.arange(base + (layer < remainder), D_FFN) for layer in range(N_LAYERS)]


def test_visual_flap_uses_every_main_pool_image_once() -> None:
    order, manifest = full_calibration_manifest(42, 128)
    assert order.numel() == 500_000
    assert torch.unique(order).numel() == 500_000
    assert manifest["unique_samples"] == 500_000
    assert manifest["calibration_exposures"] == 500_000
    assert manifest["batches"] == 3_907
    assert manifest["final_batch_size"] == 32
    validate_data_manifest(manifest)


def test_visual_flap_rejects_the_old_280k_exposure_protocol() -> None:
    manifest = complete_manifest()
    manifest["calibration_exposures"] = 280_000
    manifest["unique_samples"] = 160_000
    try:
        validate_data_manifest(manifest)
    except ValueError as error:
        assert "differs from the main experiment" in str(error)
    else:
        raise AssertionError("visual FLAP accepted the old partial calibration")


def test_visual_flap_streaming_moments_match_direct_statistics() -> None:
    values = torch.tensor(
        [
            [1.0, 3.0, -2.0],
            [2.0, 5.0, 4.0],
            [4.0, -1.0, 3.0],
            [8.0, 2.0, 1.0],
            [7.0, 0.0, -4.0],
            [3.0, 6.0, 2.0],
        ]
    )
    moments = RunningFeatureMoments(torch.device("cpu"), width=3)
    moments.add(values[:4].reshape(2, 2, 3))
    state = moments.state_dict()
    resumed = RunningFeatureMoments(torch.device("cpu"), width=3)
    resumed.load_state_dict(state)
    resumed.add(values[4:].reshape(1, 2, 3))
    mean, variance = resumed.finish()
    assert torch.allclose(mean, values.mean(dim=0), atol=1e-6)
    assert torch.allclose(variance, values.var(dim=0, correction=1), atol=1e-6)


def test_visual_flap_global_allocation_removes_the_exact_budget() -> None:
    base = torch.arange(D_FFN, dtype=torch.float64)
    raw = torch.stack([torch.roll(base, layer * 173) for layer in range(N_LAYERS)])
    kept, standardized, removed = select_channels(raw, TARGET_FFN_REDUCTION)
    assert removed == round(N_LAYERS * D_FFN * TARGET_FFN_REDUCTION)
    assert sum(indices.numel() for indices in kept) == N_LAYERS * D_FFN - removed
    assert all(indices.numel() > 0 for indices in kept)
    assert torch.allclose(
        standardized.mean(dim=1),
        torch.zeros(N_LAYERS, dtype=standardized.dtype),
        atol=1e-12,
    )


def test_visual_flap_bias_compensation_matches_removed_mean_output() -> None:
    fc = nn.Linear(2, 4)
    proj = nn.Linear(4, 2)
    with torch.no_grad():
        proj.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -2.0]]))
        proj.bias.copy_(torch.tensor([0.25, -0.5]))
    kept = torch.tensor([0, 2])
    mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = proj.weight[:, [1, 3]] @ mean[[1, 3]]
    new_fc, new_proj, compensation = prune_linear_pair_with_compensation(fc, proj, kept, mean)
    assert tuple(new_fc.weight.shape) == (2, 2)
    assert tuple(new_proj.weight.shape) == (2, 2)
    assert torch.allclose(compensation, expected)
    assert torch.allclose(new_proj.bias, proj.bias + expected)


def test_visual_flap_statistics_match_the_paper_budget() -> None:
    kept = retained_indices_for_target()
    result = ffn_statistics([indices.numel() for indices in kept])
    assert result["removed_channels"] == 35_026
    assert abs(result["ffn_reduction_percent"] - 35.630289713541664) < 1e-12
    assert abs(result["active_visual_parameters_m"] - 232.197934) < 1e-9


def test_visual_flap_checkpoint_requires_complete_500k_manifest() -> None:
    kept = retained_indices_for_target()
    hidden_sizes = [indices.numel() for indices in kept]
    checkpoint = {
        "method": CHECKPOINT_METHOD,
        "complete": True,
        "seed": 42,
        "target_ffn_reduction": TARGET_FFN_REDUCTION,
        "data_manifest": complete_manifest(),
        "kept_indices": kept,
        "hidden_sizes": hidden_sizes,
        "statistics": ffn_statistics(hidden_sizes),
        "visual_state_dict": {},
    }
    validate_checkpoint(checkpoint)
    checkpoint["data_manifest"]["selected_samples"] = 280_000
    try:
        validate_checkpoint(checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError("FLAP checkpoint accepted incomplete calibration data")

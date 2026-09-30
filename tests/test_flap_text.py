import torch

from sparmoe_vl.baselines.text.common import D_FFN, N_LAYERS, POOL_SIZE
from sparmoe_vl.baselines.text.flap import (
    EXPECTED_EXPOSURE_SHA256,
    STAGE_EXPOSURES,
    TOTAL_EXPOSURES,
    RunningFeatureMoments,
    full_exposure_manifest,
    select_channels,
    validate_data_manifest,
)


def test_flap_reconstructs_both_main_stage_sequences() -> None:
    indices, manifest = full_exposure_manifest(42)
    assert indices.numel() == TOTAL_EXPOSURES
    assert torch.equal(indices[:STAGE_EXPOSURES], indices[STAGE_EXPOSURES:])
    assert manifest["unique_samples"] == POOL_SIZE
    assert manifest["exposure_indices_sha256"] == EXPECTED_EXPOSURE_SHA256[42]


def test_flap_manifest_rejects_incomplete_main_pool() -> None:
    _, manifest = full_exposure_manifest(123)
    manifest["unique_samples"] = POOL_SIZE - 1
    try:
        validate_data_manifest(manifest)
    except ValueError as error:
        assert "differs from the main run" in str(error)
    else:
        raise AssertionError("an incomplete FLAP data pool was accepted")


def test_running_feature_moments_match_direct_statistics() -> None:
    first = torch.tensor([[[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]]])
    second = torch.tensor([[[3.0, 6.0, 12.0], [4.0, 8.0, 16.0]]])
    accumulator = RunningFeatureMoments(torch.device("cpu"), width=3)
    accumulator.add(first)
    accumulator.add(second)
    mean, variance = accumulator.finish()
    direct = torch.cat((first, second), dim=0).reshape(-1, 3)
    assert torch.allclose(mean, direct.mean(dim=0))
    assert torch.allclose(variance, direct.var(dim=0, correction=1))


def test_flap_global_selection_meets_exact_channel_budget() -> None:
    scores = torch.arange(N_LAYERS * D_FFN, dtype=torch.float32).reshape(
        N_LAYERS,
        D_FFN,
    )
    scores = scores + torch.arange(D_FFN, dtype=torch.float32).square()
    kept, standardized, removed = select_channels(scores, 0.45)
    assert tuple(standardized.shape) == (N_LAYERS, D_FFN)
    assert sum(indices.numel() for indices in kept) + removed == N_LAYERS * D_FFN
    assert removed == round(N_LAYERS * D_FFN * 0.45)

import torch

from sparmoe_vl.baselines.text.common import (
    D_FFN,
    EXPECTED_PRETRAINED_SHA256,
    N_LAYERS,
    POOL_SIZE,
)
from sparmoe_vl.baselines.text.mope import (
    EXPECTED_STAGE_EXPOSURE_SHA256,
    GROUP_SIZE,
    NUM_GROUPS,
    STAGE_EXPOSURES,
    TARGET_KEEP_GROUPS,
    kept_channel_indices,
    one_stage_exposure_indices,
    rankings_to_groups,
    recall_counts,
    recovery_data_manifest,
    selected_groups,
    statistics,
    structure_data_manifest,
    tensor_sha256,
    validate_recovery_data_manifest,
)


def test_mope_replays_one_identical_complete_sequence_per_stage() -> None:
    for seed, expected_sha256 in EXPECTED_STAGE_EXPOSURE_SHA256.items():
        stage1 = one_stage_exposure_indices(seed)
        stage2 = one_stage_exposure_indices(seed)
        assert stage1.numel() == STAGE_EXPOSURES
        assert torch.equal(stage1, stage2)
        assert torch.unique(stage1).numel() == POOL_SIZE
        assert tensor_sha256(stage1) == expected_sha256
        manifest = recovery_data_manifest(seed)
        assert manifest["stage_sequences_identical"] is True
        assert manifest["unique_samples"] == POOL_SIZE


def test_mope_rejects_any_smaller_data_pool() -> None:
    manifest = recovery_data_manifest(42)
    manifest["unique_samples"] = POOL_SIZE - 1
    try:
        validate_recovery_data_manifest(manifest)
    except ValueError as error:
        assert "differs from the text main run" in str(error)
    else:
        raise AssertionError("MoPE accepted an incomplete main-experiment pool")


def test_mope_structure_selection_uses_all_main_samples() -> None:
    manifest = structure_data_manifest()
    assert manifest["pool_size"] == POOL_SIZE
    assert manifest["selection_samples"] == POOL_SIZE
    assert manifest["held_out_samples"] == 0
    assert manifest["uses_complete_main_pool"] is True


def test_mope_ranked_groups_reconstruct_all_channels() -> None:
    scores = torch.arange(N_LAYERS * D_FFN, dtype=torch.float32).reshape(
        N_LAYERS,
        D_FFN,
    )
    rankings, groups = rankings_to_groups(scores)
    assert tuple(groups.shape) == (N_LAYERS, NUM_GROUPS, GROUP_SIZE)
    assert torch.equal(groups.reshape(N_LAYERS, D_FFN), rankings)
    for layer in range(N_LAYERS):
        assert torch.equal(torch.sort(rankings[layer]).values, torch.arange(D_FFN))


def test_mope_selection_applies_seed_specific_group_budget() -> None:
    selection = {
        "format_version": 1,
        "method": "MoPE-CLIP Text-FFN",
        "stage": "structure_selection",
        "structure_seed": 42,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "group_size": GROUP_SIZE,
        "num_groups": NUM_GROUPS,
        "data_manifest": structure_data_manifest(),
        "group_priority": list(range(NUM_GROUPS)),
        "groups": torch.arange(N_LAYERS * D_FFN).reshape(
            N_LAYERS,
            NUM_GROUPS,
            GROUP_SIZE,
        )
        % D_FFN,
        "complete": True,
    }
    for seed, count in TARGET_KEEP_GROUPS.items():
        retained_groups = selected_groups(selection, seed)
        retained_channels = kept_channel_indices(selection["groups"], retained_groups)
        assert len(retained_groups) == count
        assert all(indices.numel() == count * GROUP_SIZE for indices in retained_channels)
        assert statistics(count * GROUP_SIZE)["retained_width"] == count * GROUP_SIZE


def test_mope_batch_local_recall_counts_exact_matches() -> None:
    features = torch.eye(12)
    assert torch.equal(recall_counts(features, features), torch.tensor([12, 12, 12]))

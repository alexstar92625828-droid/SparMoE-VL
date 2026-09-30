import torch

from sparmoe_vl.baselines.vision.common import (
    D_FFN,
    EXPECTED_IMAGE_POOL_SHA256,
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_PROCESSING_ORDER_SHA256,
    EXPECTED_VISION_CAPTION_POOL_SHA256,
    N_LAYERS,
    POOL_SIZE,
    full_pool_permutation,
)
from sparmoe_vl.baselines.vision.mope import (
    CHECKPOINT_METHOD,
    GLOBAL_BATCH_SIZE,
    GROUP_SIZE,
    KEEP_GROUPS,
    NUM_GROUPS,
    RETAINED_WIDTH,
    STAGE_EXPOSURES,
    STEPS_PER_STAGE,
    TARGET_FFN_REDUCTION,
    TOTAL_EXPOSURES,
    kept_channel_indices,
    rankings_to_groups,
    recall_counts,
    recovery_data_manifest,
    recovery_rank_indices,
    selected_groups,
    statistics,
    structure_data_manifest,
    validate_final_checkpoint,
    validate_recovery_data_manifest,
    validate_selection,
)


def complete_selection() -> dict:
    scores = torch.arange(N_LAYERS * D_FFN, dtype=torch.float32).reshape(
        N_LAYERS,
        D_FFN,
    )
    rankings, groups = rankings_to_groups(scores)
    records = [
        {
            "group": group,
            "mope": float(group),
            "ablated_recall": {"r1": 0.0, "r5": 0.0, "r10": 0.0, "mean": 0.0},
        }
        for group in range(NUM_GROUPS)
    ]
    return {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": "structure_selection",
        "structure_seed": 42,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "group_size": GROUP_SIZE,
        "num_groups": NUM_GROUPS,
        "data_manifest": structure_data_manifest(),
        "taylor_scores": scores,
        "rankings": rankings,
        "groups": groups,
        "group_records": records,
        "group_priority": list(reversed(range(NUM_GROUPS))),
        "complete": True,
    }


def test_visual_mope_selection_uses_the_exact_500k_main_pool() -> None:
    manifest = structure_data_manifest()
    assert manifest["selection_samples"] == POOL_SIZE
    assert manifest["unique_samples"] == POOL_SIZE
    assert manifest["held_out_samples"] == 0
    assert manifest["dataset_sha256"] == EXPECTED_IMAGE_POOL_SHA256
    assert manifest["paired_caption_sha256"] == EXPECTED_VISION_CAPTION_POOL_SHA256
    assert manifest["batches"] == 15_625


def test_visual_mope_recovery_replays_all_500k_pairs_in_both_stages() -> None:
    for seed, order_sha256 in EXPECTED_PROCESSING_ORDER_SHA256.items():
        manifest = recovery_data_manifest(seed)
        assert manifest["stage1_exposures"] == STAGE_EXPOSURES == POOL_SIZE
        assert manifest["stage2_exposures"] == STAGE_EXPOSURES == POOL_SIZE
        assert manifest["total_exposures"] == TOTAL_EXPOSURES == 1_000_000
        assert manifest["stage_sequences_identical"] is True
        assert manifest["stage_processing_order_sha256"] == order_sha256
        assert manifest["global_batch_size"] == GLOBAL_BATCH_SIZE == 256
        assert manifest["steps_per_stage"] == STEPS_PER_STAGE == 1_954
        assert manifest["final_global_batch_size"] == 32


def test_visual_mope_rank_shards_reconstruct_the_registered_order() -> None:
    expected = full_pool_permutation(42)
    reconstructed = torch.empty_like(expected)
    for rank in range(8):
        shard = recovery_rank_indices(42, rank)
        assert shard.numel() == 62_500
        reconstructed[rank::8] = shard
    assert torch.equal(reconstructed, expected)
    assert torch.unique(reconstructed).numel() == POOL_SIZE


def test_visual_mope_rejects_old_partial_recovery_data() -> None:
    manifest = recovery_data_manifest(42)
    manifest["stage1_exposures"] = 280_064
    manifest["stage2_exposures"] = 280_064
    manifest["unique_samples"] = 280_064
    try:
        validate_recovery_data_manifest(manifest)
    except ValueError as error:
        assert "differs from the visual main pool" in str(error)
    else:
        raise AssertionError("visual MoPE accepted the old 280,064-pair recovery subset")


def test_visual_mope_groups_reconstruct_every_ffn_channel() -> None:
    scores = torch.arange(N_LAYERS * D_FFN, dtype=torch.float32).reshape(
        N_LAYERS,
        D_FFN,
    )
    rankings, groups = rankings_to_groups(scores)
    assert tuple(groups.shape) == (N_LAYERS, NUM_GROUPS, GROUP_SIZE)
    assert torch.equal(groups.reshape(N_LAYERS, D_FFN), rankings)
    for layer in range(N_LAYERS):
        assert torch.equal(torch.sort(rankings[layer]).values, torch.arange(D_FFN))


def test_visual_mope_selection_retains_exact_paper_budget() -> None:
    selection = complete_selection()
    validate_selection(selection)
    retained_groups = selected_groups(selection)
    retained_channels = kept_channel_indices(selection["groups"], retained_groups)
    assert retained_groups == list(range(NUM_GROUPS - KEEP_GROUPS, NUM_GROUPS))
    assert all(indices.numel() == RETAINED_WIDTH for indices in retained_channels)
    assert RETAINED_WIDTH == 2_624
    assert TARGET_FFN_REDUCTION == 0.359375


def test_visual_mope_batch_local_recall_counts_exact_matches() -> None:
    features = torch.eye(12)
    assert torch.equal(recall_counts(features, features), torch.tensor([12, 12, 12]))


def test_visual_mope_statistics_match_the_measured_checkpoint() -> None:
    result = statistics()
    assert result["retained_width"] == 2_624
    assert result["removed_channels"] == 35_328
    assert abs(result["active_visual_parameters_m"] - 231.579136) < 1e-9
    assert abs(result["visual_total_macs_g"] - 62.41837056) < 1e-9
    assert abs(result["visual_ffn_macs_g"] - 33.146535936) < 1e-9
    assert result["ffn_reduction_percent"] == 35.9375


def test_visual_mope_final_checkpoint_requires_both_complete_stages() -> None:
    checkpoint = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": 2,
        "stage_step": STEPS_PER_STAGE,
        "global_step": 2 * STEPS_PER_STAGE,
        "seed": 42,
        "data_manifest": recovery_data_manifest(42),
        "selection_sha256": "a" * 64,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "kept_group_indices": list(range(KEEP_GROUPS)),
        "retained_width": RETAINED_WIDTH,
        "target_ffn_reduction": TARGET_FFN_REDUCTION,
        "actual_ffn_reduction": TARGET_FFN_REDUCTION,
        "statistics": statistics(),
        "visual_state_dict": {},
        "complete": True,
    }
    validate_final_checkpoint(checkpoint)
    checkpoint["data_manifest"]["stage2_exposures"] = 280_064
    try:
        validate_final_checkpoint(checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError("visual MoPE accepted an incomplete Stage 2")

from types import SimpleNamespace

import torch
from torch import nn

from sparmoe_vl.baselines.vision.common import (
    D_FFN,
    D_MODEL,
    EXPECTED_IMAGE_POOL_SHA256,
    EXPECTED_PROCESSING_ORDER_SHA256,
    N_LAYERS,
    N_TOKENS,
    POOL_SIZE,
    full_pool_permutation,
    tensor_sha256,
)
from sparmoe_vl.baselines.vision.teal import (
    METHOD,
    TEALVisualMLP,
    activity_statistics,
    histogram_thresholds_from_counts,
    sparsify_patch_tokens,
    validate_checkpoint,
    validate_data_manifest,
)


def complete_manifest(seed: int = 42) -> dict:
    return {
        "data_seed": 42,
        "pool_size": POOL_SIZE,
        "unique_records": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "selected_samples": POOL_SIZE,
        "processing_seed": seed,
        "processing_order_sha256": EXPECTED_PROCESSING_ORDER_SHA256[seed],
        "patch_tokens_per_image": N_TOKENS - 1,
        "calibration_patch_tokens": POOL_SIZE * (N_TOKENS - 1),
    }


def test_visual_teal_processes_every_main_pool_image() -> None:
    for seed, fingerprint in EXPECTED_PROCESSING_ORDER_SHA256.items():
        indices = full_pool_permutation(seed)
        assert indices.numel() == POOL_SIZE
        assert torch.unique(indices).numel() == POOL_SIZE
        assert tensor_sha256(indices) == fingerprint
        validate_data_manifest(complete_manifest(seed))


def test_visual_teal_rejects_a_small_calibration_subset() -> None:
    manifest = complete_manifest()
    manifest["selected_samples"] = 80
    try:
        validate_data_manifest(manifest)
    except ValueError as error:
        assert "differs from the main experiment" in str(error)
    else:
        raise AssertionError("visual TEAL accepted an 80-image calibration subset")


def test_visual_teal_keeps_cls_dense() -> None:
    values = torch.tensor([[[0.1, -0.2], [0.1, 0.8], [-0.7, 0.2]]])
    padding = torch.zeros((1, N_TOKENS - values.shape[1], 2))
    values = torch.cat((values, padding), dim=1)
    sparse, kept, total = sparsify_patch_tokens(values, threshold=0.5, sparsity=0.5)
    assert torch.equal(sparse[:, 0], values[:, 0])
    assert torch.equal(sparse[:, 1], torch.tensor([[0.0, 0.8]]))
    assert kept == 2
    assert total == (N_TOKENS - 1) * 2


def test_visual_teal_histogram_thresholds_follow_quantiles() -> None:
    counts = torch.tensor([1.0, 2.0, 3.0, 4.0])
    thresholds = histogram_thresholds_from_counts(counts, 4.0, (0.0, 0.3, 0.6))
    assert thresholds == {0.0: 0.0, 0.3: 2.0, 0.6: 3.0}


def test_visual_teal_wrapper_counts_only_patch_activity() -> None:
    mlp = SimpleNamespace(
        c_fc=nn.Linear(3, 4),
        gelu=nn.GELU(),
        c_proj=nn.Linear(4, 3),
    )
    schedule = {
        "layer": 0,
        "fc_sparsity": 0.5,
        "fc_threshold": 0.2,
        "proj_sparsity": 0.5,
        "proj_threshold": 0.1,
    }
    wrapper = TEALVisualMLP(mlp, schedule)
    with torch.inference_mode():
        output = wrapper(torch.randn(2, N_TOKENS, 3))
    assert tuple(output.shape) == (2, N_TOKENS, 3)
    assert wrapper.images_seen == 2
    assert wrapper.fc_total == 2 * (N_TOKENS - 1) * 3
    assert wrapper.proj_total == 2 * (N_TOKENS - 1) * 4


def test_visual_teal_activity_uses_all_24_layers() -> None:
    images = 2
    wrappers = []
    for _ in range(N_LAYERS):
        fc_total = images * (N_TOKENS - 1) * D_MODEL
        proj_total = images * (N_TOKENS - 1) * D_FFN
        wrappers.append(
            SimpleNamespace(
                images_seen=images,
                fc_kept=fc_total // 2,
                fc_total=fc_total,
                proj_kept=proj_total // 2,
                proj_total=proj_total,
            )
        )
    activity = activity_statistics(wrappers)
    expected_reduction = 100.0 * 0.5 * (N_TOKENS - 1) / N_TOKENS
    assert abs(activity.reduction_percent - expected_reduction) < 1e-12


def test_visual_teal_checkpoint_requires_complete_main_pool() -> None:
    checkpoint = {
        "format_version": 1,
        "method": METHOD,
        "stage": "calibrated",
        "complete": True,
        "calibration_manifest": complete_manifest(),
        "layers": [{"layer": layer} for layer in range(N_LAYERS)],
    }
    validate_checkpoint(checkpoint)
    checkpoint["calibration_manifest"]["pool_size"] -= 1
    try:
        validate_checkpoint(checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError("TEAL checkpoint accepted a non-main data pool")

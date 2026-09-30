import torch

from sparmoe_vl.baselines.text.teal import (
    effective_sparsity,
    histogram_thresholds_from_counts,
    sparsify_tokens,
)


def test_teal_thresholds_follow_histogram_quantiles() -> None:
    counts = torch.tensor([2.0, 3.0, 4.0, 1.0])
    thresholds = histogram_thresholds_from_counts(counts, 4.0, (0.0, 0.5, 0.9))
    assert thresholds == {0.0: 0.0, 0.5: 2.0, 0.9: 3.0}


def test_teal_sparsification_and_effective_budget() -> None:
    values = torch.tensor([-2.0, -0.5, 0.0, 1.5])
    sparse, kept, total = sparsify_tokens(values, threshold=1.0, sparsity=0.5)
    assert torch.equal(sparse, torch.tensor([-2.0, 0.0, 0.0, 1.5]))
    assert (kept, total) == (2, 4)
    assert effective_sparsity({"fc": 0.4, "proj": 0.5}) == 0.45

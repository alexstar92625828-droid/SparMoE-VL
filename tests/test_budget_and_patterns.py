import torch

from sparmoe_vl.common.budgets import LayerAdaptiveBudget
from sparmoe_vl.common.sparse_patterns import SparsePatternGenerator


def test_nested_masks_and_target_budget() -> None:
    budget = LayerAdaptiveBudget(3, 0.7, (0.7, 0.8, 0.9, 1.0))
    ratios = budget()
    assert ratios.shape == (3, 4)
    assert torch.allclose(budget.base_ratios().mean(), torch.tensor(0.7), atol=1e-5)

    generator = SparsePatternGenerator(
        num_layers=3,
        ffn_dims=32,
        num_capacity_levels=4,
        embedding_dim=8,
        latent_dim=4,
        hyper_hidden_dim=4,
    )
    patterns = generator.all_layers(ratios)
    for pattern in patterns:
        assert torch.all(pattern.masks[:-1] <= pattern.masks[1:])
        assert torch.all(pattern.retained_channels[:-1] < pattern.retained_channels[1:])

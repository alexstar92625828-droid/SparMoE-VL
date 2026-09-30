"""Layer-adaptive FFN capacity allocation for SparMoE-VL."""

from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn


def inverse_sigmoid(value: Tensor, eps: float = 1e-4) -> Tensor:
    """Return a numerically stable logit."""

    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value / (1.0 - value))


class LayerAdaptiveBudget(nn.Module):
    """Learn one base FFN retention ratio for each Transformer layer.

    The base ratio ``alpha_l`` is multiplied by fixed, increasing capacity
    factors ``rho_n``.  The resulting ``alpha_l * rho_n`` values define the
    nested expert widths used by the sparse pattern generator.
    """

    def __init__(
        self,
        num_layers: int,
        target_ratio: float,
        capacity_factors: Sequence[float] = (0.7, 0.8, 0.9, 1.0),
        initial_ratios: Optional[Union[float, Sequence[float], Tensor]] = None,
        min_base_ratio: float = 0.02,
        max_base_ratio: float = 0.995,
        min_retention_ratio: float = 0.01,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 < target_ratio <= 1:
            raise ValueError("target_ratio must lie in (0, 1]")
        if not 0 < min_base_ratio < max_base_ratio <= 1:
            raise ValueError("invalid base-ratio bounds")
        if not 0 < min_retention_ratio < 1:
            raise ValueError("min_retention_ratio must lie in (0, 1)")

        factors = torch.as_tensor(capacity_factors, dtype=torch.float32)
        if factors.ndim != 1 or factors.numel() < 2:
            raise ValueError("capacity_factors must contain at least two values")
        if torch.any(factors <= 0) or torch.any(factors > 1):
            raise ValueError("capacity factors must lie in (0, 1]")
        if torch.any(factors[1:] <= factors[:-1]):
            raise ValueError("capacity factors must be strictly increasing")
        if not torch.isclose(factors[-1], torch.tensor(1.0)):
            raise ValueError("the largest capacity factor must equal 1.0")

        self.num_layers = int(num_layers)
        self.num_capacity_levels = int(factors.numel())
        self.min_base_ratio = float(min_base_ratio)
        self.max_base_ratio = float(max_base_ratio)
        self.min_retention_ratio = float(min_retention_ratio)

        self.register_buffer("target_ratio", torch.tensor(float(target_ratio)))
        self.register_buffer("capacity_factors", factors)

        if initial_ratios is None:
            initial = torch.full((num_layers,), float(target_ratio))
        elif isinstance(initial_ratios, (float, int)):
            initial = torch.full((num_layers,), float(initial_ratios))
        else:
            initial = torch.as_tensor(initial_ratios, dtype=torch.float32)
            if tuple(initial.shape) != (num_layers,):
                raise ValueError(
                    f"initial_ratios must have shape ({num_layers},), "
                    f"got {tuple(initial.shape)}"
                )
        if torch.any(initial <= 0) or torch.any(initial >= 1):
            raise ValueError("initial ratios must lie strictly between 0 and 1")
        self.base_ratio_logits = nn.Parameter(inverse_sigmoid(initial))

    def base_ratios(self) -> Tensor:
        """Return layer-wise base ratios ``alpha`` with shape ``[L]``."""

        return torch.sigmoid(self.base_ratio_logits).clamp(
            self.min_base_ratio,
            self.max_base_ratio,
        )

    def forward(self, layer_index: Optional[int] = None) -> Tensor:
        """Return retention ratios for one layer or all layers.

        Returns ``[N]`` when ``layer_index`` is provided and ``[L, N]``
        otherwise.
        """

        base = self.base_ratios()
        ratios = (base[:, None] * self.capacity_factors[None, :]).clamp(
            self.min_retention_ratio,
            self.max_base_ratio,
        )
        if layer_index is None:
            return ratios
        self._validate_layer_index(layer_index)
        return ratios[layer_index]

    def retained_channels(
        self,
        ffn_dims: Union[int, Sequence[int], Tensor],
    ) -> Tensor:
        """Return exact rounded expert widths with shape ``[L, N]``."""

        if isinstance(ffn_dims, int):
            dims = torch.full(
                (self.num_layers,),
                int(ffn_dims),
                device=self.base_ratio_logits.device,
            )
        else:
            dims = torch.as_tensor(
                ffn_dims,
                device=self.base_ratio_logits.device,
            )
            if tuple(dims.shape) != (self.num_layers,):
                raise ValueError(
                    f"ffn_dims must have shape ({self.num_layers},), got {tuple(dims.shape)}"
                )
        if torch.any(dims <= 0):
            raise ValueError("all FFN dimensions must be positive")

        return torch.round(self() * dims[:, None]).long().clamp_min(1)

    def _validate_layer_index(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index must be in [0, {self.num_layers}), got {layer_index}"
            )

    def extra_repr(self) -> str:
        factors = ", ".join(f"{factor:.2f}" for factor in self.capacity_factors)
        return (
            f"num_layers={self.num_layers}, "
            f"target_ratio={float(self.target_ratio):.3f}, "
            f"capacity_factors=({factors})"
        )

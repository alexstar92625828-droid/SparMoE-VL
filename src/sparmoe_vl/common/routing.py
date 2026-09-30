"""Token-level capacity routing shared by vision and text encoders."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class TokenRoutingOutput:
    """Router tensors for a flattened collection of routed tokens."""

    logits: Tensor
    probabilities: Tensor
    gates: Tensor


def straight_through_gumbel_softmax(
    logits: Tensor,
    temperature: float = 0.4,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample hard one-hot gates with softmax straight-through gradients."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logits.ndim < 2:
        raise ValueError("logits must have a capacity-level dimension")

    work_logits = logits.float()
    uniform = torch.rand(
        work_logits.shape,
        dtype=work_logits.dtype,
        device=work_logits.device,
        generator=generator,
    ).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(uniform))
    soft = F.softmax((work_logits + gumbel) / temperature, dim=-1)
    indices = soft.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
    return (hard - soft).detach() + soft


def ffn_output_contribution_scores(
    hidden_activations: Tensor,
    output_weight: Tensor,
) -> Tensor:
    """Compute the frozen-FFN response proxy used for capacity supervision.

    Args:
        hidden_activations: Post-activation FFN states with final dimension
            ``d_ff``.
        output_weight: Frozen FFN output projection with shape
            ``[d_model, d_ff]``.

    Returns:
        One scalar capacity score per input token.
    """

    if hidden_activations.ndim < 2:
        raise ValueError("hidden_activations must have shape [..., d_ff]")
    if output_weight.ndim != 2:
        raise ValueError("output_weight must have shape [d_model, d_ff]")
    if hidden_activations.shape[-1] != output_weight.shape[1]:
        raise ValueError("hidden activation width must match the output projection width")

    with torch.no_grad():
        channel_strength = output_weight.detach().float().norm(dim=0)
        scores = hidden_activations.detach().abs().float() @ channel_strength
    return scores


def balanced_capacity_targets(
    capacity_scores: Tensor,
    num_capacity_levels: int,
    valid_mask: Optional[Tensor] = None,
    ignore_index: int = -100,
) -> Tensor:
    """Convert response scores into balanced ordinal capacity pseudo-labels.

    Only valid tokens participate in ranking.  Invalid positions receive
    ``ignore_index`` and are excluded from routing supervision and statistics.
    """

    if num_capacity_levels < 2:
        raise ValueError("num_capacity_levels must be at least 2")

    scores = capacity_scores.detach().reshape(-1)
    if valid_mask is None:
        valid = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
    else:
        valid = valid_mask.to(device=scores.device, dtype=torch.bool).reshape(-1)
        if valid.numel() != scores.numel():
            raise ValueError("valid_mask must match capacity_scores")

    valid_indices = valid.nonzero(as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        raise ValueError("at least one valid token is required")

    order_within_valid = torch.argsort(scores[valid_indices], descending=False)
    ranked_indices = valid_indices[order_within_valid]
    ranks = torch.arange(ranked_indices.numel(), device=scores.device)
    labels = torch.div(
        ranks * num_capacity_levels,
        ranked_indices.numel(),
        rounding_mode="floor",
    ).clamp_max(num_capacity_levels - 1)

    targets = torch.full(
        (scores.numel(),),
        fill_value=ignore_index,
        dtype=torch.long,
        device=scores.device,
    )
    targets[ranked_indices] = labels.long()
    return targets.reshape(capacity_scores.shape)


class TokenCapacityRouter(nn.Module):
    """Select one nested FFN capacity level for each routed token.

    ``largest`` is the structure-learning path: every token uses the largest
    candidate subspace without consulting or training the router.
    """

    VALID_MODES = frozenset({"largest", "learned", "random"})

    def __init__(
        self,
        model_dim: int,
        num_capacity_levels: int = 4,
        temperature: float = 0.4,
    ) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if num_capacity_levels < 2:
            raise ValueError("num_capacity_levels must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.model_dim = int(model_dim)
        self.num_capacity_levels = int(num_capacity_levels)
        self.temperature = float(temperature)
        self.projection = nn.Linear(model_dim, num_capacity_levels, bias=False)

    def forward(
        self,
        token_states: Tensor,
        mode: str = "learned",
        generator: Optional[torch.Generator] = None,
    ) -> TokenRoutingOutput:
        """Route flattened token states with shape ``[M, d_model]``."""

        if token_states.ndim != 2 or token_states.shape[-1] != self.model_dim:
            raise ValueError(
                f"token_states must have shape [M, {self.model_dim}], "
                f"got {tuple(token_states.shape)}"
            )
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")

        logits = self.projection(token_states)
        probabilities = F.softmax(logits.float(), dim=-1)

        if mode == "largest":
            indices = torch.full(
                (token_states.shape[0],),
                self.num_capacity_levels - 1,
                dtype=torch.long,
                device=token_states.device,
            )
            gates = F.one_hot(indices, self.num_capacity_levels).float()
        elif mode == "random":
            indices = torch.randint(
                self.num_capacity_levels,
                (token_states.shape[0],),
                device=token_states.device,
                generator=generator,
            )
            gates = F.one_hot(indices, self.num_capacity_levels).float()
        elif self.training:
            gates = straight_through_gumbel_softmax(
                logits,
                temperature=self.temperature,
                generator=generator,
            )
        else:
            indices = logits.argmax(dim=-1)
            gates = F.one_hot(indices, self.num_capacity_levels).float()

        return TokenRoutingOutput(
            logits=logits,
            probabilities=probabilities,
            gates=gates,
        )

"""Training objectives for the two-stage SparMoE-VL protocol.

Stage 1 learns the nested FFN subspaces and their layer-wise reference
capacities under a global budget. Stage 2 freezes that complete structure and
optimizes only token routing with representation preservation and routing
supervision. Structural budget and separation values remain diagnostics, not
Stage-2 loss terms.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class LayerRoutingLossInput:
    """Routing tensors and supervision for one Transformer layer."""

    logits: Tensor
    targets: Tensor
    gates: Tensor
    valid_mask: Optional[Tensor] = None


@dataclass(frozen=True)
class Stage2RouterLossOutput:
    """Stage-2 router-only loss plus inherited-structure diagnostics."""

    total: Tensor
    distillation: Tensor
    routing: Tensor
    inherited_budget_error: Tensor
    inherited_separation: Tensor
    router_accuracy: Tensor
    base_ratio_mean: Tensor
    expected_ratio_mean: Tensor
    expert_usage: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "total": self.total,
            "distillation": self.distillation,
            "routing": self.routing,
            "inherited_budget_error": self.inherited_budget_error,
            "inherited_separation": self.inherited_separation,
            "router_accuracy": self.router_accuracy,
            "base_ratio_mean": self.base_ratio_mean,
            "expected_ratio_mean": self.expected_ratio_mean,
            "expert_usage": self.expert_usage,
        }


@dataclass(frozen=True)
class Stage1SubspaceLossOutput:
    """Stage-1 nested-subspace losses and statistics."""

    total: Tensor
    distillation: Tensor
    budget: Tensor
    separation: Tensor
    base_ratio_mean: Tensor
    layer_ratios: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "total": self.total,
            "distillation": self.distillation,
            "budget": self.budget,
            "separation": self.separation,
            "base_ratio_mean": self.base_ratio_mean,
            "layer_ratios": self.layer_ratios,
        }


def linear_warmup_weight(step: int, warmup_steps: int = 1000) -> float:
    """Return the routing-loss warm-up used by the main experiments."""

    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step) / float(warmup_steps))


def symmetric_log_ratio(left: Tensor, right: Tensor, eps: float = 1e-8) -> Tensor:
    """Return a non-negative, symmetric multiplicative distance."""

    left = left.clamp_min(eps)
    right = right.clamp_min(eps)
    return torch.log(torch.maximum(left, right) / torch.minimum(left, right))


def representation_preservation_loss(
    sparse_features: Tensor,
    dense_features: Tensor,
) -> Tensor:
    """Cosine representation loss against the frozen dense encoder."""

    if sparse_features.shape != dense_features.shape:
        raise ValueError("sparse and dense features must have identical shapes")
    if sparse_features.ndim != 2:
        raise ValueError("feature tensors must have shape [batch, embedding_dim]")
    return (
        1.0
        - F.cosine_similarity(
            sparse_features.float(),
            dense_features.detach().float(),
            dim=-1,
        ).mean()
    )


def budget_constraint_loss(base_ratios: Tensor, target_ratio: Tensor) -> Tensor:
    """Constrain the mean layer-wise base capacity to the global target."""

    if base_ratios.ndim != 1 or base_ratios.numel() == 0:
        raise ValueError("base_ratios must be a non-empty one-dimensional tensor")
    target = target_ratio.to(device=base_ratios.device, dtype=base_ratios.dtype)
    return symmetric_log_ratio(base_ratios.mean(), target)


def capacity_separation_loss(
    retention_ratios: Tensor,
    minimum_margin: float = 0.03,
) -> Tensor:
    """Keep adjacent nested capacity levels sufficiently separated."""

    if retention_ratios.ndim != 2 or retention_ratios.shape[1] < 2:
        raise ValueError("retention_ratios must have shape [layers, levels>=2]")
    if minimum_margin < 0:
        raise ValueError("minimum_margin must be non-negative")
    gaps = retention_ratios[:, 1:] - retention_ratios[:, :-1]
    return F.relu(minimum_margin - gaps).mean()


class Stage1SubspaceObjective(nn.Module):
    """Learn nested subspaces and layer capacities under global budget ``p``."""

    def __init__(
        self,
        target_budget_ratio: float,
        distillation_weight: float = 100.0,
        budget_weight: float = 50.0,
        separation_weight: float = 1.0,
        minimum_capacity_margin: float = 0.03,
    ) -> None:
        super().__init__()
        if not 0 < target_budget_ratio <= 1:
            raise ValueError("target_budget_ratio must lie in (0, 1]")
        if minimum_capacity_margin < 0:
            raise ValueError("minimum_capacity_margin must be non-negative")
        weights = (distillation_weight, budget_weight, separation_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("stage-1 loss weights must be non-negative")
        self.distillation_weight = float(distillation_weight)
        self.budget_weight = float(budget_weight)
        self.separation_weight = float(separation_weight)
        self.minimum_capacity_margin = float(minimum_capacity_margin)
        self.register_buffer(
            "target_budget_ratio",
            torch.tensor(float(target_budget_ratio)),
        )

    def forward(
        self,
        sparse_features: Tensor,
        dense_features: Tensor,
        base_ratios: Tensor,
        retention_ratios: Tensor,
    ) -> Stage1SubspaceLossOutput:
        if retention_ratios.ndim != 2:
            raise ValueError("retention_ratios must have shape [layers, levels]")
        if base_ratios.shape != retention_ratios.shape[:1]:
            raise ValueError("base_ratios must contain one value per layer")

        distillation = representation_preservation_loss(
            sparse_features,
            dense_features,
        )
        budget = budget_constraint_loss(
            base_ratios,
            self.target_budget_ratio,
        )
        separation = capacity_separation_loss(
            retention_ratios,
            self.minimum_capacity_margin,
        )
        total = (
            self.distillation_weight * distillation
            + self.budget_weight * budget
            + self.separation_weight * separation
        )
        return Stage1SubspaceLossOutput(
            total=total,
            distillation=distillation,
            budget=budget,
            separation=separation,
            base_ratio_mean=base_ratios.mean(),
            layer_ratios=base_ratios,
        )


class Stage2RouterObjective(nn.Module):
    """Optimize only token routers over the frozen Stage-1 expert space."""

    def __init__(
        self,
        target_budget_ratio: float,
        distillation_weight: float = 100.0,
        minimum_capacity_margin: float = 0.03,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if not 0 < target_budget_ratio <= 1:
            raise ValueError("target_budget_ratio must lie in (0, 1]")
        if minimum_capacity_margin < 0:
            raise ValueError("minimum_capacity_margin must be non-negative")
        if distillation_weight < 0:
            raise ValueError("stage-2 loss weights must be non-negative")

        self.distillation_weight = float(distillation_weight)
        self.minimum_capacity_margin = float(minimum_capacity_margin)
        self.ignore_index = int(ignore_index)
        self.register_buffer(
            "target_budget_ratio",
            torch.tensor(float(target_budget_ratio)),
        )

    def forward(
        self,
        sparse_features: Tensor,
        dense_features: Tensor,
        routing_layers: Sequence[LayerRoutingLossInput],
        base_ratios: Tensor,
        retention_ratios: Tensor,
        routing_weight: float = 1.0,
    ) -> Stage2RouterLossOutput:
        """Compute the router-only loss over a frozen nested expert space.

        ``routing_layers`` and rows of ``retention_ratios`` must use the same
        Transformer-layer order.  Vision passes all patch tokens as valid;
        text passes a mask selecting positions 1 through EOT, including EOT.

        Representation preservation and routing supervision update only the
        token router. The Stage-1 budget and separation measurements are
        detached diagnostics: including them in the Stage-2 objective would
        either be constant for frozen structure or violate parameter isolation.
        """

        if routing_weight < 0:
            raise ValueError("routing_weight must be non-negative")
        if retention_ratios.ndim != 2:
            raise ValueError("retention_ratios must have shape [layers, levels]")
        if len(routing_layers) != retention_ratios.shape[0]:
            raise ValueError("routing_layers must match the number of retention-ratio rows")
        if base_ratios.shape != retention_ratios.shape[:1]:
            raise ValueError("base_ratios must contain one value per layer")

        distillation = representation_preservation_loss(
            sparse_features,
            dense_features,
        )
        routing, accuracy, usage, expected = self._routing_terms(
            routing_layers,
            retention_ratios,
        )
        inherited_budget_error = budget_constraint_loss(
            base_ratios.detach(), self.target_budget_ratio
        )
        inherited_separation = capacity_separation_loss(
            retention_ratios.detach(), self.minimum_capacity_margin
        )

        total = self.distillation_weight * distillation + float(routing_weight) * routing
        return Stage2RouterLossOutput(
            total=total,
            distillation=distillation,
            routing=routing,
            inherited_budget_error=inherited_budget_error,
            inherited_separation=inherited_separation,
            router_accuracy=accuracy,
            base_ratio_mean=base_ratios.detach().mean(),
            expected_ratio_mean=expected,
            expert_usage=usage,
        )

    def _routing_terms(
        self,
        routing_layers: Sequence[LayerRoutingLossInput],
        retention_ratios: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not routing_layers:
            raise ValueError("at least one routed layer is required")

        num_levels = retention_ratios.shape[1]
        routing_losses = []
        accuracies = []
        usages = []
        expected_ratios = []

        for layer_index, layer in enumerate(routing_layers):
            logits = layer.logits
            gates = layer.gates
            targets = layer.targets.reshape(-1).long()
            if logits.ndim != 2 or logits.shape[1] != num_levels:
                raise ValueError(
                    f"layer {layer_index} logits must have shape [tokens, {num_levels}]"
                )
            if gates.shape != logits.shape:
                raise ValueError(f"layer {layer_index} gates must match logits")
            if targets.numel() != logits.shape[0]:
                raise ValueError(f"layer {layer_index} targets must match logits")

            valid = targets != self.ignore_index
            if layer.valid_mask is not None:
                supplied_mask = layer.valid_mask.to(
                    device=logits.device,
                    dtype=torch.bool,
                ).reshape(-1)
                if supplied_mask.numel() != logits.shape[0]:
                    raise ValueError(f"layer {layer_index} valid_mask must match logits")
                valid = valid & supplied_mask
            if not torch.any(valid):
                raise ValueError(f"layer {layer_index} has no valid routed tokens")

            valid_targets = targets[valid]
            if torch.any(valid_targets < 0) or torch.any(valid_targets >= num_levels):
                raise ValueError(f"layer {layer_index} targets are out of range")

            routing_losses.append(F.cross_entropy(logits[valid].float(), valid_targets))
            accuracies.append((logits[valid].argmax(dim=-1) == valid_targets).float().mean())
            with torch.no_grad():
                usage = gates[valid].float().mean(dim=0)
                usages.append(usage)
                expected_ratios.append(
                    (usage * retention_ratios[layer_index].detach().float()).sum()
                )

        return (
            torch.stack(routing_losses).mean(),
            torch.stack(accuracies).mean(),
            torch.stack(usages).mean(dim=0),
            torch.stack(expected_ratios).mean(),
        )

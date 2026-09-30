"""Canonical two-stage contract shared by every SparMoE-VL experiment.

Stage 1 optimizes only the sparse-pattern generator (SPG), including channel
importance, layer-wise reference capacities, and nested expert subspaces,
under the global budget ``p``. Stage 2 initializes from and freezes that
complete structure, then optimizes only the token router. The global budget is
therefore inherited as a structural upper bound rather than re-optimized in
Stage 2.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn


TWO_STAGE_PROTOCOL = "spg_global_budget_then_frozen_spg_token_router"
STAGE1_PROTOCOL = "spg_global_budget_stage1"
STAGE2_PROTOCOL = "frozen_spg_token_router_stage2"

STAGE1_LEARNS = (
    "sparse_pattern_generator",
    "channel_importance",
    "layer_reference_capacities",
    "nested_expert_subspaces",
)
STAGE1_FROZEN = ("pretrained_backbone", "token_router")
STAGE2_LEARNS = ("token_router",)
STAGE2_FROZEN = ("pretrained_backbone", *STAGE1_LEARNS)


def protocol_for_stage(stage: int) -> str:
    """Return the checkpoint protocol identifier for one training stage."""

    if stage == 1:
        return STAGE1_PROTOCOL
    if stage == 2:
        return STAGE2_PROTOCOL
    raise ValueError("stage must be 1 or 2")


def validated_trainable_parameters(
    model: nn.Module,
    stage: int,
    *,
    structure_parameters: Iterable[nn.Parameter],
    router_parameters: Iterable[nn.Parameter],
) -> tuple[nn.Parameter, ...]:
    """Fail fast unless the model obeys the stage-specific parameter contract.

    The optimizer receives the complete structure set in Stage 1 and only the
    router set in Stage 2. Comparing parameter identities here protects every
    launcher, including architecture transfer, downstream transfer, and
    ablation workflows, from accidentally updating the frozen Stage-1
    structure.
    """

    protocol_for_stage(stage)
    structure = tuple(structure_parameters)
    routers = tuple(router_parameters)
    structure_ids = {id(parameter) for parameter in structure}
    router_ids = {id(parameter) for parameter in routers}
    overlap = structure_ids & router_ids
    if overlap:
        raise RuntimeError("structure and token-router parameter sets overlap")

    expected_ids = structure_ids if stage == 1 else router_ids
    named = tuple(model.named_parameters())
    name_by_id = {id(parameter): name for name, parameter in named}
    model_ids = set(name_by_id)
    outside_model = expected_ids - model_ids
    if outside_model:
        raise RuntimeError("the stage contract contains parameters outside the model")

    actual = tuple(parameter for _, parameter in named if parameter.requires_grad)
    actual_ids = {id(parameter) for parameter in actual}
    missing = sorted(name_by_id[index] for index in expected_ids - actual_ids)
    unexpected = sorted(name_by_id[index] for index in actual_ids - expected_ids)
    if missing or unexpected:
        raise RuntimeError(
            f"Stage {stage} violates parameter isolation: "
            f"missing_trainable={missing}, unexpected_trainable={unexpected}"
        )
    if not actual:
        raise RuntimeError(f"Stage {stage} has no trainable parameters")
    return actual


__all__ = [
    "STAGE1_FROZEN",
    "STAGE1_LEARNS",
    "STAGE1_PROTOCOL",
    "STAGE2_FROZEN",
    "STAGE2_LEARNS",
    "STAGE2_PROTOCOL",
    "TWO_STAGE_PROTOCOL",
    "protocol_for_stage",
    "validated_trainable_parameters",
]

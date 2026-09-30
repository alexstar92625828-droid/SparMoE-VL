"""N=8 CLIP vision model and allocation controls used by Table 7."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ...architecture_transfer.clip.model import (
    CLIPSparMoE,
    build_model as build_clip_model,
    controller_state,
    initialize_stage2 as initialize_clip_stage2,
    load_stage2_controller,
)
from .protocol import CAPACITY_FACTORS, TARGET_RATIO


def build_model(
    clip_model: nn.Module, tau: float = 0.4, *, training_stage: int = 2
) -> CLIPSparMoE:
    """Build the result-generating nested N=8 visual encoder."""

    return build_clip_model(
        clip_model,
        modality="vision",
        stage=training_stage,
        target_ratio=TARGET_RATIO,
        levels=CAPACITY_FACTORS,
        tau=tau,
    )


def initialize_stage2(model: CLIPSparMoE, checkpoint: Mapping[str, Any]) -> list[float]:
    """Load Stage-1 SPG/subspaces without loading any router parameters."""

    return initialize_clip_stage2(model, checkpoint)


def load_controller(model: CLIPSparMoE, checkpoint: Mapping[str, Any]) -> None:
    """Load either a genuine experiment weight or a compact cleaned weight."""

    load_stage2_controller(model, checkpoint)


def layer_base_logits(model: CLIPSparMoE) -> Tensor:
    return torch.stack(
        [layer.full_ratio_logit.detach().clone() for layer in model.layers.values()]
    )


@torch.no_grad()
def set_layer_allocation(
    model: CLIPSparMoE,
    allocation: str,
    original_logits: Tensor,
) -> None:
    """Apply the exact Self, Uniform, or descending reassignment intervention."""

    if tuple(original_logits.shape) != (len(model.layers),):
        raise ValueError("original layer logits do not match the visual tower")
    if allocation == "self":
        assigned = original_logits
    elif allocation == "uniform":
        assigned = original_logits.mean().expand_as(original_logits)
    elif allocation == "shuffled":
        assigned = torch.sort(original_logits, descending=True).values
    else:
        raise ValueError("layer allocation must be 'self', 'uniform', or 'shuffled'")
    for value, layer in zip(assigned, model.layers.values()):
        layer.full_ratio_logit.copy_(value)


__all__ = [
    "CLIPSparMoE",
    "build_model",
    "controller_state",
    "initialize_stage2",
    "layer_base_logits",
    "load_controller",
    "set_layer_allocation",
]

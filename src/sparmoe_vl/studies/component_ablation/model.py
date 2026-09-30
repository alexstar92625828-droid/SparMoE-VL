"""Exact model replacements used by the Table-8 component ablations."""

from __future__ import annotations

import types
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...architecture_transfer.clip.model import (
    CLIPSparMoE,
    build_model as build_clip_model,
    controller_state as base_controller_state,
    initialize_stage2 as initialize_clip_stage2,
    load_stage2_controller,
)
from .protocol import CAPACITY_FACTORS, TARGET_RATIO, TRAINED_METHODS


def _logit(value: Tensor) -> Tensor:
    value = value.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(value / (1.0 - value))


def _independent_budget_masks(
    self: nn.Module,
    embedding: Tensor,
    tau: float = 0.4,
) -> tuple[Tensor, Tensor]:
    del embedding
    ratios = (self.full_ratio() * self.level_factors).clamp(0.01, 0.995)
    masks = []
    for expert_index, ratio in enumerate(ratios):
        logits = self.independent_mask_logits[expert_index]
        width = max(
            1,
            min(self.ffn_dim, int(round(float(ratio.detach()) * self.ffn_dim))),
        )
        detached_threshold = logits.detach().topk(width).values[-1]
        adjusted = logits - detached_threshold + (_logit(ratio) - _logit(ratio.detach()))
        soft = torch.sigmoid(adjusted / tau)
        hard = (logits >= detached_threshold).to(soft.dtype)
        if self.training:
            hardness = float(getattr(self, "mask_hardness", 1.0))
            mask = soft + hardness * (hard - soft).detach()
        else:
            mask = hard
        masks.append(mask)
    return torch.stack(masks), ratios


def enable_independent_expert_masks(
    model: CLIPSparMoE, seed: int, *, training_stage: int = 1
) -> list[nn.Parameter]:
    """Replace SPG with four independently learned masks in every layer."""

    parameters = []
    for layer_index, layer in enumerate(model.layers.values()):
        first = layer.original_mlp.c_fc.weight.detach().float()
        second = layer.original_mlp.c_proj.weight.detach().float()
        magnitude = first.norm(dim=1) * second.norm(dim=0)
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-6)
        generator = torch.Generator(device=magnitude.device)
        generator.manual_seed(int(seed) + layer_index)
        noise = (
            torch.randn(
                layer.num_budgets,
                layer.ffn_dim,
                generator=generator,
                device=magnitude.device,
                dtype=magnitude.dtype,
            )
            * 0.02
        )
        layer.register_parameter(
            "independent_mask_logits",
            nn.Parameter(magnitude.unsqueeze(0).repeat(layer.num_budgets, 1) + noise),
        )
        layer.independent_mask_logits.requires_grad_(training_stage == 1)
        layer.mask_hardness = 0.0
        layer.budget_masks = types.MethodType(_independent_budget_masks, layer)
        for parameter in layer.proj_mlp_d.parameters():
            parameter.requires_grad_(False)
        parameters.append(layer.independent_mask_logits)
    for parameter in model.hypernetwork.parameters():
        parameter.requires_grad_(False)
    return parameters


def freeze_layer_adaptive_budget(model: CLIPSparMoE) -> None:
    fixed = _logit(torch.tensor(TARGET_RATIO))
    for layer in model.layers.values():
        with torch.no_grad():
            layer.full_ratio_logit.copy_(fixed.to(layer.full_ratio_logit.device))
        layer.full_ratio_logit.requires_grad_(False)


def build_model(
    clip_model: nn.Module,
    method: str,
    seed: int,
    tau: float = 0.4,
    *,
    training_stage: int = 2,
) -> tuple[CLIPSparMoE, tuple[nn.Parameter, ...]]:
    if method not in TRAINED_METHODS:
        raise ValueError(f"{method} is not a separately trained Table-8 ablation")
    model = build_clip_model(
        clip_model,
        modality="vision",
        stage=training_stage,
        target_ratio=TARGET_RATIO,
        levels=CAPACITY_FACTORS,
        tau=tau,
    )
    extra: Sequence[nn.Parameter] = ()
    if method == "without_spg":
        extra = enable_independent_expert_masks(model, seed, training_stage=training_stage)
    elif method == "without_layer_adaptive_budget":
        freeze_layer_adaptive_budget(model)
    # Each ablation intentionally replaces or removes part of the normal SPG
    # structure. Register its exact Stage-1 structural parameter set. Stage 2
    # freezes every replacement and optimizes only the router.
    router_ids = {
        id(parameter)
        for layer in model.layers.values()
        for parameter in layer.router.parameters()
    }
    model.register_structure_parameters(
        tuple(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in router_ids
        )
    )
    return model, tuple(extra)


@torch.no_grad()
def initialize_stage2(
    model: CLIPSparMoE,
    checkpoint: Mapping[str, Any],
    method: str,
) -> list[float]:
    """Load a variant's Stage-1 structure while leaving routers untouched."""

    ratios = initialize_clip_stage2(model, checkpoint)
    if method == "without_spg":
        controller = checkpoint.get("controller")
        layers = controller.get("layers") if isinstance(controller, Mapping) else None
        if not isinstance(layers, Mapping):
            raise ValueError("w/o SPG Stage-1 checkpoint is missing mask state")
        for name, layer in model.layers.items():
            layer.independent_mask_logits.copy_(layers[f"{name}.independent_mask_logits"])
            layer.independent_mask_logits.requires_grad_(False)
    return ratios


def controller_state(model: CLIPSparMoE, method: str) -> dict[str, Mapping[str, Tensor]]:
    state = base_controller_state(model)
    layers = dict(state["layers"])
    if method == "without_spg":
        for name, layer in model.layers.items():
            layers[f"{name}.independent_mask_logits"] = layer.independent_mask_logits.detach()
    return {"hypernetwork": state["hypernetwork"], "layers": layers}


def load_controller(
    model: CLIPSparMoE,
    checkpoint: Mapping[str, Any],
    method: str,
) -> None:
    load_stage2_controller(model, checkpoint)
    if method != "without_spg":
        return
    controller = checkpoint.get("controller")
    layers = (
        controller.get("layers")
        if isinstance(controller, Mapping)
        else checkpoint.get("budget_mlps")
    )
    if not isinstance(layers, Mapping):
        raise ValueError("w/o SPG checkpoint is missing independent mask state")
    with torch.no_grad():
        for name, layer in model.layers.items():
            key = f"{name}.independent_mask_logits"
            if key not in layers:
                raise ValueError(f"w/o SPG checkpoint is missing {key}")
            layer.independent_mask_logits.copy_(layers[key])


def _structural_statistics(
    model: CLIPSparMoE,
    auxiliary: Sequence[Mapping[str, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    router = torch.stack(
        [
            F.cross_entropy(item["router_logits"].float(), item["router_targets"])
            for item in auxiliary
        ]
    ).mean()
    accuracy = torch.stack(
        [
            (item["router_logits"].argmax(-1) == item["router_targets"]).float().mean()
            for item in auxiliary
        ]
    ).mean()
    full = torch.stack([item["full_ratio"] for item in auxiliary])
    target = torch.tensor(model.target_ratio, device=full.device)
    base = torch.log(
        torch.maximum(full.mean(), target) / torch.minimum(full.mean(), target).clamp_min(1e-8)
    )
    gaps = torch.stack(
        [(item["ratios"][1:] - item["ratios"][:-1]).mean() for item in auxiliary]
    )
    spread = F.relu(0.03 - gaps).mean()
    usage = torch.stack([item["G"].float().mean(0) for item in auxiliary]).mean(0)
    expected = torch.stack(
        [(item["G"].float().mean(0) * item["ratios"]).sum() for item in auxiliary]
    )
    return router, base, spread, accuracy, full, usage, expected


def distillation_loss(
    model: CLIPSparMoE,
    sparse: Tensor,
    dense: Tensor,
    auxiliary: Sequence[Mapping[str, Tensor]],
    training_stage: int,
    router_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Any]]:
    distance = 1.0 - F.cosine_similarity(sparse, dense.detach(), dim=-1).mean()
    router, base, spread, accuracy, full, usage, expected = _structural_statistics(
        model, auxiliary
    )
    if training_stage == 1:
        loss = 100.0 * distance + 50.0 * base + spread
    elif training_stage == 2:
        loss = 100.0 * distance + router_weight * router
    else:
        raise ValueError("training_stage must be 1 or 2")
    return loss, {
        "total": float(loss.detach()),
        "distill": float(distance.detach()),
        "Rrouter": float(router.detach()),
        "router_acc": float(accuracy.detach()),
        "structure_budget_error": float(base.detach()),
        "structure_separation": float(spread.detach()),
        "base_mean": float(full.detach().mean()),
        "actual_mean": float(expected.detach().mean()),
        "usage": usage.detach().cpu().tolist(),
        "full_ratios": full.detach().cpu().tolist(),
    }


def contrastive_loss(image_features: Tensor, text_features: Tensor, scale: Tensor) -> Tensor:
    logits = scale * image_features @ text_features.T
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def geometry_replacement_loss(
    model: CLIPSparMoE,
    sparse: Tensor,
    text: Tensor,
    auxiliary: Sequence[Mapping[str, Tensor]],
    scale: Tensor,
    training_stage: int,
    router_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Any]]:
    contrastive = contrastive_loss(sparse, text, scale)
    router, base, spread, accuracy, full, usage, expected = _structural_statistics(
        model, auxiliary
    )
    if training_stage == 1:
        loss = contrastive + 50.0 * base + spread
    elif training_stage == 2:
        loss = contrastive + router_weight * router
    else:
        raise ValueError("training_stage must be 1 or 2")
    return loss, {
        "total": float(loss.detach()),
        "contrastive": float(contrastive.detach()),
        "Rrouter": float(router.detach()),
        "router_acc": float(accuracy.detach()),
        "structure_budget_error": float(base.detach()),
        "structure_separation": float(spread.detach()),
        "base_mean": float(full.detach().mean()),
        "actual_mean": float(expected.detach().mean()),
        "usage": usage.detach().cpu().tolist(),
    }

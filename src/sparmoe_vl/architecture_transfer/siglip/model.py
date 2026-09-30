"""Self-contained SigLIP SparMoE encoders for both training stages.

SigLIP differs from the CLIP implementation in two important ways. Its vision
tower is a timm trunk with no CLS token, so every patch is routed. Its text
tower pools the last position, so that last position remains dense while all
preceding positions use sparse FFNs. Stage 1 learns SPG channel rankings,
layer reference capacities, and nested subspaces under global budget ``p``;
Stage 2 freezes that structure and optimizes only token routers inside the
inherited budget-bounded expert space.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...common.two_stage import validated_trainable_parameters


def st_gumbel_softmax(logits: Tensor, tau: float = 0.4) -> Tensor:
    noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-10, 1.0)))
    soft = F.softmax((logits + noise) / tau, dim=-1)
    hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
    return (hard - soft).detach() + soft


def _freg(left: Tensor, right: Tensor) -> Tensor:
    return torch.log(torch.maximum(left, right) / torch.minimum(left, right).clamp_min(1e-8))


def _logit(value: Tensor) -> Tensor:
    value = value.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(value / (1.0 - value))


def _st_sigmoid(logits: Tensor, tau: float) -> Tensor:
    soft = torch.sigmoid(logits / tau)
    hard = (soft >= 0.5).to(soft.dtype)
    return (hard - soft).detach() + soft


class HyperNetwork(nn.Module):
    """Generate four shared structural embeddings for every sparse layer."""

    def __init__(self, num_experts: int, num_layers: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.bigru = nn.GRU(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(128, embedding_dim)
        self.register_buffer("z", torch.randn(num_experts, 32))

    def forward(self) -> Tensor:
        encoded, _ = self.bigru(self.z.unsqueeze(0))
        embeddings = self.proj(encoded.squeeze(0))
        return embeddings.unsqueeze(0).expand(self.num_layers, -1, -1)


class NestedFFN(nn.Module):
    def __init__(
        self,
        original_mlp: nn.Module,
        model_dim: int,
        ffn_dim: int,
        levels: Sequence[float],
    ) -> None:
        super().__init__()
        self.original_mlp = original_mlp
        self.ffn_dim = int(ffn_dim)
        self.levels = tuple(float(value) for value in levels)
        self.num_budgets = len(self.levels)
        for parameter in original_mlp.parameters():
            parameter.requires_grad_(False)
        self.router = nn.Linear(model_dim, self.num_budgets, bias=False)
        self.proj_mlp_d = nn.Sequential(
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, ffn_dim),
        )
        self.full_ratio_logit = nn.Parameter(torch.tensor(0.84729786))
        self.register_buffer("level_factors", torch.tensor(self.levels))
        self.routing_mode = "learned"

    def full_ratio(self) -> Tensor:
        return torch.sigmoid(self.full_ratio_logit).clamp(0.02, 0.995)

    def channel_scores(self, embedding: Tensor) -> Tensor:
        scores = self.proj_mlp_d(embedding).mean(0)
        return scores - scores.mean()

    def budget_masks(self, embedding: Tensor, tau: float) -> tuple[Tensor, Tensor]:
        scores = self.channel_scores(embedding)
        ratios = (self.full_ratio() * self.level_factors).clamp(0.01, 0.995)
        masks = []
        for ratio in ratios:
            width = max(1, min(self.ffn_dim, int(round(float(ratio.detach()) * self.ffn_dim))))
            threshold_detached = scores.detach().topk(width).values[-1]
            threshold = threshold_detached - (_logit(ratio) - _logit(ratio.detach()))
            masks.append(_st_sigmoid(scores - threshold, tau))
        return torch.stack(masks), ratios

    @staticmethod
    def importance_targets(importance: Tensor, num_budgets: int) -> Tensor:
        order = torch.argsort(importance.detach())
        targets = torch.empty_like(order)
        for index in range(num_budgets):
            start = index * order.numel() // num_budgets
            end = (index + 1) * order.numel() // num_budgets
            targets[order[start:end]] = index
        return targets

    def _gates(self, logits: Tensor, tau: float) -> Tensor:
        if self.routing_mode == "largest":
            indices = torch.full(
                (logits.shape[0],),
                self.num_budgets - 1,
                dtype=torch.long,
                device=logits.device,
            )
            return F.one_hot(indices, self.num_budgets).to(logits.dtype)
        if self.routing_mode == "random":
            indices = torch.randint(self.num_budgets, (logits.shape[0],), device=logits.device)
            return F.one_hot(indices, self.num_budgets).to(logits.dtype)
        if self.training:
            return st_gumbel_softmax(logits, tau)
        return F.one_hot(logits.argmax(-1), self.num_budgets).to(logits.dtype)

    def _sparse(
        self,
        x: Tensor,
        masks: Tensor,
        gates: Tensor,
        *,
        vision: bool,
    ) -> tuple[Tensor, Tensor]:
        if vision:
            w1, b1 = self.original_mlp.fc1.weight, self.original_mlp.fc1.bias
            w2, b2 = self.original_mlp.fc2.weight, self.original_mlp.fc2.bias
            hidden = F.gelu(x @ w1.T + b1, approximate="tanh")
        else:
            w1, b1 = self.original_mlp.c_fc.weight, self.original_mlp.c_fc.bias
            w2, b2 = self.original_mlp.c_proj.weight, self.original_mlp.c_proj.bias
            hidden = self.original_mlp.gelu(x @ w1.T + b1)
        output = (hidden * (gates.to(masks.dtype) @ masks).to(hidden.dtype)) @ w2.T + b2
        importance = hidden.detach().abs().float() @ w2.detach().float().norm(dim=0)
        return output, importance

    def _auxiliary(
        self,
        logits: Tensor,
        gates: Tensor,
        masks: Tensor,
        ratios: Tensor,
        importance: Tensor,
    ) -> dict[str, Tensor]:
        return {
            "G": gates,
            "router_probs": F.softmax(logits.float(), dim=-1),
            "router_logits": logits,
            "router_targets": self.importance_targets(importance, self.num_budgets),
            "masks": masks,
            "ratios": ratios,
            "full_ratio": self.full_ratio(),
        }

    def vision_forward(
        self, x: Tensor, embedding: Tensor, tau: float
    ) -> tuple[Tensor, dict[str, Tensor]]:
        shape = x.shape
        flattened = x.reshape(-1, shape[-1])
        logits = self.router(flattened)
        gates = self._gates(logits, tau)
        masks, ratios = self.budget_masks(embedding, tau)
        output, importance = self._sparse(flattened, masks, gates, vision=True)
        return output.reshape(shape), self._auxiliary(logits, gates, masks, ratios, importance)

    def text_forward(
        self, x: Tensor, embedding: Tensor, tau: float
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, tokens, width = x.shape
        routed = x[:, :-1].reshape(-1, width)
        logits = self.router(routed)
        gates = self._gates(logits, tau)
        masks, ratios = self.budget_masks(embedding, tau)
        sparse, importance = self._sparse(routed, masks, gates, vision=False)
        dense_last = self.original_mlp(x[:, -1:])
        output = torch.cat((sparse.reshape(batch, tokens - 1, width), dense_last), dim=1)
        return output, self._auxiliary(logits, gates, masks, ratios, importance)


class SigLIPSparMoE(nn.Module):
    """SparMoE wrapper shared by the three registered SigLIP scales."""

    def __init__(
        self,
        clip_model: nn.Module,
        modality: str,
        stage: int,
        target_ratio: float,
        levels: Sequence[float],
        tau: float = 0.4,
    ) -> None:
        super().__init__()
        if modality not in ("vision", "text") or stage not in (1, 2):
            raise ValueError("invalid SigLIP modality or stage")
        self.clip_model = clip_model
        self.modality = modality
        self.stage = int(stage)
        self.target_ratio = float(target_ratio)
        self.levels = tuple(float(value) for value in levels)
        self.tau = float(tau)
        for parameter in clip_model.parameters():
            parameter.requires_grad_(False)

        if modality == "vision":
            self.trunk = clip_model.visual.trunk
            self.blocks = self.trunk.blocks
        else:
            self.text_tower = clip_model.text
            self.blocks = self.text_tower.transformer.resblocks
        first_mlp = self.blocks[0].mlp
        if modality == "vision":
            model_dim = int(first_mlp.fc1.weight.shape[1])
            ffn_dim = int(first_mlp.fc1.weight.shape[0])
        else:
            model_dim = int(first_mlp.c_fc.weight.shape[1])
            ffn_dim = int(first_mlp.c_fc.weight.shape[0])
        self.ffn_dim = ffn_dim
        self.moe_layers = list(range(len(self.blocks)))
        self.hypernetwork = HyperNetwork(4, len(self.blocks), 128)
        self.layers = nn.ModuleDict(
            {
                str(index): NestedFFN(block.mlp, model_dim, ffn_dim, self.levels)
                for index, block in enumerate(self.blocks)
            }
        )
        with torch.no_grad():
            initial = _logit(torch.tensor(self.target_ratio))
            for layer in self.layers.values():
                layer.full_ratio_logit.copy_(initial)
        self._configure_stage_parameters()

    def _configure_stage_parameters(self) -> None:
        structure_is_trainable = self.stage == 1
        self.hypernetwork.requires_grad_(structure_is_trainable)
        for layer in self.layers.values():
            layer.proj_mlp_d.requires_grad_(structure_is_trainable)
            layer.full_ratio_logit.requires_grad_(structure_is_trainable)
            layer.router.requires_grad_(self.stage == 2)
            layer.routing_mode = "largest" if self.stage == 1 else "learned"

    def train(self, mode: bool = True) -> "SigLIPSparMoE":
        super().train(mode)
        self.clip_model.eval()
        return self

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        structure = list(self.hypernetwork.parameters())
        routers = []
        for layer in self.layers.values():
            structure.extend(layer.proj_mlp_d.parameters())
            structure.append(layer.full_ratio_logit)
            routers.extend(layer.router.parameters())
        return validated_trainable_parameters(
            self,
            self.stage,
            structure_parameters=structure,
            router_parameters=routers,
        )

    def set_routing_mode(self, mode: str) -> None:
        if self.stage != 2 or mode not in ("learned", "random"):
            if self.stage != 2:
                raise RuntimeError("routing modes are only available in Stage 2")
            raise ValueError("routing mode must be 'learned' or 'random'")
        for layer in self.layers.values():
            layer.routing_mode = mode

    @torch.no_grad()
    def dense_features(self, inputs: Tensor) -> Tensor:
        encoded = (
            self.clip_model.encode_image(inputs)
            if self.modality == "vision"
            else self.clip_model.encode_text(inputs)
        )
        return F.normalize(encoded, dim=-1)

    def encode_sparse(self, inputs: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]:
        return (
            self._encode_vision(inputs)
            if self.modality == "vision"
            else self._encode_text(inputs)
        )

    def _encode_vision(self, images: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]:
        x = self.trunk.patch_embed(images)
        x = self.trunk._pos_embed(x)
        x = self.trunk.patch_drop(x)
        x = self.trunk.norm_pre(x)
        embeddings = self.hypernetwork()
        auxiliary = []
        for index, block in enumerate(self.blocks):
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            normalized = block.norm2(x)
            layer = self.layers[str(index)]
            output, info = layer.vision_forward(normalized, embeddings[index], self.tau)
            x = x + block.drop_path2(block.ls2(output))
            auxiliary.append(info)
        x = self.trunk.norm(x)
        x = self.trunk.attn_pool(x)
        x = self.trunk.fc_norm(x)
        x = self.trunk.head_drop(x)
        x = self.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return F.normalize(x, dim=-1), auxiliary

    def _encode_text(self, tokens: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]:
        x, attention_mask = self.text_tower._embeds(tokens)
        embeddings = self.hypernetwork()
        auxiliary = []
        for index, block in enumerate(self.blocks):
            x = x + block.ls_1(block.attention(q_x=block.ln_1(x), attn_mask=attention_mask))
            normalized = block.ln_2(x)
            layer = self.layers[str(index)]
            output, info = layer.text_forward(normalized, embeddings[index], self.tau)
            x = x + block.ls_2(output)
            auxiliary.append(info)
        x = self.text_tower.ln_final(x)[:, -1]
        projection = self.text_tower.text_projection
        if projection is not None:
            x = projection(x) if isinstance(projection, nn.Linear) else x @ projection
        return F.normalize(x, dim=-1), auxiliary

    def stage1_loss(
        self,
        sparse: Tensor,
        dense: Tensor,
        auxiliary: Sequence[Mapping[str, Tensor]],
    ) -> tuple[Tensor, dict[str, Any]]:
        distance = 1.0 - F.cosine_similarity(sparse, dense.detach(), dim=-1).mean()
        full = torch.stack([item["full_ratio"] for item in auxiliary])
        budget = _freg(
            full.mean(),
            torch.tensor(self.target_ratio, device=full.device),
        )
        gaps = torch.stack(
            [(item["ratios"][1:] - item["ratios"][:-1]).mean() for item in auxiliary]
        )
        separation = F.relu(0.03 - gaps).mean()
        loss = 100.0 * distance + 50.0 * budget + separation
        return loss, {
            "total": float(loss.detach()),
            "distill": float(distance.detach()),
            "Rp": float(budget.detach()),
            "Rsep": float(separation.detach()),
            "layer_ratios": full.detach().cpu().tolist(),
        }

    def stage2_loss(
        self,
        sparse: Tensor,
        dense: Tensor,
        auxiliary: Sequence[Mapping[str, Tensor]],
        router_weight: float,
    ) -> tuple[Tensor, dict[str, Any]]:
        distance = 1.0 - F.cosine_similarity(sparse, dense.detach(), dim=-1).mean()
        router_loss = torch.stack(
            [
                F.cross_entropy(item["router_logits"].float(), item["router_targets"])
                for item in auxiliary
            ]
        ).mean()
        router_accuracy = torch.stack(
            [
                (item["router_logits"].argmax(-1) == item["router_targets"]).float().mean()
                for item in auxiliary
            ]
        ).mean()
        full = torch.stack([item["full_ratio"] for item in auxiliary])
        budget = _freg(full.mean(), torch.tensor(self.target_ratio, device=full.device))
        gaps = torch.stack(
            [(item["ratios"][1:] - item["ratios"][:-1]).mean() for item in auxiliary]
        )
        spread = F.relu(0.03 - gaps).mean()
        usages = [item["G"].float().mean(0) for item in auxiliary]
        expected = torch.stack(
            [(usage * item["ratios"]).sum() for usage, item in zip(usages, auxiliary)]
        )
        loss = 100.0 * distance + router_weight * router_loss
        return loss, {
            "total": float(loss.detach()),
            "distill": float(distance.detach()),
            "Rrouter": float(router_loss.detach()),
            "router_acc": float(router_accuracy.detach()),
            "inherited_budget_error": float(budget.detach()),
            "inherited_separation": float(spread.detach()),
            "base_mean": float(full.detach().mean()),
            "actual_mean": float(expected.detach().mean()),
            "usage": torch.stack(usages).mean(0).detach().cpu().tolist(),
            "full_ratios": full.detach().cpu().tolist(),
        }


def controller_state(model: SigLIPSparMoE) -> dict[str, Mapping[str, Tensor]]:
    """Return only learned controller tensors, excluding the frozen backbone."""

    layer_state: dict[str, Tensor] = {}
    for name, layer in model.layers.items():
        for key, value in layer.proj_mlp_d.state_dict().items():
            layer_state[f"{name}.proj_mlp_d.{key}"] = value
        layer_state[f"{name}.full_ratio_logit"] = layer.full_ratio_logit.detach()
        layer_state[f"{name}.level_factors"] = layer.level_factors.detach()
        if model.stage == 2:
            for key, value in layer.router.state_dict().items():
                layer_state[f"{name}.router.{key}"] = value
    return {"hypernetwork": model.hypernetwork.state_dict(), "layers": layer_state}


def _load_structure_layers(model: SigLIPSparMoE, state: Mapping[str, Tensor]) -> None:
    for name, layer in model.layers.items():
        prefix = f"{name}.proj_mlp_d."
        projection = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        layer.proj_mlp_d.load_state_dict(projection, strict=True)
        layer.full_ratio_logit.copy_(
            torch.as_tensor(state[f"{name}.full_ratio_logit"]).reshape(())
        )


@torch.no_grad()
def initialize_stage2(model: SigLIPSparMoE, checkpoint: Mapping[str, Any]) -> list[float]:
    """Load the immutable Stage-1 structure used by Stage 2."""

    if model.stage != 2:
        raise RuntimeError("expected a Stage-2 model")
    controller = checkpoint.get("controller")
    hyper_state = controller.get("hypernetwork") if isinstance(controller, Mapping) else None
    layer_state = controller.get("layers") if isinstance(controller, Mapping) else None
    if not isinstance(hyper_state, Mapping) or not isinstance(layer_state, Mapping):
        raise ValueError("Stage-1 checkpoint has no paper-protocol controller state")
    model.hypernetwork.load_state_dict(hyper_state, strict=True)
    _load_structure_layers(model, layer_state)
    return [float(layer.full_ratio()) for layer in model.layers.values()]


def load_stage2_controller(model: SigLIPSparMoE, checkpoint: Mapping[str, Any]) -> None:
    """Load a paper-protocol frozen structure and its Stage-2 routers."""

    if model.stage != 2:
        raise RuntimeError("expected a Stage-2 model")
    controller = checkpoint.get("controller")
    hyper_state = controller.get("hypernetwork") if isinstance(controller, Mapping) else None
    layer_state = controller.get("layers") if isinstance(controller, Mapping) else None
    if not isinstance(hyper_state, Mapping) or not isinstance(layer_state, Mapping):
        raise ValueError("Stage-2 checkpoint has no paper-protocol controller state")
    model.hypernetwork.load_state_dict(hyper_state, strict=True)
    with torch.no_grad():
        _load_structure_layers(model, layer_state)
        for name, layer in model.layers.items():
            layer.router.load_state_dict(
                {"weight": layer_state[f"{name}.router.weight"]}, strict=True
            )


def build_model(
    clip_model: nn.Module,
    modality: str,
    stage: int,
    target_ratio: float,
    levels: Sequence[float],
    tau: float = 0.4,
) -> SigLIPSparMoE:
    return SigLIPSparMoE(clip_model, modality, stage, target_ratio, levels, tau)

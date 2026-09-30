"""Shared, dimension-aware SparMoE implementation for OpenAI CLIP ViTs.

The visual CLS token and text position zero use the original dense FFN, while
all remaining positions use sparse FFNs. Stage 1 learns channel rankings,
layer reference capacities, and nested subspaces under global budget ``p``.
Stage 2 freezes those structures and optimizes only token routers inside the
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
    denominator = torch.minimum(left, right).clamp_min(1e-8)
    return torch.log(torch.maximum(left, right) / denominator)


def _logit(value: Tensor) -> Tensor:
    value = value.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(value / (1.0 - value))


def _st_sigmoid(logits: Tensor, tau: float) -> Tensor:
    soft = torch.sigmoid(logits / tau)
    hard = (soft >= 0.5).to(soft.dtype)
    return (hard - soft).detach() + soft


class HyperNetwork(nn.Module):
    """Generate shared structural embeddings for every sparse layer."""

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


class NestedCLIPFFN(nn.Module):
    """Nested FFN structure learned in Stage 1 and routed in Stage 2."""

    def __init__(
        self,
        original_mlp: nn.Module,
        model_dim: int,
        ffn_dim: int,
        levels: Sequence[float],
        initial_ratio: float,
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
        self.full_ratio_logit = nn.Parameter(_logit(torch.tensor(float(initial_ratio))))
        self.register_buffer("level_factors", torch.tensor(self.levels, dtype=torch.float32))
        self.routing_mode = "learned"
        self.forced_expert_ids: Tensor | None = None

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
            width = max(
                1,
                min(self.ffn_dim, int(round(float(ratio.detach()) * self.ffn_dim))),
            )
            detached_threshold = scores.detach().topk(width).values[-1]
            threshold = detached_threshold - (_logit(ratio) - _logit(ratio.detach()))
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

    def forward(
        self,
        states: Tensor,
        embedding: Tensor,
        tau: float,
        valid_nonfirst: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch, tokens, width = states.shape
        first_output = self.original_mlp(states[:, :1])
        routed = states[:, 1:].reshape(-1, width)
        logits = self.router(routed)
        probabilities = F.softmax(logits.float(), dim=-1)
        masks, ratios = self.budget_masks(embedding, tau)
        if self.routing_mode == "largest":
            indices = torch.full(
                (len(routed),),
                self.num_budgets - 1,
                dtype=torch.long,
                device=routed.device,
            )
            gates = F.one_hot(indices, self.num_budgets).to(routed.dtype)
        elif self.routing_mode == "forced":
            if self.forced_expert_ids is None:
                raise RuntimeError("forced routing requires forced_expert_ids")
            indices = self.forced_expert_ids.to(device=routed.device, dtype=torch.long)
            if indices.ndim != 1 or indices.numel() != len(routed):
                raise ValueError("forced route must contain one expert id per routed token")
            if int(indices.min()) < 0 or int(indices.max()) >= self.num_budgets:
                raise ValueError("forced route contains an invalid expert id")
            gates = F.one_hot(indices, self.num_budgets).to(routed.dtype)
        elif self.routing_mode == "shuffled":
            learned = logits.argmax(-1)
            token_order = torch.argsort(learned, stable=True)
            capacity_pool = torch.sort(learned, descending=True).values
            indices = torch.empty_like(learned)
            indices[token_order] = capacity_pool
            gates = F.one_hot(indices, self.num_budgets).to(routed.dtype)
        elif self.routing_mode == "uniform":
            measured_widths = masks.float().mean(-1)
            learned = logits.argmax(-1)
            usage = F.one_hot(learned, self.num_budgets).float().mean(0)
            target_width = (usage * measured_widths).sum()
            uniform_width = int(torch.ceil(target_width.detach() * self.ffn_dim).item())
            uniform_width = max(1, min(self.ffn_dim, uniform_width))
            scores = self.channel_scores(embedding)
            uniform_mask = torch.zeros_like(scores)
            uniform_mask.scatter_(0, scores.topk(uniform_width).indices, 1.0)
            masks = uniform_mask.unsqueeze(0).expand(self.num_budgets, -1)
            gates = F.one_hot(
                torch.zeros(len(routed), dtype=torch.long, device=routed.device),
                self.num_budgets,
            ).to(routed.dtype)
        elif self.routing_mode == "random":
            indices = torch.randint(self.num_budgets, (len(routed),), device=routed.device)
            gates = F.one_hot(indices, self.num_budgets).to(routed.dtype)
        elif self.training:
            gates = st_gumbel_softmax(logits, tau)
        else:
            gates = F.one_hot(logits.argmax(-1), self.num_budgets).to(routed.dtype)

        hidden = self.original_mlp.gelu(self.original_mlp.c_fc(routed))
        intermediate_norm = getattr(self.original_mlp, "ln", None)
        if intermediate_norm is not None:
            hidden = intermediate_norm(hidden)
        sparse = self.original_mlp.c_proj(
            hidden * (gates.to(masks.dtype) @ masks).to(hidden.dtype)
        )
        with torch.no_grad():
            importance = (
                hidden.detach().abs().float()
                @ self.original_mlp.c_proj.weight.detach().float().norm(dim=0)
            )
            targets = self.importance_targets(importance, self.num_budgets)
        valid = (
            torch.ones(len(routed), dtype=torch.bool, device=routed.device)
            if valid_nonfirst is None
            else valid_nonfirst.reshape(-1)
        )
        output = torch.cat((first_output, sparse.reshape(batch, tokens - 1, width)), dim=1)
        return output, {
            "G": gates,
            "router_probs": probabilities,
            "router_logits": logits,
            "router_targets": targets,
            "valid": valid,
            "masks": masks,
            "ratios": ratios,
            "full_ratio": self.full_ratio(),
        }


class CLIPSparMoE(nn.Module):
    """One implementation shared by both CLIP versions, towers, and stages."""

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
            raise ValueError("invalid CLIP modality or stage")
        self.clip_model = clip_model
        self.modality = modality
        self.stage = int(stage)
        self.target_ratio = float(target_ratio)
        self.levels = tuple(float(value) for value in levels)
        self.tau = float(tau)
        for parameter in clip_model.parameters():
            parameter.requires_grad_(False)
        transformer = (
            clip_model.visual.transformer if modality == "vision" else clip_model.transformer
        )
        self.blocks = transformer.resblocks
        self.batch_first = bool(transformer.batch_first)
        first_mlp = self.blocks[0].mlp
        model_dim = int(first_mlp.c_fc.weight.shape[1])
        self.ffn_dim = int(first_mlp.c_fc.weight.shape[0])
        self.moe_layers = list(range(len(self.blocks)))
        self.hypernetwork = HyperNetwork(len(self.levels), len(self.blocks), 128)
        self.layers = nn.ModuleDict(
            {
                str(index): NestedCLIPFFN(
                    block.mlp,
                    model_dim,
                    self.ffn_dim,
                    self.levels,
                    self.target_ratio,
                )
                for index, block in enumerate(self.blocks)
            }
        )
        self._structure_parameters_override: tuple[nn.Parameter, ...] | None = None
        self._configure_stage_parameters()

    def _configure_stage_parameters(self) -> None:
        structure_is_trainable = self.stage == 1
        self.hypernetwork.requires_grad_(structure_is_trainable)
        for layer in self.layers.values():
            layer.proj_mlp_d.requires_grad_(structure_is_trainable)
            layer.full_ratio_logit.requires_grad_(structure_is_trainable)
            layer.router.requires_grad_(self.stage == 2)
            layer.routing_mode = "largest" if self.stage == 1 else "learned"

    def train(self, mode: bool = True) -> "CLIPSparMoE":
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
        if self._structure_parameters_override is not None:
            structure = list(self._structure_parameters_override)
        return validated_trainable_parameters(
            self,
            self.stage,
            structure_parameters=structure,
            router_parameters=routers,
        )

    def register_structure_parameters(self, parameters: Sequence[nn.Parameter]) -> None:
        """Register an intentional structure replacement for an ablation."""

        self._structure_parameters_override = tuple(parameters)

    def set_routing_mode(self, mode: str) -> None:
        if self.stage != 2:
            raise RuntimeError("routing modes are only available in Stage 2")
        if mode not in ("learned", "random", "uniform", "shuffled", "forced"):
            raise ValueError(
                "routing mode must be 'learned', 'random', 'uniform', 'shuffled', or 'forced'"
            )
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
        if self.modality == "vision":
            return self._encode_vision(inputs)
        return self._encode_text(inputs)

    def _encode_vision(self, images: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]:
        visual = self.clip_model.visual
        cast_dtype = visual.transformer.get_cast_dtype()
        x = visual.conv1(images.to(cast_dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(cast_dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat((class_token, x), dim=1)
        x = visual.patch_dropout(x + visual.positional_embedding.to(cast_dtype))
        x = visual.ln_pre(x)
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        embeddings = self.hypernetwork()
        auxiliary = []
        for index, block in enumerate(self.blocks):
            x = x + block.ls_1(block.attention(q_x=block.ln_1(x), attn_mask=None))
            normalized = block.ln_2(x)
            normalized_bf = normalized if self.batch_first else normalized.transpose(0, 1)
            layer = self.layers[str(index)]
            output, info = layer(normalized_bf, embeddings[index], self.tau, None)
            if not self.batch_first:
                output = output.transpose(0, 1)
            x = x + block.ls_2(output)
            auxiliary.append(info)
        if not self.batch_first:
            x = x.transpose(0, 1)
        x = visual.ln_post(x)[:, 0]
        if visual.proj is not None:
            x = x @ visual.proj
        return F.normalize(x, dim=-1), auxiliary

    def _encode_text(self, tokens: Tensor) -> tuple[Tensor, list[dict[str, Tensor]]]:
        clip = self.clip_model
        cast_dtype = clip.transformer.get_cast_dtype()
        attention_mask = clip.attn_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(tokens.device)
        x = clip.token_embedding(tokens).to(cast_dtype)
        x = x + clip.positional_embedding.to(cast_dtype)
        eos = tokens.argmax(dim=-1)
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        valid_nonfirst = (positions <= eos[:, None])[:, 1:]
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        embeddings = self.hypernetwork()
        auxiliary = []
        for index, block in enumerate(self.blocks):
            x = x + block.ls_1(block.attention(q_x=block.ln_1(x), attn_mask=attention_mask))
            normalized = block.ln_2(x)
            normalized_bf = normalized if self.batch_first else normalized.transpose(0, 1)
            layer = self.layers[str(index)]
            output, info = layer(
                normalized_bf,
                embeddings[index],
                self.tau,
                valid_nonfirst,
            )
            if not self.batch_first:
                output = output.transpose(0, 1)
            x = x + block.ls_2(output)
            auxiliary.append(info)
        if not self.batch_first:
            x = x.transpose(0, 1)
        x = clip.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), eos]
        projection = clip.text_projection
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
                F.cross_entropy(
                    item["router_logits"][item["valid"]].float(),
                    item["router_targets"][item["valid"]],
                )
                for item in auxiliary
            ]
        ).mean()
        router_accuracy = torch.stack(
            [
                (
                    item["router_logits"][item["valid"]].argmax(-1)
                    == item["router_targets"][item["valid"]]
                )
                .float()
                .mean()
                for item in auxiliary
            ]
        ).mean()
        full = torch.stack([item["full_ratio"] for item in auxiliary])
        budget = _freg(full.mean(), torch.tensor(self.target_ratio, device=full.device))
        gaps = torch.stack(
            [(item["ratios"][1:] - item["ratios"][:-1]).mean() for item in auxiliary]
        )
        spread = F.relu(0.03 - gaps).mean()
        usages = [item["G"][item["valid"]].float().mean(0) for item in auxiliary]
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


def controller_state(model: CLIPSparMoE) -> dict[str, Mapping[str, Tensor]]:
    """Return only learned controller state, excluding the frozen CLIP backbone."""

    layers: dict[str, Tensor] = {}
    for name, layer in model.layers.items():
        for key, value in layer.proj_mlp_d.state_dict().items():
            layers[f"{name}.proj_mlp_d.{key}"] = value
        layers[f"{name}.full_ratio_logit"] = layer.full_ratio_logit.detach()
        layers[f"{name}.level_factors"] = layer.level_factors.detach()
        if model.stage == 2:
            layers[f"{name}.router.weight"] = layer.router.weight.detach()
    return {
        "hypernetwork": model.hypernetwork.state_dict(),
        "layers": layers,
    }


def _controller_parts(
    checkpoint: Mapping[str, Any],
) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
    controller = checkpoint.get("controller")
    hypernetwork = controller.get("hypernetwork") if isinstance(controller, Mapping) else None
    layers = controller.get("layers") if isinstance(controller, Mapping) else None
    if not isinstance(hypernetwork, Mapping) or not isinstance(layers, Mapping):
        raise ValueError("checkpoint has no paper-protocol CLIP controller state")
    return hypernetwork, layers


def _load_structure_layers(model: CLIPSparMoE, state: Mapping[str, Tensor]) -> None:
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
def initialize_stage2(model: CLIPSparMoE, checkpoint: Mapping[str, Any]) -> list[float]:
    """Load the immutable Stage-1 structure used by Stage 2."""

    if model.stage != 2:
        raise RuntimeError("expected a Stage-2 model")
    hypernetwork, layers = _controller_parts(checkpoint)
    model.hypernetwork.load_state_dict(hypernetwork, strict=True)
    _load_structure_layers(model, layers)
    return [float(layer.full_ratio()) for layer in model.layers.values()]


def load_stage2_controller(model: CLIPSparMoE, checkpoint: Mapping[str, Any]) -> None:
    """Load a paper-protocol frozen structure and its Stage-2 routers."""

    if model.stage != 2:
        raise RuntimeError("expected a Stage-2 model")
    hypernetwork, layers = _controller_parts(checkpoint)
    model.hypernetwork.load_state_dict(hypernetwork, strict=True)
    with torch.no_grad():
        _load_structure_layers(model, layers)
        for name, layer in model.layers.items():
            layer.router.load_state_dict(
                {"weight": layers[f"{name}.router.weight"]}, strict=True
            )


def build_model(
    clip_model: nn.Module,
    modality: str,
    stage: int,
    target_ratio: float,
    levels: Sequence[float],
    tau: float = 0.4,
) -> CLIPSparMoE:
    return CLIPSparMoE(clip_model, modality, stage, target_ratio, levels, tau)

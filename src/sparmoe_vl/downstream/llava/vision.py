"""Paper-protocol SparMoE-VL adapter for LLaVA's CLIP-336 vision tower."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from ...common.budgets import LayerAdaptiveBudget
from ...common.routing import (
    TokenCapacityRouter,
    TokenRoutingOutput,
    balanced_capacity_targets,
    ffn_output_contribution_scores,
)
from ...common.sparse_patterns import SparsePatternGenerator, SparsePatternOutput
from ...common.two_stage import validated_trainable_parameters
from .protocol import (
    CAPACITY_FACTORS,
    TARGET_RATIO,
    checkpoint_metadata,
    torch_load,
)


NUM_LAYERS = 24
MODEL_DIM = 1_024
FFN_DIM = 4_096


@dataclass(frozen=True)
class CLIP336LayerOutput:
    transformer_layer: int
    routing: TokenRoutingOutput
    router_targets: Tensor
    sparse_pattern: SparsePatternOutput


@dataclass(frozen=True)
class CLIP336VisionOutput:
    last_hidden_state: Tensor
    pooler_output: Tensor
    hidden_states: tuple[Tensor, ...]
    layers: tuple[CLIP336LayerOutput, ...]
    base_ratios: Tensor
    retention_ratios: Tensor


def load_local_clip_vision(path: str | Path) -> nn.Module:
    """Load the trusted local CLIP vision weights used by LLaVA-v1.5.

    Loading the visual state directly preserves compatibility with PyTorch 2.5,
    including Transformers versions that reject legacy ``.bin`` model loading.
    """

    try:
        from transformers import CLIPConfig, CLIPVisionModel
    except ImportError as error:
        raise RuntimeError("install the llava optional dependencies") from error
    model_root = Path(path)
    config_path = model_root / "config.json"
    weights_path = model_root / "pytorch_model.bin"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete local CLIP-336 model: {model_root}")
    config = CLIPConfig.from_pretrained(str(model_root), local_files_only=True)
    if config.vision_config.image_size != 336:
        raise ValueError("the LLaVA study requires CLIP input resolution 336")
    model = CLIPVisionModel(config.vision_config)
    try:
        state = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model_keys = model.state_dict().keys()
    if any(key.startswith("vision_model.") for key in model_keys):
        vision_state = {
            key: value for key, value in state.items() if key.startswith("vision_model.")
        }
    else:
        vision_state = {
            key.removeprefix("vision_model."): value
            for key, value in state.items()
            if key.startswith("vision_model.")
        }
    missing, unexpected = model.load_state_dict(vision_state, strict=False)
    missing = [key for key in missing if not key.endswith("position_ids")]
    unexpected = [key for key in unexpected if not key.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(
            "CLIP-336 visual state mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return model


class SparMoECLIP336VisionTower(nn.Module):
    """Apply the exact two-stage sparse conversion to Hugging Face CLIP-336."""

    def __init__(
        self,
        clip_vision: nn.Module,
        *,
        training_stage: int = 2,
        target_ratio: float = TARGET_RATIO,
        capacity_factors: Sequence[float] = CAPACITY_FACTORS,
        router_temperature: float = 0.4,
        mask_temperature: float = 0.4,
    ) -> None:
        super().__init__()
        if training_stage not in (1, 2):
            raise ValueError("training_stage must be 1 or 2")
        self.clip = clip_vision
        self.training_stage = int(training_stage)
        self._freeze_backbone()
        self.vision_model = getattr(clip_vision, "vision_model", clip_vision)
        self.layers = self.vision_model.encoder.layers
        self.config = clip_vision.config
        if len(self.layers) != NUM_LAYERS:
            raise ValueError(f"CLIP-336 must contain {NUM_LAYERS} visual layers")
        if int(self.config.hidden_size) != MODEL_DIM:
            raise ValueError(f"CLIP-336 hidden size must be {MODEL_DIM}")
        if int(self.config.intermediate_size) != FFN_DIM:
            raise ValueError(f"CLIP-336 FFN size must be {FFN_DIM}")
        factors = tuple(float(value) for value in capacity_factors)
        if factors != CAPACITY_FACTORS:
            raise ValueError(f"capacity factors must remain {CAPACITY_FACTORS}")
        self.num_capacity_levels = len(factors)
        self.budget = LayerAdaptiveBudget(
            num_layers=NUM_LAYERS,
            target_ratio=target_ratio,
            capacity_factors=factors,
        )
        self.sparse_pattern_generator = SparsePatternGenerator(
            num_layers=NUM_LAYERS,
            ffn_dims=FFN_DIM,
            num_capacity_levels=self.num_capacity_levels,
            mask_temperature=mask_temperature,
        )
        self.routers = nn.ModuleList(
            TokenCapacityRouter(
                model_dim=MODEL_DIM,
                num_capacity_levels=self.num_capacity_levels,
                temperature=router_temperature,
            )
            for _ in range(NUM_LAYERS)
        )
        self._configure_stage_parameters()

    def _configure_stage_parameters(self) -> None:
        structure_is_trainable = self.training_stage == 1
        self.budget.requires_grad_(structure_is_trainable)
        self.sparse_pattern_generator.requires_grad_(structure_is_trainable)
        self.routers.requires_grad_(self.training_stage == 2)

    def _freeze_backbone(self) -> None:
        self.clip.requires_grad_(False)
        self.clip.eval()

    def train(self, mode: bool = True) -> "SparMoECLIP336VisionTower":
        super().train(mode)
        self.clip.eval()
        return self

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return validated_trainable_parameters(
            self,
            self.training_stage,
            structure_parameters=(
                *self.budget.parameters(),
                *self.sparse_pattern_generator.parameters(),
            ),
            router_parameters=self.routers.parameters(),
        )

    @torch.no_grad()
    def dense_forward(self, pixel_values: Tensor) -> Any:
        return self.clip(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

    def forward(
        self,
        pixel_values: Tensor,
        *,
        routing_mode: str = "learned",
        output_hidden_states: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> CLIP336VisionOutput:
        if pixel_values.ndim != 4 or pixel_values.shape[0] == 0:
            raise ValueError("pixel_values must have shape [batch, channels, 336, 336]")
        retention_ratios = self.budget()
        base_ratios = self.budget.base_ratios()
        patterns = self.sparse_pattern_generator.all_layers(retention_ratios)

        hidden_states = self.vision_model.embeddings(pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)
        all_hidden_states = [hidden_states]
        layer_outputs = []
        for layer_index, encoder_layer in enumerate(self.layers):
            residual = hidden_states
            normalized = encoder_layer.layer_norm1(hidden_states)
            attended, _ = encoder_layer.self_attn(
                hidden_states=normalized,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )
            hidden_states = residual + attended
            residual = hidden_states
            normalized = encoder_layer.layer_norm2(hidden_states)
            mlp_output, layer_output = self._sparse_mlp(
                normalized,
                encoder_layer.mlp,
                layer_index,
                patterns[layer_index],
                routing_mode,
                generator,
            )
            hidden_states = residual + mlp_output
            layer_outputs.append(layer_output)
            all_hidden_states.append(hidden_states)
        pooled = self.vision_model.post_layernorm(hidden_states[:, 0, :])
        return CLIP336VisionOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=(tuple(all_hidden_states) if output_hidden_states else tuple()),
            layers=tuple(layer_outputs),
            base_ratios=base_ratios,
            retention_ratios=retention_ratios,
        )

    def _sparse_mlp(
        self,
        token_states: Tensor,
        original_mlp: nn.Module,
        layer_index: int,
        pattern: SparsePatternOutput,
        routing_mode: str,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, CLIP336LayerOutput]:
        batch_size, token_count, model_dim = token_states.shape
        dense_class = original_mlp(token_states[:, :1])
        patch_states = token_states[:, 1:].reshape(-1, model_dim)
        router = self.routers[layer_index]
        routing = router(
            patch_states.to(dtype=router.projection.weight.dtype),
            mode="largest" if self.training_stage == 1 else routing_mode,
            generator=generator,
        )
        hidden = original_mlp.activation_fn(
            patch_states @ original_mlp.fc1.weight.T + original_mlp.fc1.bias
        )
        selected = routing.gates.to(pattern.masks.dtype) @ pattern.masks
        patch_output = (
            hidden * selected.to(dtype=hidden.dtype)
        ) @ original_mlp.fc2.weight.T + original_mlp.fc2.bias
        if self.training_stage == 1:
            targets = torch.full(
                (patch_states.shape[0],),
                self.num_capacity_levels - 1,
                dtype=torch.long,
                device=patch_states.device,
            )
        else:
            importance = ffn_output_contribution_scores(hidden, original_mlp.fc2.weight)
            targets = balanced_capacity_targets(
                importance,
                self.num_capacity_levels,
            )
        output = torch.cat(
            [dense_class, patch_output.reshape(batch_size, token_count - 1, model_dim)],
            dim=1,
        )
        return output, CLIP336LayerOutput(
            transformer_layer=layer_index,
            routing=routing,
            router_targets=targets,
            sparse_pattern=pattern,
        )

    @torch.no_grad()
    def initialize_stage2_from_stage1(self, checkpoint: Mapping[str, Any]) -> Tensor:
        if self.training_stage != 2:
            raise RuntimeError("Stage-1 initialization requires a Stage-2 encoder")
        metadata = checkpoint_metadata(checkpoint, expected_stage=1)
        self._load_controller_state(checkpoint, metadata, load_routers=False)
        return self.budget.base_ratios().detach().clone()

    def load_checkpoint(self, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        metadata = checkpoint_metadata(checkpoint, expected_stage=self.training_stage)
        self._load_controller_state(
            checkpoint,
            metadata,
            load_routers=self.training_stage == 2,
        )
        return metadata

    def _load_controller_state(
        self,
        checkpoint: Mapping[str, Any],
        metadata: Mapping[str, Any],
        *,
        load_routers: bool,
    ) -> None:
        if metadata["format"] != "release_v3":
            raise ValueError("checkpoint does not use the paper-defined two-stage protocol")
        state = checkpoint.get("encoder")
        if not isinstance(state, Mapping):
            raise ValueError("release checkpoint is missing encoder state")
        self.sparse_pattern_generator.load_state_dict(
            state["sparse_pattern_generator"],
            strict=True,
        )
        self.budget.load_state_dict(state["budget"], strict=True)
        if load_routers:
            self.routers.load_state_dict(state["routers"], strict=True)

    def encoder_state(self) -> dict[str, Any]:
        state = {
            "budget": self.budget.state_dict(),
            "sparse_pattern_generator": self.sparse_pattern_generator.state_dict(),
        }
        if self.training_stage == 2:
            state["routers"] = self.routers.state_dict()
        return state


def load_sparse_tower(
    checkpoint_path: str | Path,
    clip_path: str | Path,
    device: str | torch.device,
) -> tuple[SparMoECLIP336VisionTower, dict[str, Any]]:
    checkpoint = torch_load(checkpoint_path)
    metadata = checkpoint_metadata(checkpoint, expected_stage=2)
    clip = load_local_clip_vision(clip_path)
    tower = SparMoECLIP336VisionTower(
        clip,
        training_stage=2,
        target_ratio=metadata["target_ratio"],
        capacity_factors=metadata["capacity_factors"],
    )
    tower.load_checkpoint(checkpoint)
    tower = tower.to(device).eval()
    metadata["checkpoint"] = str(Path(checkpoint_path).resolve())
    return tower, metadata

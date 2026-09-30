"""Two-stage SparMoE-VL encoder for an OpenCLIP vision Transformer.

The pretrained vision tower is frozen.  For every configured Transformer
layer, only patch tokens are routed through sparse FFN channel sets; the class
token keeps the original dense FFN path. Stage 1 learns only the SPG channel
ordering, layer-wise reference capacities, and nested subspaces under global
budget ``p``. Stage 2 freezes that complete structure and optimizes only the
token routers inside the resulting budget-bounded expert space.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..common.budgets import LayerAdaptiveBudget
from ..common.losses import LayerRoutingLossInput
from ..common.routing import (
    TokenCapacityRouter,
    TokenRoutingOutput,
    balanced_capacity_targets,
    ffn_output_contribution_scores,
)
from ..common.sparse_patterns import SparsePatternGenerator, SparsePatternOutput
from ..common.two_stage import validated_trainable_parameters


@dataclass(frozen=True)
class VisionLayerOutput:
    """Routing and sparse-pattern state for one vision Transformer layer."""

    transformer_layer: int
    routing: TokenRoutingOutput
    router_targets: Tensor
    sparse_pattern: SparsePatternOutput

    def as_loss_input(self) -> LayerRoutingLossInput:
        """Convert this layer output to the common objective interface."""

        return LayerRoutingLossInput(
            logits=self.routing.logits,
            targets=self.router_targets,
            gates=self.routing.gates,
        )


@dataclass(frozen=True)
class SparMoEVisionOutput:
    """Outputs required by training, evaluation, and budget reporting."""

    features: Tensor
    layers: Tuple[VisionLayerOutput, ...]
    base_ratios: Tensor
    retention_ratios: Tensor

    @property
    def routing_loss_inputs(self) -> Tuple[LayerRoutingLossInput, ...]:
        """Return routed layers in the order used by the budget module."""

        return tuple(layer.as_loss_input() for layer in self.layers)


class SparMoEVisionEncoder(nn.Module):
    """Insert two-stage SparMoE modules into a frozen OpenCLIP ViT.

    Args:
        clip_model: A loaded OpenCLIP model whose ``visual`` member is a
            ``VisionTransformer``.
        sparse_layers: Zero-based Transformer block indices to sparsify.
            ``None`` selects all blocks, as in the CLIP ViT-L/14 experiment.
        target_ratio: Global target for the mean layer-wise base ratio.
        capacity_factors: Increasing multipliers for the nested capacities.
        router_temperature: Straight-through Gumbel-softmax temperature.
        mask_temperature: Straight-through channel-mask temperature.

        training_stage: ``1`` learns only nested subspaces and layer
            capacities; ``2`` freezes them and learns only token routing.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        sparse_layers: Optional[Sequence[int]] = None,
        target_ratio: float = 0.7,
        capacity_factors: Sequence[float] = (0.7, 0.8, 0.9, 1.0),
        router_temperature: float = 0.4,
        mask_temperature: float = 0.4,
        structural_embedding_dim: int = 128,
        latent_dim: int = 32,
        hyper_hidden_dim: int = 64,
        training_stage: int = 2,
    ) -> None:
        super().__init__()
        self._validate_backbone(clip_model)
        if training_stage not in (1, 2):
            raise ValueError("training_stage must be 1 or 2")
        self.clip_model = clip_model
        self.training_stage = int(training_stage)
        self._freeze_backbone()

        blocks = self._blocks
        num_blocks = len(blocks)
        if sparse_layers is None:
            selected_layers = tuple(range(num_blocks))
        else:
            selected_layers = tuple(int(index) for index in sparse_layers)
        if not selected_layers:
            raise ValueError("sparse_layers must contain at least one layer")
        if len(set(selected_layers)) != len(selected_layers):
            raise ValueError("sparse_layers must not contain duplicate indices")
        if tuple(sorted(selected_layers)) != selected_layers:
            raise ValueError("sparse_layers must be in strictly increasing order")
        if selected_layers[0] < 0 or selected_layers[-1] >= num_blocks:
            raise IndexError(
                f"sparse layer indices must lie in [0, {num_blocks}), got {selected_layers}"
            )

        model_dims = []
        ffn_dims = []
        for layer_index in selected_layers:
            model_dim, ffn_dim = self._validate_mlp(
                blocks[layer_index].mlp,
                layer_index,
            )
            model_dims.append(model_dim)
            ffn_dims.append(ffn_dim)

        self.sparse_layers = selected_layers
        self._sparse_position_by_layer = {
            layer_index: position for position, layer_index in enumerate(selected_layers)
        }
        capacity_factors = tuple(float(factor) for factor in capacity_factors)
        self.num_capacity_levels = len(capacity_factors)

        self.budget = LayerAdaptiveBudget(
            num_layers=len(selected_layers),
            target_ratio=target_ratio,
            capacity_factors=capacity_factors,
        )
        self.sparse_pattern_generator = SparsePatternGenerator(
            num_layers=len(selected_layers),
            ffn_dims=ffn_dims,
            num_capacity_levels=self.num_capacity_levels,
            embedding_dim=structural_embedding_dim,
            latent_dim=latent_dim,
            hyper_hidden_dim=hyper_hidden_dim,
            mask_temperature=mask_temperature,
        )
        self.routers = nn.ModuleList(
            TokenCapacityRouter(
                model_dim=model_dim,
                num_capacity_levels=self.num_capacity_levels,
                temperature=router_temperature,
            )
            for model_dim in model_dims
        )
        self._configure_stage_parameters()

    @property
    def _visual(self) -> nn.Module:
        return self.clip_model.visual

    @property
    def _transformer(self) -> nn.Module:
        return self._visual.transformer

    @property
    def _blocks(self) -> nn.ModuleList:
        return self._transformer.resblocks

    @property
    def backbone_dtype(self) -> torch.dtype:
        """Return the dtype OpenCLIP uses inside the vision Transformer."""

        return self._transformer.get_cast_dtype()

    def train(self, mode: bool = True) -> "SparMoEVisionEncoder":
        """Change train mode while keeping the frozen backbone deterministic."""

        super().train(mode)
        self.clip_model.eval()
        return self

    def trainable_parameters(self) -> Tuple[nn.Parameter, ...]:
        """Validate and return the parameters optimized in this stage."""

        return validated_trainable_parameters(
            self,
            self.training_stage,
            structure_parameters=(
                *self.budget.parameters(),
                *self.sparse_pattern_generator.parameters(),
            ),
            router_parameters=self.routers.parameters(),
        )

    def _configure_stage_parameters(self) -> None:
        """Enforce Stage-1 structure learning and Stage-2 router isolation."""

        structure_is_trainable = self.training_stage == 1
        self.budget.requires_grad_(structure_is_trainable)
        self.sparse_pattern_generator.requires_grad_(structure_is_trainable)
        self.routers.requires_grad_(self.training_stage == 2)

    @torch.no_grad()
    def initialize_stage2_from_stage1(self, checkpoint: dict) -> Tensor:
        """Load the immutable Stage-1 structure used by Stage 2."""

        if self.training_stage != 2:
            raise RuntimeError("stage-1 initialization requires a stage-2 encoder")
        state = checkpoint.get("encoder")
        if not isinstance(state, dict):
            raise ValueError("stage-1 checkpoint does not contain encoder state")
        self.sparse_pattern_generator.load_state_dict(
            state["sparse_pattern_generator"],
            strict=True,
        )
        self.budget.load_state_dict(state["budget"], strict=True)
        return self.budget.base_ratios().detach().clone()

    @torch.no_grad()
    def dense_features(self, images: Tensor) -> Tensor:
        """Encode images with the unmodified frozen tower for distillation."""

        self._validate_images(images)
        features = self.clip_model.encode_image(
            images.to(dtype=self.backbone_dtype),
            normalize=False,
        )
        if not isinstance(features, Tensor) or features.ndim != 2:
            raise RuntimeError("clip_model.encode_image must return [batch, embedding_dim]")
        return F.normalize(features, dim=-1)

    def forward(
        self,
        images: Tensor,
        routing_mode: str = "learned",
        generator: Optional[torch.Generator] = None,
    ) -> SparMoEVisionOutput:
        """Run the sparse vision tower and collect stage-specific state."""

        self._validate_images(images)
        retention_ratios = self.budget()
        base_ratios = self.budget.base_ratios()
        sparse_patterns = self.sparse_pattern_generator.all_layers(retention_ratios)

        x = self._visual._embeds(images.to(dtype=self.backbone_dtype))
        batch_first = bool(self._transformer.batch_first)
        if not batch_first:
            x = x.transpose(0, 1).contiguous()

        layer_outputs = []
        for layer_index, block in enumerate(self._blocks):
            sparse_position = self._sparse_position_by_layer.get(layer_index)
            if sparse_position is None:
                x = block(x, attn_mask=None)
                continue

            x = x + block.ls_1(block.attention(q_x=block.ln_1(x), attn_mask=None))
            normalized = block.ln_2(x)
            normalized_batch_first = normalized if batch_first else normalized.transpose(0, 1)
            mlp_output, layer_output = self._sparse_mlp(
                normalized_batch_first,
                block.mlp,
                layer_index=layer_index,
                sparse_position=sparse_position,
                pattern=sparse_patterns[sparse_position],
                routing_mode=routing_mode,
                generator=generator,
            )
            if not batch_first:
                mlp_output = mlp_output.transpose(0, 1)
            x = x + block.ls_2(mlp_output)
            layer_outputs.append(layer_output)

        if not batch_first:
            x = x.transpose(0, 1)
        pooled, _ = self._visual._pool(x)
        if self._visual.proj is not None:
            pooled = pooled @ self._visual.proj
        features = F.normalize(pooled, dim=-1)

        return SparMoEVisionOutput(
            features=features,
            layers=tuple(layer_outputs),
            base_ratios=base_ratios,
            retention_ratios=retention_ratios,
        )

    def _sparse_mlp(
        self,
        token_states: Tensor,
        original_mlp: nn.Module,
        layer_index: int,
        sparse_position: int,
        pattern: SparsePatternOutput,
        routing_mode: str,
        generator: Optional[torch.Generator],
    ) -> Tuple[Tensor, VisionLayerOutput]:
        if token_states.ndim != 3:
            raise ValueError("vision token states must have shape [batch, tokens, dim]")
        batch_size, num_tokens, model_dim = token_states.shape
        if num_tokens < 2:
            raise ValueError("vision input must contain a class token and patch tokens")

        class_output = original_mlp(token_states[:, :1])
        patch_states = token_states[:, 1:].reshape(-1, model_dim)
        router = self.routers[sparse_position]
        router_states = patch_states.to(dtype=router.projection.weight.dtype)
        routing = router(
            router_states,
            mode="largest" if self.training_stage == 1 else routing_mode,
            generator=generator,
        )

        hidden = original_mlp.c_fc(patch_states)
        hidden = original_mlp.gelu(hidden)
        intermediate_norm = getattr(original_mlp, "ln", None)
        if intermediate_norm is not None:
            hidden = intermediate_norm(hidden)
        selected_masks = routing.gates.to(pattern.masks.dtype) @ pattern.masks
        sparse_hidden = hidden * selected_masks.to(dtype=hidden.dtype)
        patch_output = original_mlp.c_proj(sparse_hidden)

        if self.training_stage == 1:
            targets = torch.full(
                (patch_states.shape[0],),
                self.num_capacity_levels - 1,
                dtype=torch.long,
                device=patch_states.device,
            )
        else:
            contribution_scores = ffn_output_contribution_scores(
                hidden,
                original_mlp.c_proj.weight,
            )
            targets = balanced_capacity_targets(
                contribution_scores,
                self.num_capacity_levels,
            )
        output = torch.cat(
            [
                class_output,
                patch_output.reshape(batch_size, num_tokens - 1, model_dim),
            ],
            dim=1,
        )
        return output, VisionLayerOutput(
            transformer_layer=layer_index,
            routing=routing,
            router_targets=targets,
            sparse_pattern=pattern,
        )

    def _freeze_backbone(self) -> None:
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        self.clip_model.eval()

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        if images.shape[0] == 0:
            raise ValueError("images must contain at least one sample")

    @staticmethod
    def _validate_backbone(clip_model: nn.Module) -> None:
        if not isinstance(clip_model, nn.Module):
            raise TypeError("clip_model must be a torch.nn.Module")
        visual = getattr(clip_model, "visual", None)
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        if visual is None or transformer is None or blocks is None:
            raise TypeError("clip_model must expose visual.transformer.resblocks")
        if len(blocks) == 0:
            raise ValueError("the vision Transformer must contain at least one block")
        for method_name in ("_embeds", "_pool"):
            if not callable(getattr(visual, method_name, None)):
                raise TypeError(f"clip_model.visual must provide callable {method_name}()")
        if not callable(getattr(transformer, "get_cast_dtype", None)):
            raise TypeError("visual.transformer must provide get_cast_dtype()")
        if not callable(getattr(clip_model, "encode_image", None)):
            raise TypeError("clip_model must provide encode_image()")

    @staticmethod
    def _validate_mlp(mlp: nn.Module, layer_index: int) -> Tuple[int, int]:
        c_fc = getattr(mlp, "c_fc", None)
        activation = getattr(mlp, "gelu", None)
        c_proj = getattr(mlp, "c_proj", None)
        if not isinstance(c_fc, nn.Linear) or not isinstance(c_proj, nn.Linear):
            raise TypeError(f"layer {layer_index} MLP must expose Linear c_fc and c_proj")
        if not isinstance(activation, nn.Module):
            raise TypeError(f"layer {layer_index} MLP must expose gelu module")
        if c_proj.in_features != c_fc.out_features:
            raise ValueError(f"layer {layer_index} FFN projection widths do not match")
        if c_proj.out_features != c_fc.in_features:
            raise ValueError(f"layer {layer_index} FFN must preserve model width")
        return c_fc.in_features, c_fc.out_features

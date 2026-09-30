"""Sparse-pattern generation for the two-stage SparMoE-VL protocol.

Stage 1 learns one channel-priority ordering per layer and constructs every
capacity level as a Top-k prefix of that ordering. Stage 2 initializes from
and freezes those orderings, reference capacities, and nested masks, then
optimizes only the token routers within the inherited structural budget.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .budgets import inverse_sigmoid


@dataclass(frozen=True)
class SparsePatternOutput:
    """Sparse-pattern tensors for one FFN layer.

    Attributes:
        channel_scores: Learned channel-priority scores with shape ``[d_ff]``.
        masks: Straight-through binary masks with shape ``[N, d_ff]``.
        retention_ratios: Requested capacity ratios with shape ``[N]``.
        retained_channels: Exact forward-pass widths with shape ``[N]``.
    """

    channel_scores: Tensor
    masks: Tensor
    retention_ratios: Tensor
    retained_channels: Tensor


class StructuralHyperNetwork(nn.Module):
    """Map fixed latent codes to structural capacity embeddings.

    The resulting embeddings are shared across Transformer layers.  Layer-wise
    specialization is introduced later by the projections owned by
    :class:`SparsePatternGenerator`.
    """

    def __init__(
        self,
        num_capacity_levels: int,
        embedding_dim: int = 128,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        latent_codes: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        if num_capacity_levels < 2:
            raise ValueError("num_capacity_levels must be at least 2")
        if min(embedding_dim, latent_dim, hidden_dim) <= 0:
            raise ValueError("embedding dimensions must be positive")

        self.num_capacity_levels = int(num_capacity_levels)
        self.embedding_dim = int(embedding_dim)

        if latent_codes is None:
            latent_codes = torch.randn(num_capacity_levels, latent_dim)
        expected_shape = (num_capacity_levels, latent_dim)
        if tuple(latent_codes.shape) != expected_shape:
            raise ValueError(
                f"latent_codes must have shape {expected_shape}, "
                f"got {tuple(latent_codes.shape)}"
            )
        self.register_buffer("latent_codes", latent_codes.detach().float().clone())

        self.encoder = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Linear(2 * hidden_dim, embedding_dim)

    def forward(self) -> Tensor:
        encoded, _ = self.encoder(self.latent_codes.unsqueeze(0))
        return self.projection(encoded.squeeze(0))


class SparsePatternGenerator(nn.Module):
    """Learn layer-wise channel orderings and generate nested Top-k masks."""

    def __init__(
        self,
        num_layers: int,
        ffn_dims: Union[int, Sequence[int]],
        num_capacity_levels: int = 4,
        embedding_dim: int = 128,
        latent_dim: int = 32,
        hyper_hidden_dim: int = 64,
        mask_temperature: float = 0.4,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if mask_temperature <= 0:
            raise ValueError("mask_temperature must be positive")

        if isinstance(ffn_dims, int):
            layer_dims = (int(ffn_dims),) * num_layers
        else:
            layer_dims = tuple(int(dim) for dim in ffn_dims)
            if len(layer_dims) != num_layers:
                raise ValueError("ffn_dims must contain one width per layer")
        if any(dim <= 0 for dim in layer_dims):
            raise ValueError("all FFN dimensions must be positive")

        self.num_layers = int(num_layers)
        self.num_capacity_levels = int(num_capacity_levels)
        self.ffn_dims = layer_dims
        self.mask_temperature = float(mask_temperature)

        self.hypernetwork = StructuralHyperNetwork(
            num_capacity_levels=num_capacity_levels,
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            hidden_dim=hyper_hidden_dim,
        )
        self.layer_projections = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, d_ff),
            )
            for d_ff in layer_dims
        )

    def structural_embeddings(self) -> Tensor:
        """Return the shared structural embeddings with shape ``[N, d_e]``."""

        return self.hypernetwork()

    def layer_logits(
        self,
        layer_index: int,
        structural_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """Return structural channel logits with shape ``[N, d_ff]``."""

        self._validate_layer_index(layer_index)
        embeddings = (
            self.structural_embeddings()
            if structural_embeddings is None
            else structural_embeddings
        )
        expected = (self.num_capacity_levels, self.hypernetwork.embedding_dim)
        if tuple(embeddings.shape) != expected:
            raise ValueError(
                f"structural_embeddings must have shape {expected}, "
                f"got {tuple(embeddings.shape)}"
            )
        return self.layer_projections[layer_index](embeddings)

    def channel_scores(
        self,
        layer_index: int,
        structural_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """Return centered channel-priority scores for one layer."""

        self._validate_layer_index(layer_index)
        embeddings = (
            self.structural_embeddings()
            if structural_embeddings is None
            else structural_embeddings
        )
        expected = (self.num_capacity_levels, self.hypernetwork.embedding_dim)
        if tuple(embeddings.shape) != expected:
            raise ValueError(
                f"structural_embeddings must have shape {expected}, "
                f"got {tuple(embeddings.shape)}"
            )
        scores = self.layer_logits(layer_index, embeddings).mean(dim=0)
        return scores - scores.mean()

    def forward(
        self,
        layer_index: int,
        retention_ratios: Tensor,
        structural_embeddings: Optional[Tensor] = None,
        temperature: Optional[float] = None,
    ) -> SparsePatternOutput:
        """Build exact nested masks for one Transformer layer.

        The forward values are hard binary Top-k masks.  Gradients with respect
        to channel scores and retention ratios use a sigmoid straight-through
        approximation around each ranking threshold.
        """

        self._validate_layer_index(layer_index)
        if retention_ratios.ndim != 1:
            raise ValueError("retention_ratios must be a one-dimensional tensor")
        if retention_ratios.numel() != self.num_capacity_levels:
            raise ValueError("retention_ratios must contain one value per capacity level")
        detached_ratios = retention_ratios.detach()
        if torch.any(detached_ratios <= 0) or torch.any(detached_ratios > 1):
            raise ValueError("retention ratios must lie in (0, 1]")
        if torch.any(detached_ratios[1:] <= detached_ratios[:-1]):
            raise ValueError("retention ratios must be strictly increasing")

        tau = self.mask_temperature if temperature is None else float(temperature)
        if tau <= 0:
            raise ValueError("temperature must be positive")

        scores = self.channel_scores(layer_index, structural_embeddings)
        masks, widths = self._nested_topk_masks(scores, retention_ratios, tau)
        return SparsePatternOutput(
            channel_scores=scores,
            masks=masks,
            retention_ratios=retention_ratios,
            retained_channels=widths,
        )

    def all_layers(
        self,
        retention_ratios: Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[SparsePatternOutput, ...]:
        """Build sparse patterns for all layers from ``[L, N]`` ratios."""

        expected = (self.num_layers, self.num_capacity_levels)
        if tuple(retention_ratios.shape) != expected:
            raise ValueError(
                f"retention_ratios must have shape {expected}, "
                f"got {tuple(retention_ratios.shape)}"
            )
        embeddings = self.structural_embeddings()
        return tuple(
            self(
                layer_index,
                retention_ratios[layer_index],
                structural_embeddings=embeddings,
                temperature=temperature,
            )
            for layer_index in range(self.num_layers)
        )

    @staticmethod
    def _nested_topk_masks(
        scores: Tensor,
        retention_ratios: Tensor,
        temperature: float,
    ) -> Tuple[Tensor, Tensor]:
        d_ff = scores.numel()
        widths = torch.round(retention_ratios.detach() * d_ff).long()
        widths = widths.clamp(1, d_ff)

        # A single detached ordering guarantees exact nesting even if scores tie.
        ordering = torch.argsort(scores.detach(), descending=True)
        masks = []
        for ratio, width in zip(retention_ratios, widths):
            k = int(width.item())
            hard = torch.zeros_like(scores)
            hard.scatter_(0, ordering[:k], 1.0)

            threshold_detached = scores.detach()[ordering[k - 1]]
            threshold = threshold_detached - (
                inverse_sigmoid(ratio) - inverse_sigmoid(ratio.detach())
            )
            soft = torch.sigmoid((scores - threshold) / temperature)
            masks.append((hard - soft).detach() + soft)
        return torch.stack(masks, dim=0), widths

    def _validate_layer_index(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index must be in [0, {self.num_layers}), got {layer_index}"
            )

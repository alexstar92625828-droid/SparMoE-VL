"""Analytical CLIP ViT-L/14 MACs used by main-table FFN baselines."""

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import Tensor


@dataclass(frozen=True)
class TowerMacs:
    dense_total_g: float
    sparse_total_g: float
    dense_ffn_g: float
    sparse_ffn_g: float

    @property
    def ffn_reduction_percent(self) -> float:
        return 100.0 * (1.0 - self.sparse_ffn_g / self.dense_ffn_g)

    def as_dict(self) -> Dict[str, float]:
        result = asdict(self)
        result["ffn_reduction_percent"] = self.ffn_reduction_percent
        return result


def clip_vitl14_vision_static_ffn_macs(
    layer_ratios: Tensor,
    dense_cls: bool = True,
) -> TowerMacs:
    """Match the table-1 MACs convention, including a Dense CLS FFN path."""

    ratios = _validated_ratios(layer_ratios, expected_layers=24)
    d_model = 1024
    d_ff = 4096
    num_heads = 16
    num_patches = 256
    num_tokens = num_patches + 1
    head_dim = d_model // num_heads

    patch_embedding = num_patches * (3 * 14 * 14) * d_model
    attention = 24 * (
        num_tokens * d_model * (3 * d_model)
        + 2 * num_heads * num_tokens * num_tokens * head_dim
        + num_tokens * d_model * d_model
    )
    dense_ffn = 24 * num_tokens * 2 * d_model * d_ff
    if dense_cls:
        sparse_token_ratio_sum = 24 + num_patches * float(ratios.sum())
    else:
        sparse_token_ratio_sum = num_tokens * float(ratios.sum())
    sparse_ffn = sparse_token_ratio_sum * 2 * d_model * d_ff
    projection = d_model * 768
    return TowerMacs(
        dense_total_g=(patch_embedding + attention + dense_ffn + projection) / 1e9,
        sparse_total_g=(patch_embedding + attention + sparse_ffn + projection) / 1e9,
        dense_ffn_g=dense_ffn / 1e9,
        sparse_ffn_g=sparse_ffn / 1e9,
    )


def clip_vitl14_text_static_ffn_macs(layer_ratios: Tensor) -> TowerMacs:
    """Match the table-2 convention, which reports SOS within the static ratio."""

    ratios = _validated_ratios(layer_ratios, expected_layers=12)
    d_model = 768
    d_ff = 3072
    num_heads = 12
    sequence_length = 77
    head_dim = d_model // num_heads

    attention = 12 * (
        3 * sequence_length * d_model * d_model
        + 2 * num_heads * sequence_length * sequence_length * head_dim
        + sequence_length * d_model * d_model
    )
    dense_ffn = 12 * 2 * sequence_length * d_model * d_ff
    sparse_ffn = 2 * sequence_length * d_model * d_ff * float(ratios.sum())
    return TowerMacs(
        dense_total_g=(attention + dense_ffn) / 1e9,
        sparse_total_g=(attention + sparse_ffn) / 1e9,
        dense_ffn_g=dense_ffn / 1e9,
        sparse_ffn_g=sparse_ffn / 1e9,
    )


def clip_vitl14_text_routed_ffn_macs(
    layer_ratios: Tensor,
    dense_first_position: bool = True,
) -> TowerMacs:
    """Text-tower MACs from measured token-routing ratios.

    The generalization evaluator routes positions 1--76 and keeps the first
    text position on the Dense FFN path. This is distinct from static text
    baselines, whose reported ratio applies to all 77 positions.
    """

    ratios = _validated_ratios(layer_ratios, expected_layers=12)
    d_model = 768
    d_ff = 3072
    num_heads = 12
    sequence_length = 77
    head_dim = d_model // num_heads
    attention = 12 * (
        3 * sequence_length * d_model * d_model
        + 2 * num_heads * sequence_length * sequence_length * head_dim
        + sequence_length * d_model * d_model
    )
    dense_ffn = 12 * 2 * sequence_length * d_model * d_ff
    routed_positions = sequence_length - 1 if dense_first_position else sequence_length
    dense_positions = 12 if dense_first_position else 0
    sparse_ffn = (dense_positions + routed_positions * float(ratios.sum())) * 2 * d_model * d_ff
    return TowerMacs(
        dense_total_g=(attention + dense_ffn) / 1e9,
        sparse_total_g=(attention + sparse_ffn) / 1e9,
        dense_ffn_g=dense_ffn / 1e9,
        sparse_ffn_g=sparse_ffn / 1e9,
    )


def _validated_ratios(layer_ratios: Tensor, expected_layers: int) -> Tensor:
    ratios = torch.as_tensor(layer_ratios, dtype=torch.float64).reshape(-1)
    if ratios.numel() != expected_layers:
        raise ValueError(f"expected {expected_layers} layer ratios")
    if torch.any(ratios <= 0) or torch.any(ratios > 1):
        raise ValueError("layer ratios must lie in (0, 1]")
    return ratios

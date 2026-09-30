"""Evaluate one visual budget-sweep checkpoint on COCO and Flickr30k."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..baselines.retrieval import (
    COCO_ANNOTATIONS,
    COCO_IMAGES,
    FLICKR_ANNOTATIONS,
    FLICKR_IMAGES,
    EvaluationImages,
    RetrievalDataset,
    encode_texts,
    prepare_coco,
    prepare_flickr30k,
    retrieval_metrics,
)
from ..common.macs import clip_vitl14_vision_static_ffn_macs
from ..paths import repository_root, workspace_root
from ..vision.encoder import SparMoEVisionEncoder
from .vision_budget_sweep import (
    DATASET_SHA256,
    MODEL_NAME,
    NUM_LAYERS,
    inspect_checkpoint,
    load_encoder,
)


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
DEFAULT_PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
N_PATCHES = 256


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--coco-annotations", type=Path, default=COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=COCO_IMAGES)
    parser.add_argument("--flickr-annotations", type=Path, default=FLICKR_ANNOTATIONS)
    parser.add_argument("--flickr-images", type=Path, default=FLICKR_IMAGES)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.checkpoint, "Stage-2 checkpoint"),
        (args.pretrained, "Dense CLIP checkpoint"),
        (args.coco_annotations, "COCO annotations"),
        (args.flickr_annotations, "Flickr30k annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    for path, label in (
        (args.coco_images, "COCO images"),
        (args.flickr_images, "Flickr30k images"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"missing {label}: {path}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and worker count non-negative")


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@torch.inference_mode()
def encode_retrieval_corpus(
    encoder: SparMoEVisionEncoder,
    preprocess: Any,
    tokenizer: Any,
    corpus: RetrievalDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    name: str,
    collect_routing: bool,
) -> dict[str, Any]:
    """Encode dense/sparse images together and calculate retrieval metrics."""

    text_features = encode_texts(
        encoder.clip_model,
        tokenizer,
        corpus.captions,
        str(device),
        batch_size,
        f"{name} text",
    )
    loader = DataLoader(
        EvaluationImages(corpus.image_paths, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    dense_batches: list[Tensor] = []
    sparse_batches: list[Tensor] = []
    ratio_sums = torch.zeros(NUM_LAYERS, dtype=torch.float64)
    routed_tokens = torch.zeros(NUM_LAYERS, dtype=torch.float64)
    usage_sums = torch.zeros(NUM_LAYERS, encoder.num_capacity_levels, dtype=torch.float64)
    maximum_ratios: Optional[Tensor] = None

    for images in tqdm(loader, desc=f"{name} dense + sparse images"):
        images = images.to(device, non_blocking=device.type == "cuda")
        dense_batches.append(encoder.dense_features(images).cpu())
        output = encoder(images, routing_mode="learned")
        sparse_batches.append(output.features.cpu())
        if collect_routing:
            batch_maximums = []
            for position, layer in enumerate(output.layers):
                gates = layer.routing.gates.detach().float()
                ratios = layer.sparse_pattern.masks.detach().float().mean(dim=-1)
                ratio_sums[position] += float((gates @ ratios).sum())
                routed_tokens[position] += gates.shape[0]
                usage_sums[position] += gates.sum(dim=0).cpu().double()
                batch_maximums.append(float(ratios.max()))
            current_maximums = torch.tensor(batch_maximums, dtype=torch.float64)
            if maximum_ratios is None:
                maximum_ratios = current_maximums
            elif not torch.equal(maximum_ratios, current_maximums):
                raise RuntimeError("deterministic nested expert widths changed between batches")

    dense_images = torch.cat(dense_batches)
    sparse_images = torch.cat(sparse_batches)
    result: dict[str, Any] = {
        "images": len(corpus.image_paths),
        "captions": len(corpus.captions),
        "dense_metrics": retrieval_metrics(
            dense_images,
            text_features,
            corpus.caption_image_indices,
        ),
        "sparse_metrics": retrieval_metrics(
            sparse_images,
            text_features,
            corpus.caption_image_indices,
        ),
        "feature_cosine_to_dense": float(
            F.cosine_similarity(sparse_images, dense_images, dim=-1).mean()
        ),
    }
    if collect_routing:
        if torch.any(routed_tokens == 0) or maximum_ratios is None:
            raise RuntimeError("COCO evaluation did not route every visual layer")
        result["layer_routed_patch_ratios"] = (ratio_sums / routed_tokens).tolist()
        result["layer_maximum_patch_ratios"] = maximum_ratios.tolist()
        result["expert_usage_by_layer"] = (usage_sums / routed_tokens[:, None]).tolist()
    return result


def active_visual_parameters(
    encoder: SparMoEVisionEncoder,
    layer_ratios: Sequence[float],
) -> tuple[float, float]:
    """Return Dense and token-average sparse visual parameter counts in M."""

    if len(layer_ratios) != NUM_LAYERS:
        raise ValueError(f"expected {NUM_LAYERS} visual layer ratios")
    dense_total = sum(parameter.numel() for parameter in encoder.clip_model.visual.parameters())
    dense_ffn = 0
    sparse_ffn = 0.0
    for ratio, block in zip(layer_ratios, encoder._blocks):
        mlp = block.mlp
        model_dim = mlp.c_fc.weight.shape[1]
        ffn_dim = mlp.c_fc.weight.shape[0]
        dense_ffn += sum(parameter.numel() for parameter in mlp.parameters())
        executed_ratio = (1.0 + N_PATCHES * float(ratio)) / (N_PATCHES + 1)
        active_channels = ffn_dim * executed_ratio
        sparse_ffn += (
            model_dim * active_channels
            + active_channels
            + active_channels * model_dim
            + mlp.c_proj.bias.numel()
        )
    return dense_total / 1e6, (dense_total - dense_ffn + sparse_ffn) / 1e6


def maximum_active_visual_parameters(
    encoder: SparMoEVisionEncoder,
    maximum_ratios: Sequence[float],
) -> float:
    """Return the sparse-backbone upper bound from each layer's widest expert."""

    if len(maximum_ratios) != NUM_LAYERS:
        raise ValueError(f"expected {NUM_LAYERS} visual layer ratios")
    dense_total = sum(parameter.numel() for parameter in encoder.clip_model.visual.parameters())
    dense_ffn = 0
    sparse_ffn = 0
    for ratio, block in zip(maximum_ratios, encoder._blocks):
        mlp = block.mlp
        model_dim = mlp.c_fc.weight.shape[1]
        ffn_dim = mlp.c_fc.weight.shape[0]
        width = max(1, min(ffn_dim, round(ffn_dim * float(ratio))))
        dense_ffn += sum(parameter.numel() for parameter in mlp.parameters())
        sparse_ffn += model_dim * width + width + width * model_dim + mlp.c_proj.bias.numel()
    return (dense_total - dense_ffn + sparse_ffn) / 1e6


def _atomic_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, output)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    metadata = inspect_checkpoint(args.checkpoint)
    if args.check_only:
        print(json.dumps(metadata, indent=2, ensure_ascii=True))
        return

    device = resolve_device(args.device)
    encoder, metadata, preprocess, tokenizer = load_encoder(
        args.checkpoint,
        args.pretrained,
        device,
    )
    coco = encode_retrieval_corpus(
        encoder,
        preprocess,
        tokenizer,
        prepare_coco(args.coco_annotations, args.coco_images),
        device,
        args.batch_size,
        args.num_workers,
        "COCO",
        collect_routing=True,
    )
    flickr = encode_retrieval_corpus(
        encoder,
        preprocess,
        tokenizer,
        prepare_flickr30k(args.flickr_annotations, args.flickr_images),
        device,
        args.batch_size,
        args.num_workers,
        "Flickr30k",
        collect_routing=False,
    )

    layer_ratios = coco.pop("layer_routed_patch_ratios")
    maximum_ratios = coco.pop("layer_maximum_patch_ratios")
    expert_usage = coco.pop("expert_usage_by_layer")
    macs = clip_vitl14_vision_static_ffn_macs(
        torch.tensor(layer_ratios, dtype=torch.float64),
        dense_cls=True,
    ).as_dict()
    dense_active, sparse_active = active_visual_parameters(encoder, layer_ratios)
    maximum_active = maximum_active_visual_parameters(encoder, maximum_ratios)
    controller_parameters = (
        sum(parameter.numel() for parameter in encoder.trainable_parameters()) / 1e6
    )
    retention = 25.0 * sum(
        sparse_r1[key] / dense_r1[key]
        for key in ("i2t_r1", "t2i_r1")
        for sparse_r1, dense_r1 in (
            (coco["sparse_metrics"], coco["dense_metrics"]),
            (flickr["sparse_metrics"], flickr["dense_metrics"]),
        )
    )

    payload = {
        "format_version": 1,
        "method": "SparMoE-VL",
        "study": "vision_budget_sweep",
        "model_name": MODEL_NAME,
        "checkpoint": metadata,
        "training_data": {
            "dataset": "ShareGPT4V",
            "candidate_pool_size": metadata["pool_size"],
            "data_seed": metadata["data_seed"],
            "ordered_sha256": metadata["dataset_sha256"],
            "same_candidate_pool_in_both_stages": True,
        },
        "evaluation": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "routing": "learned_argmax",
            "special_cls_position_dense": True,
            "efficiency_basis": "COCO_dataset_average_routed_patch_width",
        },
        "dense": {
            "active_visual_parameters_m": dense_active,
            "coco": coco["dense_metrics"],
            "flickr30k": flickr["dense_metrics"],
        },
        "sparse": {
            "active_visual_parameters_m": sparse_active,
            "maximum_active_visual_parameters_m": maximum_active,
            "controller_parameters_m": controller_parameters,
            "total_macs_g": macs["sparse_total_g"],
            "ffn_macs_g": macs["sparse_ffn_g"],
            "ffn_reduction_percent": macs["ffn_reduction_percent"],
            "layer_routed_patch_ratios": layer_ratios,
            "layer_maximum_patch_ratios": maximum_ratios,
            "expert_usage_by_layer": expert_usage,
            "coco": coco["sparse_metrics"],
            "flickr30k": flickr["sparse_metrics"],
            "average_r1_retention_percent": retention,
            "feature_cosine_to_dense": {
                "coco": coco["feature_cosine_to_dense"],
                "flickr30k": flickr["feature_cosine_to_dense"],
            },
        },
    }
    if payload["training_data"]["ordered_sha256"] != DATASET_SHA256:
        raise RuntimeError("training-data identity changed after checkpoint loading")
    _atomic_json(payload, args.output)
    print(
        f"p={metadata['target_ratio']:.1f} seed={metadata['training_seed']} "
        f"FFN={macs['sparse_ffn_g']:.2f}G "
        f"reduction={macs['ffn_reduction_percent']:.2f}% "
        f"average_R1_retention={retention:.2f}%"
    )
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()

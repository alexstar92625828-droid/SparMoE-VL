"""Evaluate the two-stage CLIP ViT-L/14 vision main experiment on COCO."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import open_clip
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..common.data import ImagePathDataset, RetrievalCorpus, load_coco_retrieval
from ..common.macs import clip_vitl14_vision_static_ffn_macs
from ..common.metrics import retrieval_recalls
from ..paths import repository_root, workspace_root
from .encoder import SparMoEVisionEncoder


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
DEFAULT_PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "main" / "vision" / "seed_42" / "stage2" / "best.pt"
)
DEFAULT_COCO_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "coco" / "annotations" / "captions_val2017.json"
)
DEFAULT_COCO_IMAGES = RESEARCH_ROOT / "data" / "eval" / "coco" / "val2017"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "vision_clip_vitl14_p70" / "coco_evaluation.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two-stage vision SparMoE-VL on COCO val2017"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--coco-annotations", type=Path, default=DEFAULT_COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=DEFAULT_COCO_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--max-coco-images", type=int, default=5_000)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.checkpoint, "SparMoE checkpoint"),
        (args.pretrained, "pretrained CLIP checkpoint"),
        (args.coco_annotations, "COCO annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not args.coco_images.is_dir():
        raise FileNotFoundError(f"missing COCO image root: {args.coco_images}")
    if args.batch_size <= 0 or args.max_coco_images <= 0:
        raise ValueError("batch-size and max-coco-images must be positive")
    if args.num_workers < 0 or args.num_threads < 0:
        raise ValueError("worker and thread counts must be non-negative")


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_encoder(
    checkpoint_path: Path,
    pretrained: Path,
    device: torch.device,
) -> Tuple[SparMoEVisionEncoder, dict, Any, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("invalid SparMoE checkpoint")
    if checkpoint.get("modality") != "vision":
        raise ValueError("checkpoint is not a vision SparMoE checkpoint")
    if checkpoint.get("method") != "sparmoe_vl_two_stage" or checkpoint.get("stage") != 2:
        raise ValueError("the main evaluation requires a two-stage Stage-2 checkpoint")
    encoder_state = checkpoint.get("encoder")
    if not isinstance(encoder_state, dict):
        raise ValueError("checkpoint does not contain encoder state")

    model_name = checkpoint.get("model_name", "ViT-L-14")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=str(pretrained),
        force_quick_gelu=True,
    )
    clip_model = clip_model.to(device).eval()
    encoder = SparMoEVisionEncoder(
        clip_model=clip_model,
        sparse_layers=checkpoint.get("sparse_layers"),
        target_ratio=float(checkpoint["target_ratio"]),
        capacity_factors=checkpoint["capacity_factors"],
        router_temperature=float(checkpoint["temperature"]),
        mask_temperature=float(checkpoint["temperature"]),
    ).to(device)
    encoder.budget.load_state_dict(encoder_state["budget"], strict=True)
    encoder.sparse_pattern_generator.load_state_dict(
        encoder_state["sparse_pattern_generator"],
        strict=True,
    )
    encoder.routers.load_state_dict(encoder_state["routers"], strict=True)
    encoder.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return encoder, checkpoint, preprocess, tokenizer


@torch.no_grad()
def encode_texts(
    encoder: SparMoEVisionEncoder,
    tokenizer: Any,
    captions: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Tensor:
    features = []
    for start in tqdm(range(0, len(captions), batch_size), desc="COCO text"):
        tokens = tokenizer(list(captions[start : start + batch_size])).to(device)
        features.append(F.normalize(encoder.clip_model.encode_text(tokens), dim=-1).cpu())
    return torch.cat(features)


@torch.no_grad()
def encode_images(
    encoder: SparMoEVisionEncoder,
    preprocess: Any,
    corpus: RetrievalCorpus,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    loader = DataLoader(
        ImagePathDataset(corpus.image_paths, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    num_sparse_layers = len(encoder.sparse_layers)
    num_levels = encoder.num_capacity_levels
    ratio_sums = torch.zeros(num_sparse_layers, dtype=torch.float64)
    routed_tokens = torch.zeros(num_sparse_layers, dtype=torch.float64)
    usage_sums = torch.zeros(num_sparse_layers, num_levels, dtype=torch.float64)
    dense_features = []
    sparse_features = []

    for images in tqdm(loader, desc="COCO dense + SparMoE images"):
        images = images.to(device, non_blocking=device.type == "cuda")
        dense_features.append(encoder.dense_features(images).cpu())
        output = encoder(images, routing_mode="learned")
        sparse_features.append(output.features.cpu())
        for sparse_position, layer in enumerate(output.layers):
            gates = layer.routing.gates.detach().float()
            ratios = output.retention_ratios[sparse_position].detach().float()
            ratio_sums[sparse_position] += float((gates @ ratios).sum())
            routed_tokens[sparse_position] += gates.shape[0]
            usage_sums[sparse_position] += gates.sum(dim=0).cpu().double()

    if torch.any(routed_tokens == 0):
        raise RuntimeError("one or more sparse layers routed no tokens")
    return (
        torch.cat(dense_features),
        torch.cat(sparse_features),
        ratio_sums / routed_tokens,
        usage_sums / routed_tokens[:, None],
    )


def table_row(
    method: str,
    macs: Dict[str, float],
    recalls: Dict[str, float],
) -> str:
    return (
        f"{method} | {macs['sparse_total_g']:.2f} | "
        f"{macs['sparse_ffn_g']:.2f} | "
        f"{macs['ffn_reduction_percent']:.2f}% | "
        f"{recalls['I2T_R1']:.2f} | {recalls['I2T_R5']:.2f} | "
        f"{recalls['I2T_R10']:.2f} | {recalls['T2I_R1']:.2f} | "
        f"{recalls['T2I_R5']:.2f} | {recalls['T2I_R10']:.2f}"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.check_only:
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        print(
            f"paths valid; checkpoint_step={checkpoint.get('step')} "
            f"target_ratio={checkpoint.get('target_ratio')}"
        )
        return
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    device = resolve_device(args.device)
    encoder, checkpoint, preprocess, tokenizer = load_encoder(
        args.checkpoint,
        args.pretrained,
        device,
    )
    corpus = load_coco_retrieval(
        args.coco_annotations,
        args.coco_images,
        max_images=args.max_coco_images,
    )
    text_features = encode_texts(
        encoder,
        tokenizer,
        corpus.captions,
        device,
        args.batch_size,
    )
    dense_images, sparse_images, sparse_ratios, expert_usage = encode_images(
        encoder,
        preprocess,
        corpus,
        device,
        args.batch_size,
        args.num_workers,
    )
    dense_recalls = retrieval_recalls(
        dense_images,
        text_features,
        corpus.caption_image_indices,
    )
    sparse_recalls = retrieval_recalls(
        sparse_images,
        text_features,
        corpus.caption_image_indices,
    )
    feature_cosine = float(F.cosine_similarity(sparse_images, dense_images, dim=-1).mean())

    all_layer_ratios = torch.ones(len(encoder._blocks), dtype=torch.float64)
    for sparse_position, layer_index in enumerate(encoder.sparse_layers):
        all_layer_ratios[layer_index] = sparse_ratios[sparse_position]
    macs = clip_vitl14_vision_static_ffn_macs(
        all_layer_ratios,
        dense_cls=True,
    ).as_dict()
    dense_macs = clip_vitl14_vision_static_ffn_macs(
        torch.ones_like(all_layer_ratios),
        dense_cls=True,
    ).as_dict()
    metric_keys = (
        "I2T_R1",
        "I2T_R5",
        "I2T_R10",
        "T2I_R1",
        "T2I_R5",
        "T2I_R10",
    )
    metric_deltas = {key: sparse_recalls[key] - dense_recalls[key] for key in metric_keys}
    retention = 100.0 * sum(sparse_recalls.values()) / sum(dense_recalls.values())

    summary = {
        "format_version": 1,
        "method": "sparmoe_vl_two_stage",
        "modality": "vision",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_training_metrics": checkpoint.get("metrics"),
        "model_name": checkpoint.get("model_name"),
        "pretrained": str(args.pretrained.resolve()),
        "target_ratio": checkpoint.get("target_ratio"),
        "capacity_factors": checkpoint.get("capacity_factors"),
        "coco_images": len(corpus.image_paths),
        "coco_texts": len(corpus.captions),
        "dense": dense_recalls,
        "sparmoe_vl": sparse_recalls,
        "metric_deltas": metric_deltas,
        "mean_metric_retention_percent": retention,
        "feature_cosine_to_dense": feature_cosine,
        "base_layer_ratios": encoder.budget.base_ratios().detach().cpu().tolist(),
        "routed_layer_ratios": sparse_ratios.tolist(),
        "expert_usage_by_layer": expert_usage.tolist(),
        "macs": macs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)

    print("\nTABLE 2 CHECK")
    print(
        table_row(
            "Dense check",
            {
                **dense_macs,
                "ffn_reduction_percent": 0.0,
            },
            dense_recalls,
        )
    )
    print(table_row("SparMoE-VL two-stage", macs, sparse_recalls))
    print(
        "metric_deltas="
        + json.dumps(
            {key: round(value, 4) for key, value in metric_deltas.items()},
            ensure_ascii=True,
        )
    )
    print(f"feature_cosine_to_dense={feature_cosine:.6f}")
    print(f"mean_metric_retention={retention:.4f}%")
    print(f"mean_routed_patch_ratio={float(sparse_ratios.mean()):.6f}")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()

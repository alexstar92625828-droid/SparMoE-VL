"""Evaluate one visual or text main checkpoint for paper Table 4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from ...baselines.retrieval import EvaluationImages, retrieval_metrics
from ...common.macs import (
    clip_vitl14_text_routed_ffn_macs,
    clip_vitl14_vision_static_ffn_macs,
)
from ...paths import repository_root, workspace_root
from .cache import classification_accuracy, image_loader, resolve_device
from .data import (
    CifarImages,
    ClassificationDataset,
    GeneralizationDatasets,
    add_dataset_arguments,
    datasets_from_args,
)
from .protocol import (
    DATASET_COUNTS,
    DENSE_TEXT_MACS_G,
    DENSE_VISION_MACS_G,
    MODEL_NAME,
    STUDY_NAME,
    inspect_checkpoint,
    load_main_vision_encoder,
    load_text_encoder,
    torch_load,
)


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
DEFAULT_PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    fixed_modality: Optional[str] = None,
) -> argparse.Namespace:
    description = __doc__
    if fixed_modality is not None:
        description = f"Evaluate one {fixed_modality} main checkpoint for paper Table 4."
    parser = argparse.ArgumentParser(description=description)
    if fixed_modality is None:
        parser.add_argument("--modality", choices=("vision", "text"), required=True)
    else:
        if fixed_modality not in ("vision", "text"):
            raise ValueError("fixed modality must be vision or text")
        parser.set_defaults(modality=fixed_modality)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    if fixed_modality != "text":
        parser.add_argument("--cifar-batch-size", type=int, default=32)
        parser.add_argument("--num-workers", type=int, default=8)
    else:
        parser.set_defaults(cifar_batch_size=32, num_workers=8)
    parser.add_argument("--check-only", action="store_true")
    add_dataset_arguments(parser)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.checkpoint, "main Stage-2 checkpoint"),
        (args.cache, "shared Dense cache"),
        (args.pretrained, "Dense CLIP checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if min(args.batch_size, args.cifar_batch_size) <= 0 or args.num_workers < 0:
        raise ValueError("batch sizes must be positive and workers non-negative")


def load_validated_cache(
    path: Path,
    datasets: GeneralizationDatasets,
) -> dict[str, Any]:
    cache = torch_load(path)
    expected = {
        "format_version": 1,
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "counts": DATASET_COUNTS,
        "dataset_identities": datasets.identities,
        "dense_macs": {
            "vision_g": DENSE_VISION_MACS_G,
            "text_g": DENSE_TEXT_MACS_G,
        },
    }
    for key, wanted in expected.items():
        if cache.get(key) != wanted:
            raise ValueError(f"Dense cache {key} differs from the Table-4 protocol")
    required = {"features", "prototypes", "labels", "retrieval_indices", "dense_metrics"}
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"Dense cache is missing fields: {sorted(missing)}")
    return cache


def classification_dataset(
    dataset: ClassificationDataset,
    preprocess: Any,
) -> Dataset[Tensor]:
    if isinstance(dataset.images, tuple):
        return EvaluationImages(dataset.images, preprocess)
    return CifarImages(dataset.images, preprocess)


@torch.inference_mode()
def encode_sparse_images(
    encoder: Any,
    dataset: Dataset[Tensor],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
    collect_ratios: bool,
) -> tuple[Tensor, Optional[list[float]]]:
    features = []
    ratio_sum = torch.zeros(24, dtype=torch.float64)
    image_count = 0
    for images in tqdm(
        image_loader(dataset, device, batch_size, num_workers),
        desc=description,
    ):
        current_batch = len(images)
        images = images.to(device, non_blocking=device.type == "cuda")
        output = encoder(images, routing_mode="learned")
        features.append(output.features.cpu())
        if collect_ratios:
            image_count += current_batch
            for position, layer in enumerate(output.layers):
                widths = layer.sparse_pattern.masks.detach().float().mean(dim=-1)
                usage = layer.routing.gates.detach().float().mean(dim=0)
                ratio_sum[position] += float((usage * widths).sum()) * current_batch
    ratios = (ratio_sum / image_count).tolist() if collect_ratios else None
    return torch.cat(features), ratios


@torch.inference_mode()
def classify_sparse_images(
    encoder: Any,
    dataset: Dataset[Tensor],
    labels: Tensor,
    prototypes: Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
) -> float:
    prototype_device = prototypes.to(device)
    correct = 0
    offset = 0
    for images in tqdm(
        image_loader(dataset, device, batch_size, num_workers),
        desc=description,
    ):
        images = images.to(device, non_blocking=device.type == "cuda")
        features = encoder(images, routing_mode="learned").features
        prediction = (features @ prototype_device.T).argmax(dim=1).cpu()
        targets = labels[offset : offset + len(images)]
        correct += int((prediction == targets).sum())
        offset += len(images)
    return 100.0 * correct / len(labels)


@torch.inference_mode()
def encode_sparse_texts(
    encoder: Any,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    description: str,
    collect_ratios: bool,
) -> tuple[Tensor, Optional[list[float]]]:
    features = []
    ratio_sum = torch.zeros(12, dtype=torch.float64)
    sample_count = 0
    for start in tqdm(range(0, len(texts), batch_size), desc=description):
        batch = list(texts[start : start + batch_size])
        tokens = tokenizer(batch).to(device)
        output = encoder(tokens, routing_mode="learned")
        features.append(output.features.cpu())
        if collect_ratios:
            sample_count += len(batch)
            for position, layer in enumerate(output.layers):
                widths = layer.sparse_pattern.masks.detach().float().mean(dim=-1)
                usage = layer.routing.gates.detach().float().mean(dim=0)
                ratio_sum[position] += float((usage * widths).sum()) * len(batch)
    ratios = (ratio_sum / sample_count).tolist() if collect_ratios else None
    return torch.cat(features), ratios


@torch.inference_mode()
def encode_sparse_prototypes(
    encoder: Any,
    tokenizer: Any,
    prompt_groups: Sequence[Sequence[str]],
    device: torch.device,
    description: str,
) -> Tensor:
    prototypes = []
    for prompts in tqdm(prompt_groups, desc=description):
        tokens = tokenizer(list(prompts)).to(device)
        prompt_features = encoder(tokens, routing_mode="learned").features
        prototypes.append(F.normalize(prompt_features.mean(dim=0), dim=-1).cpu())
    return torch.stack(prototypes)


def evaluate_vision(
    args: argparse.Namespace,
    cache: dict[str, Any],
    datasets: GeneralizationDatasets,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoder, metadata, preprocess, _ = load_main_vision_encoder(
        args.checkpoint,
        args.pretrained,
        device,
    )
    coco_features, ratios = encode_sparse_images(
        encoder,
        EvaluationImages(datasets.coco.image_paths, preprocess),
        device,
        args.batch_size,
        args.num_workers,
        "COCO sparse visual features",
        True,
    )
    if ratios is None:
        raise RuntimeError("visual routing statistics were not collected")
    flickr_features, _ = encode_sparse_images(
        encoder,
        EvaluationImages(datasets.flickr30k.image_paths, preprocess),
        device,
        args.batch_size,
        args.num_workers,
        "Flickr30k sparse visual features",
        False,
    )
    retrieval = {
        "coco": retrieval_metrics(
            coco_features,
            cache["features"]["texts"]["coco"],
            datasets.coco.caption_image_indices,
        ),
        "flickr30k": retrieval_metrics(
            flickr_features,
            cache["features"]["texts"]["flickr30k"],
            datasets.flickr30k.caption_image_indices,
        ),
    }
    classification = {}
    for name, dataset in (
        ("cifar100", datasets.cifar100),
        ("imagenet1k", datasets.imagenet1k),
        ("food101", datasets.food101),
    ):
        batch_size = args.cifar_batch_size if name == "cifar100" else args.batch_size
        classification[f"{name}_accuracy"] = classify_sparse_images(
            encoder,
            classification_dataset(dataset, preprocess),
            dataset.labels,
            cache["prototypes"][name],
            device,
            batch_size,
            args.num_workers,
            f"{name} sparse visual classification",
        )
    macs = clip_vitl14_vision_static_ffn_macs(
        torch.tensor(ratios, dtype=torch.float64),
        dense_cls=True,
    ).as_dict()
    return metadata, {
        "macs": {
            "vision_g": macs["sparse_total_g"],
            "vision_ffn_g": macs["sparse_ffn_g"],
            "text_g": DENSE_TEXT_MACS_G,
        },
        "retrieval": retrieval,
        "classification": classification,
        "layer_routed_ratios": ratios,
    }


def evaluate_text(
    args: argparse.Namespace,
    cache: dict[str, Any],
    datasets: GeneralizationDatasets,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    encoder, metadata, tokenizer = load_text_encoder(
        args.checkpoint,
        args.pretrained,
        device,
    )
    coco_texts, ratios = encode_sparse_texts(
        encoder,
        tokenizer,
        datasets.coco.captions,
        device,
        args.batch_size,
        "COCO sparse text features",
        True,
    )
    if ratios is None:
        raise RuntimeError("text routing statistics were not collected")
    flickr_texts, _ = encode_sparse_texts(
        encoder,
        tokenizer,
        datasets.flickr30k.captions,
        device,
        args.batch_size,
        "Flickr30k sparse text features",
        False,
    )
    retrieval = {
        "coco": retrieval_metrics(
            cache["features"]["images"]["coco"],
            coco_texts,
            datasets.coco.caption_image_indices,
        ),
        "flickr30k": retrieval_metrics(
            cache["features"]["images"]["flickr30k"],
            flickr_texts,
            datasets.flickr30k.caption_image_indices,
        ),
    }
    classification = {}
    for name, dataset in (
        ("cifar100", datasets.cifar100),
        ("imagenet1k", datasets.imagenet1k),
        ("food101", datasets.food101),
    ):
        prototypes = encode_sparse_prototypes(
            encoder,
            tokenizer,
            dataset.prompt_groups,
            device,
            f"{name} sparse text prototypes",
        )
        classification[f"{name}_accuracy"] = classification_accuracy(
            cache["features"]["images"][name],
            prototypes,
            cache["labels"][name],
        )
    macs = clip_vitl14_text_routed_ffn_macs(
        torch.tensor(ratios, dtype=torch.float64),
        dense_first_position=True,
    ).as_dict()
    return metadata, {
        "macs": {
            "vision_g": DENSE_VISION_MACS_G,
            "text_g": macs["sparse_total_g"],
            "text_ffn_g": macs["sparse_ffn_g"],
        },
        "retrieval": retrieval,
        "classification": classification,
        "layer_routed_ratios": ratios,
    }


def save_result(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    os.replace(temporary, output)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    modality: Optional[str] = None,
) -> None:
    args = parse_args(argv, fixed_modality=modality)
    validate_args(args)
    datasets = datasets_from_args(args)
    metadata = inspect_checkpoint(args.checkpoint, args.modality)
    cache = load_validated_cache(args.cache, datasets)
    if args.check_only:
        print(
            json.dumps(
                {
                    "checkpoint": metadata,
                    "counts": datasets.counts,
                    "dataset_identities": datasets.identities,
                    "cache": str(args.cache.resolve()),
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return
    device = resolve_device(args.device)
    if args.modality == "vision":
        metadata, sparse = evaluate_vision(args, cache, datasets, device)
    else:
        metadata, sparse = evaluate_text(args, cache, datasets, device)
    result = {
        "format_version": 1,
        "method": "SparMoE-VL",
        "study": STUDY_NAME,
        "modality": args.modality,
        "checkpoint": metadata,
        "training_data": {
            "candidate_pool_size": metadata["pool_size"],
            "data_seed": metadata["data_seed"],
            "ordered_sha256": metadata["dataset_sha256"],
        },
        "evaluation": {
            "counts": datasets.counts,
            "dataset_identities": datasets.identities,
            "cache": str(args.cache.resolve()),
            "batch_size": args.batch_size,
            "cifar_vision_batch_size": args.cifar_batch_size,
            "num_workers": args.num_workers,
            "routing": "learned_argmax",
        },
        "dense": {
            "macs": cache["dense_macs"],
            "retrieval": {
                "coco": cache["dense_metrics"]["coco"],
                "flickr30k": cache["dense_metrics"]["flickr30k"],
            },
            "classification": {
                key: cache["dense_metrics"][key]
                for key in (
                    "cifar100_accuracy",
                    "imagenet1k_accuracy",
                    "food101_accuracy",
                )
            },
        },
        "sparse": sparse,
    }
    save_result(result, args.output)
    print(
        f"modality={args.modality} seed={metadata['training_seed']} "
        f"COCO_I2T_R1={sparse['retrieval']['coco']['i2t_r1']:.2f} "
        f"ImageNet1K={sparse['classification']['imagenet1k_accuracy']:.2f}"
    )
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()

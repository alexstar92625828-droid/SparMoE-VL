"""Build the shared Dense feature/prototype cache used by both Table-4 rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ...baselines.retrieval import EvaluationImages, retrieval_metrics
from ...paths import repository_root, workspace_root
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
)


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
DEFAULT_PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    add_dataset_arguments(parser)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.pretrained.is_file():
        raise FileNotFoundError(f"missing Dense CLIP checkpoint: {args.pretrained}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and worker count non-negative")


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def image_loader(
    dataset: Dataset[Tensor],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader[Tensor]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


@torch.inference_mode()
def encode_dense_images(
    model: nn.Module,
    dataset: Dataset[Tensor],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
) -> Tensor:
    features = []
    for images in tqdm(
        image_loader(dataset, device, batch_size, num_workers),
        desc=description,
    ):
        images = images.to(device, non_blocking=device.type == "cuda")
        features.append(F.normalize(model.encode_image(images), dim=-1).cpu())
    return torch.cat(features)


@torch.inference_mode()
def encode_dense_texts(
    model: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
    description: str,
) -> Tensor:
    features = []
    for start in tqdm(range(0, len(texts), batch_size), desc=description):
        tokens = tokenizer(list(texts[start : start + batch_size])).to(device)
        features.append(F.normalize(model.encode_text(tokens), dim=-1).cpu())
    return torch.cat(features)


@torch.inference_mode()
def encode_dense_prototypes(
    model: nn.Module,
    tokenizer: Any,
    prompt_groups: Sequence[Sequence[str]],
    device: torch.device,
    description: str,
) -> Tensor:
    prototypes = []
    for prompts in tqdm(prompt_groups, desc=description):
        tokens = tokenizer(list(prompts)).to(device)
        # Exact original protocol: average unnormalized Dense prompt features,
        # then normalize the resulting class prototype.
        feature = model.encode_text(tokens).mean(dim=0)
        prototypes.append(F.normalize(feature, dim=-1).cpu())
    return torch.stack(prototypes)


def classification_accuracy(
    image_features: Tensor,
    prototypes: Tensor,
    labels: Tensor,
) -> float:
    correct = 0
    labels = labels.long()
    for start in range(0, len(image_features), 4_096):
        batch = image_features[start : start + 4_096].float()
        prediction = (batch @ prototypes.float().T).argmax(dim=1)
        correct += int((prediction == labels[start : start + len(batch)]).sum())
    return 100.0 * correct / len(labels)


def classification_image_dataset(
    dataset: ClassificationDataset,
    preprocess: Any,
) -> Dataset[Tensor]:
    if isinstance(dataset.images, tuple):
        return EvaluationImages(dataset.images, preprocess)
    return CifarImages(dataset.images, preprocess)


def build_cache(
    model: nn.Module,
    preprocess: Any,
    tokenizer: Any,
    datasets: GeneralizationDatasets,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    dense_images = {
        "coco": encode_dense_images(
            model,
            EvaluationImages(datasets.coco.image_paths, preprocess),
            device,
            batch_size,
            num_workers,
            "COCO Dense images",
        ),
        "flickr30k": encode_dense_images(
            model,
            EvaluationImages(datasets.flickr30k.image_paths, preprocess),
            device,
            batch_size,
            num_workers,
            "Flickr30k Dense images",
        ),
    }
    for name, dataset in (
        ("cifar100", datasets.cifar100),
        ("imagenet1k", datasets.imagenet1k),
        ("food101", datasets.food101),
    ):
        dense_images[name] = encode_dense_images(
            model,
            classification_image_dataset(dataset, preprocess),
            device,
            batch_size,
            num_workers,
            f"{name} Dense images",
        )

    dense_texts = {
        "coco": encode_dense_texts(
            model,
            tokenizer,
            datasets.coco.captions,
            device,
            batch_size,
            "COCO Dense texts",
        ),
        "flickr30k": encode_dense_texts(
            model,
            tokenizer,
            datasets.flickr30k.captions,
            device,
            batch_size,
            "Flickr30k Dense texts",
        ),
    }
    dense_prototypes = {
        name: encode_dense_prototypes(
            model,
            tokenizer,
            dataset.prompt_groups,
            device,
            f"{name} Dense class prompts",
        )
        for name, dataset in (
            ("cifar100", datasets.cifar100),
            ("imagenet1k", datasets.imagenet1k),
            ("food101", datasets.food101),
        )
    }
    dense_metrics = {
        "coco": retrieval_metrics(
            dense_images["coco"],
            dense_texts["coco"],
            datasets.coco.caption_image_indices,
        ),
        "flickr30k": retrieval_metrics(
            dense_images["flickr30k"],
            dense_texts["flickr30k"],
            datasets.flickr30k.caption_image_indices,
        ),
        "cifar100_accuracy": classification_accuracy(
            dense_images["cifar100"],
            dense_prototypes["cifar100"],
            datasets.cifar100.labels,
        ),
        "imagenet1k_accuracy": classification_accuracy(
            dense_images["imagenet1k"],
            dense_prototypes["imagenet1k"],
            datasets.imagenet1k.labels,
        ),
        "food101_accuracy": classification_accuracy(
            dense_images["food101"],
            dense_prototypes["food101"],
            datasets.food101.labels,
        ),
    }
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "counts": datasets.counts,
        "dataset_identities": datasets.identities,
        "dense_macs": {
            "vision_g": DENSE_VISION_MACS_G,
            "text_g": DENSE_TEXT_MACS_G,
        },
        "dense_metrics": dense_metrics,
        "features": {
            "images": dense_images,
            "texts": dense_texts,
        },
        "prototypes": dense_prototypes,
        "labels": {
            "cifar100": datasets.cifar100.labels,
            "imagenet1k": datasets.imagenet1k.labels,
            "food101": datasets.food101.labels,
        },
        "retrieval_indices": {
            "coco": torch.tensor(
                datasets.coco.caption_image_indices,
                dtype=torch.long,
            ),
            "flickr30k": torch.tensor(
                datasets.flickr30k.caption_image_indices,
                dtype=torch.long,
            ),
        },
    }


def save_cache(payload: dict[str, Any], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Dense cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    sidecar = {
        key: payload[key]
        for key in (
            "format_version",
            "study",
            "model_name",
            "counts",
            "dataset_identities",
            "dense_macs",
            "dense_metrics",
        )
    }
    sidecar["cache"] = str(output.resolve())
    sidecar["size_bytes"] = output.stat().st_size
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, ensure_ascii=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    datasets = datasets_from_args(args)
    if args.check_only:
        print(
            json.dumps(
                {
                    "counts": datasets.counts,
                    "dataset_identities": datasets.identities,
                    "expected_counts": DATASET_COUNTS,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        return
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("install open_clip_torch before cache preparation") from error
    device = resolve_device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(args.pretrained),
        force_quick_gelu=True,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    cache = build_cache(
        model,
        preprocess,
        tokenizer,
        datasets,
        device,
        args.batch_size,
        args.num_workers,
    )
    save_cache(cache, args.output)
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()

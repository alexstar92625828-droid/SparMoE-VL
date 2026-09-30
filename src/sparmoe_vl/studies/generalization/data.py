"""Exact Table-4 dataset splits, label mappings, and zero-shot prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ...baselines.retrieval import RetrievalDataset, prepare_coco, prepare_flickr30k
from ...paths import repository_root, workspace_root
from .protocol import DATASET_COUNTS, EVALUATION_IDENTITIES


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
DATA_ROOT = RESEARCH_ROOT / "data" / "eval"
DEFAULT_COCO_ANNOTATIONS = DATA_ROOT / "coco" / "annotations" / "captions_val2017.json"
DEFAULT_COCO_IMAGES = DATA_ROOT / "coco" / "val2017"
DEFAULT_FLICKR_ANNOTATIONS = DATA_ROOT / "flickr30k" / "flickr_annotations_30k.csv"
DEFAULT_FLICKR_IMAGES = DATA_ROOT / "flickr30k" / "flickr30k-images"
DEFAULT_CIFAR_ROOT = DATA_ROOT / "cifar100" / "cifar-100-python"
DEFAULT_IMAGENET_IMAGES = DATA_ROOT / "imagenet" / "val"
DEFAULT_IMAGENET_LABELS = DATA_ROOT / "imagenet" / "ILSVRC2012_validation_ground_truth.txt"
DEFAULT_FOOD_ROOT = DATA_ROOT / "food101" / "food-101"


@dataclass(frozen=True)
class ClassificationDataset:
    images: Sequence[Path] | np.ndarray
    labels: Tensor
    prompt_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class GeneralizationDatasets:
    coco: RetrievalDataset
    flickr30k: RetrievalDataset
    cifar100: ClassificationDataset
    imagenet1k: ClassificationDataset
    food101: ClassificationDataset
    identities: dict[str, str]

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        return {
            "coco": {"images": len(self.coco.image_paths), "texts": len(self.coco.captions)},
            "flickr30k": {
                "images": len(self.flickr30k.image_paths),
                "texts": len(self.flickr30k.captions),
            },
            "cifar100": {
                "images": len(self.cifar100.labels),
                "classes": len(self.cifar100.prompt_groups),
            },
            "imagenet1k": {
                "images": len(self.imagenet1k.labels),
                "classes": len(self.imagenet1k.prompt_groups),
            },
            "food101": {
                "images": len(self.food101.labels),
                "classes": len(self.food101.prompt_groups),
            },
        }


class CifarImages(Dataset[Tensor]):
    def __init__(self, images: np.ndarray, preprocess: Any) -> None:
        self.images = images
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tensor:
        return self.preprocess(Image.fromarray(self.images[index]))


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coco-annotations", type=Path, default=DEFAULT_COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=DEFAULT_COCO_IMAGES)
    parser.add_argument(
        "--flickr-annotations",
        type=Path,
        default=DEFAULT_FLICKR_ANNOTATIONS,
    )
    parser.add_argument("--flickr-images", type=Path, default=DEFAULT_FLICKR_IMAGES)
    parser.add_argument("--cifar-root", type=Path, default=DEFAULT_CIFAR_ROOT)
    parser.add_argument("--imagenet-images", type=Path, default=DEFAULT_IMAGENET_IMAGES)
    parser.add_argument("--imagenet-labels", type=Path, default=DEFAULT_IMAGENET_LABELS)
    parser.add_argument("--food-root", type=Path, default=DEFAULT_FOOD_ROOT)


def validate_dataset_paths(args: argparse.Namespace) -> None:
    for path, label in (
        (args.coco_annotations, "COCO annotations"),
        (args.flickr_annotations, "Flickr30k annotations"),
        (args.cifar_root / "meta", "CIFAR-100 metadata"),
        (args.cifar_root / "test", "CIFAR-100 test split"),
        (args.imagenet_labels, "ImageNet validation labels"),
        (args.food_root / "meta" / "classes.txt", "Food-101 class names"),
        (args.food_root / "meta" / "test.json", "Food-101 test split"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    for path, label in (
        (args.coco_images, "COCO images"),
        (args.flickr_images, "Flickr30k images"),
        (args.imagenet_images, "ImageNet validation images"),
        (args.food_root / "images", "Food-101 images"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"missing {label}: {path}")


def datasets_from_args(args: argparse.Namespace) -> GeneralizationDatasets:
    validate_dataset_paths(args)
    return prepare_datasets(
        coco_annotations=args.coco_annotations,
        coco_images=args.coco_images,
        flickr_annotations=args.flickr_annotations,
        flickr_images=args.flickr_images,
        cifar_root=args.cifar_root,
        imagenet_images=args.imagenet_images,
        imagenet_labels=args.imagenet_labels,
        food_root=args.food_root,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(items: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_cifar100(root: Path) -> ClassificationDataset:
    with (root / "meta").open("rb") as handle:
        metadata = pickle.load(handle, encoding="bytes")
    class_names = [name.decode("utf-8") for name in metadata[b"fine_label_names"]]
    with (root / "test").open("rb") as handle:
        test = pickle.load(handle, encoding="bytes")
    images = test[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    prompts = tuple((f"a photo of a {name}.",) for name in class_names)
    return ClassificationDataset(
        images=images,
        labels=torch.tensor(test[b"fine_labels"], dtype=torch.long),
        prompt_groups=prompts,
    )


def prepare_imagenet1k(
    image_root: Path,
    validation_labels: Path,
) -> tuple[ClassificationDataset, Path]:
    try:
        import timm
        from open_clip.zero_shot_metadata import (
            IMAGENET_CLASSNAMES,
            OPENAI_IMAGENET_TEMPLATES,
        )
    except ImportError as error:
        raise RuntimeError("timm and open_clip_torch are required for ImageNet") from error
    synset_file = (
        Path(timm.__file__).resolve().parent / "data" / "_info" / "imagenet_synsets.txt"
    )
    with synset_file.open(encoding="utf-8") as handle:
        canonical_synsets = [line.strip() for line in handle if line.strip()]
    if len(canonical_synsets) != 1_000:
        raise ValueError(f"expected 1,000 canonical synsets, got {len(canonical_synsets)}")
    synset_to_index = {synset: index for index, synset in enumerate(canonical_synsets)}
    with validation_labels.open(encoding="utf-8") as handle:
        validation_synsets = [line.strip() for line in handle if line.strip()]
    paths = tuple(sorted(path for path in image_root.iterdir() if path.is_file()))
    if len(paths) != 50_000 or len(validation_synsets) != 50_000:
        raise ValueError(
            f"ImageNet count mismatch: images={len(paths)}, labels={len(validation_synsets)}"
        )
    missing = sorted(set(validation_synsets).difference(synset_to_index))
    if missing:
        raise ValueError(f"ImageNet validation labels contain unknown synsets: {missing[:5]}")
    prompts = tuple(
        tuple(template(class_name) for template in OPENAI_IMAGENET_TEMPLATES)
        for class_name in IMAGENET_CLASSNAMES
    )
    return (
        ClassificationDataset(
            images=paths,
            labels=torch.tensor(
                [synset_to_index[synset] for synset in validation_synsets],
                dtype=torch.long,
            ),
            prompt_groups=prompts,
        ),
        synset_file,
    )


def prepare_food101(root: Path) -> ClassificationDataset:
    with (root / "meta" / "classes.txt").open(encoding="utf-8") as handle:
        class_names = [line.strip().replace("_", " ") for line in handle if line.strip()]
    with (root / "meta" / "test.json").open(encoding="utf-8") as handle:
        split = json.load(handle)
    paths: list[Path] = []
    labels: list[int] = []
    for class_key, relative_paths in split.items():
        label = class_names.index(class_key.replace("_", " "))
        for relative_path in relative_paths:
            paths.append(root / "images" / f"{relative_path}.jpg")
            labels.append(label)
    prompts = tuple(
        (f"a photo of {class_name}, a type of food.",) for class_name in class_names
    )
    return ClassificationDataset(
        images=tuple(paths),
        labels=torch.tensor(labels, dtype=torch.long),
        prompt_groups=prompts,
    )


def prepare_datasets(
    *,
    coco_annotations: Path,
    coco_images: Path,
    flickr_annotations: Path,
    flickr_images: Path,
    cifar_root: Path,
    imagenet_images: Path,
    imagenet_labels: Path,
    food_root: Path,
) -> GeneralizationDatasets:
    coco = prepare_coco(coco_annotations, coco_images)
    flickr = prepare_flickr30k(flickr_annotations, flickr_images)
    cifar = prepare_cifar100(cifar_root)
    imagenet, synset_file = prepare_imagenet1k(imagenet_images, imagenet_labels)
    food = prepare_food101(food_root)
    identities = {
        "coco_annotations_sha256": file_sha256(coco_annotations),
        "flickr_annotations_sha256": file_sha256(flickr_annotations),
        "cifar100_test_sha256": file_sha256(cifar_root / "test"),
        "imagenet_labels_sha256": file_sha256(imagenet_labels),
        "imagenet_synsets_sha256": file_sha256(synset_file),
        "imagenet_order_sha256": sequence_sha256(
            [
                f"{path.name}:{int(label)}"
                for path, label in zip(imagenet.images, imagenet.labels)
            ]
        ),
        "food101_test_sha256": file_sha256(food_root / "meta" / "test.json"),
        "food101_order_sha256": sequence_sha256(
            [f"{Path(path).name}:{int(label)}" for path, label in zip(food.images, food.labels)]
        ),
    }
    datasets = GeneralizationDatasets(coco, flickr, cifar, imagenet, food, identities)
    if datasets.counts != DATASET_COUNTS:
        raise RuntimeError(
            f"Table-4 dataset counts changed: {datasets.counts} != {DATASET_COUNTS}"
        )
    if datasets.identities != EVALUATION_IDENTITIES:
        raise RuntimeError("Table-4 evaluation dataset identities changed")
    return datasets

"""Shared COCO and Flickr30k retrieval evaluation utilities."""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
COCO_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "coco" / "annotations" / "captions_val2017.json"
)
COCO_IMAGES = RESEARCH_ROOT / "data" / "eval" / "coco" / "val2017"
FLICKR_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "flickr30k" / "flickr_annotations_30k.csv"
)
FLICKR_IMAGES = RESEARCH_ROOT / "data" / "eval" / "flickr30k" / "flickr30k-images"


@dataclass(frozen=True)
class RetrievalDataset:
    image_paths: tuple[Path, ...]
    captions: tuple[str, ...]
    caption_image_indices: tuple[int, ...]


class EvaluationImages(Dataset[Tensor]):
    def __init__(self, paths: Sequence[Path], preprocess: Any) -> None:
        self.paths = tuple(Path(path) for path in paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                return self.preprocess(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"unable to read evaluation image: {path}") from error


def prepare_coco(
    annotations: Path = COCO_ANNOTATIONS,
    image_root: Path = COCO_IMAGES,
) -> RetrievalDataset:
    with Path(annotations).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    id_to_file = {item["id"]: item["file_name"] for item in payload["images"]}
    image_ids = list(id_to_file)[:5_000]
    id_to_index = {image_id: index for index, image_id in enumerate(image_ids)}
    paths = tuple(Path(image_root) / id_to_file[image_id] for image_id in image_ids)
    captions, mapping = [], []
    for annotation in payload["annotations"]:
        image_id = annotation["image_id"]
        if image_id in id_to_index:
            captions.append(annotation["caption"])
            mapping.append(id_to_index[image_id])
    if len(paths) != 5_000 or len(captions) != 25_014:
        raise RuntimeError(f"unexpected COCO counts: {len(paths)}, {len(captions)}")
    return RetrievalDataset(paths, tuple(captions), tuple(mapping))


def prepare_flickr30k(
    annotations: Path = FLICKR_ANNOTATIONS,
    image_root: Path = FLICKR_IMAGES,
) -> RetrievalDataset:
    paths, captions, mapping = [], [], []
    with Path(annotations).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "test":
                continue
            image_index = len(paths)
            paths.append(Path(image_root) / row["filename"])
            for caption in ast.literal_eval(row["raw"]):
                captions.append(str(caption))
                mapping.append(image_index)
    if len(paths) != 1_000 or len(captions) != 5_000:
        raise RuntimeError(f"unexpected Flickr30k counts: {len(paths)}, {len(captions)}")
    return RetrievalDataset(tuple(paths), tuple(captions), tuple(mapping))


def encode_images(
    model: nn.Module,
    preprocess: Any,
    paths: Sequence[Path],
    device: str,
    batch_size: int,
    workers: int,
    description: str,
) -> Tensor:
    loader = DataLoader(
        EvaluationImages(paths, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=workers > 0,
    )
    outputs = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=description):
            encoded = model.encode_image(images.to(device, non_blocking=True))
            outputs.append(F.normalize(encoded, dim=-1).cpu())
    return torch.cat(outputs)


def encode_texts(
    model: nn.Module,
    tokenizer: Any,
    captions: Sequence[str],
    device: str,
    batch_size: int,
    description: str,
) -> Tensor:
    outputs = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(captions), batch_size), desc=description):
            tokens = tokenizer(captions[start : start + batch_size]).to(device)
            outputs.append(F.normalize(model.encode_text(tokens), dim=-1).cpu())
    return torch.cat(outputs)


def retrieval_metrics(
    image_features: Tensor,
    text_features: Tensor,
    caption_image_indices: Sequence[int],
) -> dict[str, float]:
    similarity = image_features.float() @ text_features.float().T
    image_truths = [[] for _ in range(image_features.shape[0])]
    for caption_index, image_index in enumerate(caption_image_indices):
        image_truths[image_index].append(caption_index)
    metrics = {}
    for k in (1, 5, 10):
        top_text = similarity.topk(k, dim=1).indices
        image_to_text = sum(
            bool(set(top_text[index].tolist()) & set(image_truths[index]))
            for index in range(image_features.shape[0])
        )
        top_image = similarity.topk(k, dim=0).indices
        text_to_image = sum(
            caption_image_indices[index] in top_image[:, index].tolist()
            for index in range(len(caption_image_indices))
        )
        metrics[f"i2t_r{k}"] = 100.0 * image_to_text / image_features.shape[0]
        metrics[f"t2i_r{k}"] = 100.0 * text_to_image / len(caption_image_indices)
    return metrics

"""Dataset selection helpers shared by the two main experiment tables."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RetrievalCorpus:
    """Image paths, captions, and caption-to-image ground-truth indices."""

    image_paths: Tuple[Path, ...]
    captions: Tuple[str, ...]
    caption_image_indices: Tuple[int, ...]


class ImagePathDataset(Dataset[Tensor]):
    """Load image paths with a deterministic blank-image fallback."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        preprocess: Callable[[Image.Image], Tensor],
        fallback_size: Tuple[int, int] = (224, 224),
    ) -> None:
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        self.image_paths = tuple(Path(path) for path in image_paths)
        self.preprocess = preprocess
        self.fallback_size = tuple(int(value) for value in fallback_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        try:
            with Image.open(self.image_paths[index]) as image:
                return self.preprocess(image.convert("RGB"))
        except Exception:
            return self.preprocess(Image.new("RGB", self.fallback_size))


class ShareGPT4VImageTrainingDataset(Dataset[Tensor]):
    """Deterministic ShareGPT4V image subset shared by both training stages."""

    def __init__(
        self,
        annotations: Path,
        image_root: Path,
        preprocess: Callable[[Image.Image], Tensor],
        max_samples: int = 500_000,
        data_seed: int = 42,
    ) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        records = _load_json_list(annotations)
        random.Random(data_seed).shuffle(records)
        self.image_paths = []
        self.preprocess = preprocess
        digest = hashlib.sha256()
        for record in records:
            if not isinstance(record, dict):
                continue
            relative_path = record.get("image")
            if not relative_path:
                continue
            path = image_root / str(relative_path)
            if not path.is_file():
                continue
            self.image_paths.append(path)
            digest.update(str(relative_path).encode("utf-8"))
            digest.update(b"\0")
            if len(self.image_paths) >= max_samples:
                break
        if not self.image_paths:
            raise RuntimeError(f"no ShareGPT4V images found under {image_root}")
        self.data_seed = int(data_seed)
        self.ordered_sha256 = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        try:
            with Image.open(self.image_paths[index]) as image:
                return self.preprocess(image.convert("RGB"))
        except Exception:
            return self.preprocess(Image.new("RGB", (224, 224)))


class ShareGPT4VTextTrainingDataset(Dataset[Tensor]):
    """Deterministic ShareGPT4V caption subset shared by both training stages."""

    def __init__(
        self,
        annotations: Path,
        tokenizer: Callable[[Sequence[str]], Tensor],
        max_samples: int = 500_000,
        data_seed: int = 42,
    ) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        records = _load_json_list(annotations)
        random.Random(data_seed).shuffle(records)
        self.tokens = []
        digest = hashlib.sha256()
        for record in records:
            caption = extract_sharegpt4v_caption(record)
            if not caption:
                continue
            tokenized = tokenizer([caption])
            if not isinstance(tokenized, Tensor) or tokenized.ndim != 2:
                raise TypeError("OpenCLIP tokenizer must return [batch, context]")
            self.tokens.append(tokenized[0])
            digest.update(caption.encode("utf-8"))
            digest.update(b"\0")
            if len(self.tokens) >= max_samples:
                break
        if not self.tokens:
            raise RuntimeError(f"no ShareGPT4V captions found in {annotations}")
        self.data_seed = int(data_seed)
        self.ordered_sha256 = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> Tensor:
        return self.tokens[index]


def extract_sharegpt4v_caption(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    conversations = record.get("conversations", [])
    if isinstance(conversations, list):
        for conversation in conversations:
            if not isinstance(conversation, dict):
                continue
            if conversation.get("from") == "gpt":
                caption = conversation.get("value", "")
                if isinstance(caption, str) and caption:
                    return caption
    fallback = record.get("caption", "")
    return fallback if isinstance(fallback, str) else ""


def load_coco_retrieval(
    annotations: Path,
    image_root: Path,
    max_images: int = 5_000,
) -> RetrievalCorpus:
    """Load the canonical COCO val2017 retrieval ordering."""

    if max_images <= 0:
        raise ValueError("max_images must be positive")
    with annotations.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("COCO annotations must contain a JSON object")
    images = data.get("images")
    captions = data.get("annotations")
    if not isinstance(images, list) or not isinstance(captions, list):
        raise ValueError("COCO annotations must expose images and annotations")

    id_to_file = {
        item["id"]: item["file_name"]
        for item in images
        if isinstance(item, dict) and "id" in item and "file_name" in item
    }
    image_id_to_index = {}
    ordered_image_ids = []
    ordered_captions = []
    caption_image_indices = []
    for annotation in captions:
        if not isinstance(annotation, dict):
            continue
        image_id = annotation.get("image_id")
        caption = annotation.get("caption")
        if image_id not in id_to_file or not isinstance(caption, str):
            continue
        if image_id not in image_id_to_index:
            if len(image_id_to_index) >= max_images:
                continue
            image_id_to_index[image_id] = len(image_id_to_index)
            ordered_image_ids.append(image_id)
        ordered_captions.append(caption)
        caption_image_indices.append(image_id_to_index[image_id])

    if not ordered_image_ids or not ordered_captions:
        raise RuntimeError("no COCO retrieval samples were found")
    paths = tuple(image_root / id_to_file[image_id] for image_id in ordered_image_ids)
    return RetrievalCorpus(
        image_paths=paths,
        captions=tuple(ordered_captions),
        caption_image_indices=tuple(caption_image_indices),
    )


def _load_json_list(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"expected a JSON list in {path}")
    return records

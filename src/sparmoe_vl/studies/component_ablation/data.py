"""Training-data views used by the Table-8 ablations."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def extract_contrastive_caption(record: Any) -> str:
    """Reproduce the caption selection used by the historical contrastive run."""

    if not isinstance(record, dict):
        return ""
    caption = record.get("caption")
    if caption:
        return str(caption)
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list):
        return ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") == "gpt" and turn.get("value"):
            return str(turn["value"]).replace("<image>", " ").strip()
    return ""


class ShareGPT4VImageTextTrainingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Image-text view for replacing Dense feature-geometry supervision."""

    def __init__(
        self,
        annotations: Path,
        image_root: Path,
        preprocess: Any,
        tokenizer: Any,
        max_samples: int,
        data_seed: int,
    ) -> None:
        with annotations.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"expected a JSON list in {annotations}")
        random.Random(data_seed).shuffle(records)
        self.samples: list[tuple[Path, str]] = []
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        digest = hashlib.sha256()
        for record in records:
            if not isinstance(record, dict):
                continue
            relative = record.get("image")
            caption = extract_contrastive_caption(record)
            if not relative or not caption:
                continue
            path = image_root / str(relative)
            if not path.is_file():
                continue
            self.samples.append((path, caption))
            digest.update(str(relative).encode("utf-8"))
            digest.update(b"\0")
            if len(self.samples) >= max_samples:
                break
        if not self.samples:
            raise RuntimeError(f"no valid ShareGPT4V image-text pairs under {image_root}")
        self.data_seed = int(data_seed)
        self.ordered_sha256 = digest.hexdigest()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, caption = self.samples[index]
        try:
            with Image.open(path) as image:
                pixels = self.preprocess(image.convert("RGB"))
        except Exception:
            pixels = self.preprocess(Image.new("RGB", (224, 224)))
        tokens = self.tokenizer([caption])[0]
        return pixels, tokens


def ordered_paths(dataset: ShareGPT4VImageTextTrainingDataset) -> Sequence[Path]:
    return tuple(path for path, _ in dataset.samples)

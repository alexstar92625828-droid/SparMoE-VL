"""Shared model and exact-data utilities for visual FFN baselines."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()
MODEL_NAME = "ViT-L-14"
PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
ANNOTATIONS = RESEARCH_ROOT / "ShareGPT4V" / "annotations" / "sharegpt4v_1246k.json"
IMAGE_ROOT = RESEARCH_ROOT / "ShareGPT4V" / "images"

EXPECTED_PRETRAINED_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
EXPECTED_IMAGE_POOL_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"
EXPECTED_VISION_CAPTION_POOL_SHA256 = (
    "5f416e610ebf979b0969464ad5e9e640bf09201572a33256013083e46b3e1a25"
)
EXPECTED_PROCESSING_ORDER_SHA256 = {
    42: "eb305ba0e6c23225c2d1ac4dc7c7fe1757df6380e777c6425984552ba0da01f9",
    123: "5a5c36e66881834c1b508e548c670fba6ed2d6d4997d98b516cd28d61cf98c5b",
    2026: "b9756c6260cc0165c1574c29784063474ea3a430b18a8ea5221b1a9a8dbcfd27",
}

DATA_SEED = 42
POOL_SIZE = 500_000
N_LAYERS = 24
N_TOKENS = 257
D_MODEL = 1_024
D_FFN = 4_096
EMBED_DIM = 768


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(values: Tensor) -> str:
    array = values.detach().cpu().to(torch.int64).numpy().astype("<i8", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, output)


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def visual_blocks(model: nn.Module) -> nn.ModuleList:
    return model.visual.transformer.resblocks


def create_clip(
    device: str | torch.device,
    pretrained: str | Path = PRETRAINED,
) -> tuple[nn.Module, Any, Any]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(pretrained),
        force_quick_gelu=True,
    )
    model = model.to(device).eval()
    model.requires_grad_(False)
    blocks = visual_blocks(model)
    if len(blocks) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} visual layers, got {len(blocks)}")
    if not bool(model.visual.transformer.batch_first):
        raise RuntimeError("TEAL visual implementation requires batch-first OpenCLIP blocks")
    for layer, block in enumerate(blocks):
        if tuple(block.mlp.c_fc.weight.shape) != (D_FFN, D_MODEL):
            raise RuntimeError(f"unexpected c_fc shape at visual layer {layer}")
        if tuple(block.mlp.c_proj.weight.shape) != (D_MODEL, D_FFN):
            raise RuntimeError(f"unexpected c_proj shape at visual layer {layer}")
    return model, preprocess, open_clip.get_tokenizer(MODEL_NAME)


def build_main_image_pool(
    annotations: Path = ANNOTATIONS,
    image_root: Path = IMAGE_ROOT,
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    """Build the exact ordered 500k image pool used by both main stages."""

    with Path(annotations).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("ShareGPT4V annotations must contain a JSON list")
    random.Random(DATA_SEED).shuffle(records)
    paths = []
    digest = hashlib.sha256()
    for record in records:
        if not isinstance(record, dict) or not record.get("image"):
            continue
        relative = str(record["image"])
        path = Path(image_root) / relative
        if not path.is_file():
            continue
        paths.append(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if len(paths) == POOL_SIZE:
            break
    fingerprint = digest.hexdigest()
    if len(paths) != POOL_SIZE:
        raise RuntimeError(f"visual main pool contains {len(paths):,}, expected {POOL_SIZE:,}")
    if fingerprint != EXPECTED_IMAGE_POOL_SHA256:
        raise RuntimeError(
            f"visual pool differs from the main experiment: {fingerprint} "
            f"!= {EXPECTED_IMAGE_POOL_SHA256}"
        )
    return tuple(paths), {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "unique_records": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": fingerprint,
    }


def extract_visual_pair_caption(record: Any) -> str:
    """Extract the paired text used by vision baselines that need CLIP logits."""

    if not isinstance(record, dict):
        return ""
    caption = record.get("caption")
    if caption:
        return str(caption).replace("<image>", " ").strip()
    conversations = record.get("conversations", [])
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            if turn.get("from") == "gpt" and turn.get("value"):
                return str(turn["value"]).replace("<image>", " ").strip()
    return ""


def build_main_image_text_pool(
    annotations: Path = ANNOTATIONS,
    image_root: Path = IMAGE_ROOT,
) -> tuple[tuple[Path, ...], tuple[str, ...], dict[str, Any]]:
    """Build the exact visual-main image pool and its paired CLIP captions."""

    with Path(annotations).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("ShareGPT4V annotations must contain a JSON list")
    random.Random(DATA_SEED).shuffle(records)
    paths: list[Path] = []
    captions: list[str] = []
    image_digest = hashlib.sha256()
    caption_digest = hashlib.sha256()
    for record in records:
        if not isinstance(record, dict) or not record.get("image"):
            continue
        relative = str(record["image"])
        path = Path(image_root) / relative
        if not path.is_file():
            continue
        caption = extract_visual_pair_caption(record)
        if not caption:
            raise RuntimeError("the visual main pool contains an empty paired caption")
        paths.append(path)
        captions.append(caption)
        image_digest.update(relative.encode("utf-8"))
        image_digest.update(b"\0")
        caption_digest.update(caption.encode("utf-8"))
        caption_digest.update(b"\0")
        if len(paths) == POOL_SIZE:
            break

    image_fingerprint = image_digest.hexdigest()
    caption_fingerprint = caption_digest.hexdigest()
    if len(paths) != POOL_SIZE:
        raise RuntimeError(
            f"visual paired pool contains {len(paths):,}, expected {POOL_SIZE:,}"
        )
    if image_fingerprint != EXPECTED_IMAGE_POOL_SHA256:
        raise RuntimeError("paired image pool differs from the visual main experiment")
    if caption_fingerprint != EXPECTED_VISION_CAPTION_POOL_SHA256:
        raise RuntimeError("paired captions for the visual main pool changed")
    return (
        tuple(paths),
        tuple(captions),
        {
            "data_seed": DATA_SEED,
            "pool_size": POOL_SIZE,
            "unique_records": POOL_SIZE,
            "uses_complete_main_pool": True,
            "dataset_sha256": image_fingerprint,
            "paired_caption_sha256": caption_fingerprint,
        },
    )


def full_pool_permutation(seed: int) -> Tensor:
    if seed not in EXPECTED_PROCESSING_ORDER_SHA256:
        raise ValueError(f"unsupported paper seed: {seed}")
    generator = torch.Generator().manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    indices = torch.randperm(POOL_SIZE, generator=generator)
    if tensor_sha256(indices) != EXPECTED_PROCESSING_ORDER_SHA256[seed]:
        raise RuntimeError("visual processing order differs from the main DataLoader")
    return indices


class IndexedImages(Dataset[Tensor]):
    def __init__(
        self,
        paths: Sequence[Path],
        indices: Tensor,
        preprocess: Any,
    ) -> None:
        if len(paths) != POOL_SIZE or indices.numel() != POOL_SIZE:
            raise ValueError("visual calibration requires the complete 500k pool")
        self.paths = tuple(Path(path) for path in paths)
        self.indices = indices.to(torch.int64).tolist()
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[self.indices[index]]
        try:
            with Image.open(path) as image:
                return self.preprocess(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"unable to read main-pool image: {path}") from error


class IndexedImageSlice(Dataset[Tensor]):
    """Read a validated ordered slice of the complete visual-main pool."""

    def __init__(
        self,
        paths: Sequence[Path],
        indices: Tensor,
        preprocess: Any,
    ) -> None:
        if len(paths) != POOL_SIZE:
            raise ValueError("visual calibration requires the complete 500k path pool")
        if indices.ndim != 1 or indices.numel() > POOL_SIZE:
            raise ValueError("invalid visual main-pool index slice")
        self.paths = tuple(Path(path) for path in paths)
        self.indices = indices.to(torch.int64).tolist()
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[self.indices[index]]
        try:
            with Image.open(path) as image:
                return self.preprocess(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"unable to read main-pool image: {path}") from error


class IndexedImageTextPairs(Dataset[tuple[Tensor, Tensor]]):
    """Read an ordered slice of the exact visual-main paired pool."""

    def __init__(
        self,
        paths: Sequence[Path],
        captions: Sequence[str],
        indices: Tensor,
        preprocess: Any,
        tokenizer: Any,
    ) -> None:
        if len(paths) != POOL_SIZE or len(captions) != POOL_SIZE:
            raise ValueError("visual paired calibration requires the complete 500k pool")
        if indices.ndim != 1 or indices.numel() > POOL_SIZE:
            raise ValueError("invalid visual paired-pool index slice")
        self.paths = tuple(Path(path) for path in paths)
        self.captions = tuple(str(caption) for caption in captions)
        self.indices = indices.to(torch.int64).tolist()
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        pool_index = self.indices[index]
        path = self.paths[pool_index]
        try:
            with Image.open(path) as image:
                pixels = self.preprocess(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"unable to read main-pool image: {path}") from error
        tokens = self.tokenizer([self.captions[pool_index]])[0]
        return pixels, tokens


def make_image_loader(
    paths: Sequence[Path],
    indices: Tensor,
    preprocess: Any,
    batch_size: int,
    workers: int,
    device: str,
) -> DataLoader[Tensor]:
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    return DataLoader(
        IndexedImages(paths, indices, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=workers > 0,
    )


def make_image_slice_loader(
    paths: Sequence[Path],
    indices: Tensor,
    preprocess: Any,
    batch_size: int,
    workers: int,
    device: str,
) -> DataLoader[Tensor]:
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    return DataLoader(
        IndexedImageSlice(paths, indices, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=workers > 0,
    )


def make_image_text_loader(
    paths: Sequence[Path],
    captions: Sequence[str],
    indices: Tensor,
    preprocess: Any,
    tokenizer: Any,
    batch_size: int,
    workers: int,
    device: str,
) -> DataLoader[tuple[Tensor, Tensor]]:
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    return DataLoader(
        IndexedImageTextPairs(paths, captions, indices, preprocess, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=workers > 0,
    )

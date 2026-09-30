"""Shared data, model, and retrieval utilities for text FFN baselines."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sparmoe_vl.paths import repository_root, workspace_root


PROJECT_ROOT = repository_root()
RESEARCH_ROOT = workspace_root()

MODEL_NAME = "ViT-L-14"
PRETRAINED = RESEARCH_ROOT / "models" / "ViT-L-14.pt"
ANNOTATIONS = RESEARCH_ROOT / "ShareGPT4V" / "annotations" / "sharegpt4v_1246k.json"
IMAGE_ROOT = RESEARCH_ROOT / "ShareGPT4V" / "images"
TOKEN_CACHE = PROJECT_ROOT / "data" / "cache" / "sharegpt4v_text_data_seed42_500k.pt"
IMAGE_FEATURE_CACHE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "sharegpt4v_clip_vitl14_image_features_data_seed42_500k.pt"
)

COCO_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "coco" / "annotations" / "captions_val2017.json"
)
COCO_IMAGES = RESEARCH_ROOT / "data" / "eval" / "coco" / "val2017"
FLICKR_ANNOTATIONS = (
    RESEARCH_ROOT / "data" / "eval" / "flickr30k" / "flickr_annotations_30k.csv"
)
FLICKR_IMAGES = RESEARCH_ROOT / "data" / "eval" / "flickr30k" / "flickr30k-images"

EXPECTED_PRETRAINED_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
EXPECTED_TEXT_POOL_SHA256 = "c76621edddeeb546f6fa798552b2ae11eb409c63b520080d42a8a37b66f31e9d"
EXPECTED_PAIRED_PATH_POOL_SHA256 = (
    "08c0ae51ce65ccdd8d037606fffc668b5b1d80470232b8d4e200817e446eb63e"
)

DATA_SEED = 42
POOL_SIZE = 500_000
N_LAYERS = 12
N_TOKENS = 77
D_MODEL = 768
D_FFN = 3_072
DENSE_TRANSFORMER_PARAMETERS = 85_054_464
DENSE_TOTAL_MACS_G = 6.64925184
DENSE_FFN_MACS_G = 4.359979008
NON_FFN_MACS_G = DENSE_TOTAL_MACS_G - DENSE_FFN_MACS_G


def seed_everything(seed: int) -> None:
    """Seed all random sources used by baseline calibration and evaluation."""

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


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Atomically write a UTF-8 JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, output)


def text_blocks(model: nn.Module) -> nn.ModuleList:
    return model.transformer.resblocks


def create_clip(
    device: str | torch.device,
    pretrained: str | Path = PRETRAINED,
) -> tuple[nn.Module, Any, Any]:
    """Load and validate the frozen OpenCLIP model used by every baseline."""

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(pretrained),
        force_quick_gelu=True,
    )
    model = model.to(device).eval()
    model.requires_grad_(False)
    blocks = text_blocks(model)
    if len(blocks) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} text layers, got {len(blocks)}")
    for layer_index, block in enumerate(blocks):
        if tuple(block.mlp.c_fc.weight.shape) != (D_FFN, D_MODEL):
            raise RuntimeError(f"unexpected c_fc shape at text layer {layer_index}")
        if tuple(block.mlp.c_proj.weight.shape) != (D_MODEL, D_FFN):
            raise RuntimeError(f"unexpected c_proj shape at text layer {layer_index}")
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer


def extract_main_caption(record: Any) -> str:
    """Apply the exact caption precedence used by the text main experiment."""

    if not isinstance(record, dict):
        return ""
    conversations = record.get("conversations", [])
    if isinstance(conversations, list):
        for conversation in conversations:
            if not isinstance(conversation, dict):
                continue
            if conversation.get("from") == "gpt":
                value = conversation.get("value", "")
                return value if isinstance(value, str) else ""
    fallback = record.get("caption", "")
    return fallback if isinstance(fallback, str) else ""


def prepare_token_cache(
    cache_path: Path = TOKEN_CACHE,
    annotations: Path = ANNOTATIONS,
) -> dict[str, Any]:
    """Materialize the exact 500k-example text pool shared with both stages."""

    cache_path = Path(cache_path)
    annotations = Path(annotations)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        required = {"tokens", "dataset_sha256", "data_seed", "pool_size"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise RuntimeError(f"incomplete token cache: {cache_path}")
        _validate_text_pool(payload)
        return payload

    with annotations.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("ShareGPT4V annotations must contain a JSON list")
    random.Random(DATA_SEED).shuffle(records)

    captions = []
    digest = hashlib.sha256()
    for record in records:
        caption = extract_main_caption(record)
        if not caption:
            continue
        captions.append(caption)
        digest.update(caption.encode("utf-8"))
        digest.update(b"\0")
        if len(captions) >= POOL_SIZE:
            break
    fingerprint = digest.hexdigest()
    if len(captions) != POOL_SIZE:
        raise RuntimeError(f"found only {len(captions):,} valid text samples")
    if fingerprint != EXPECTED_TEXT_POOL_SHA256:
        raise RuntimeError(
            "text pool differs from the main experiment: "
            f"{fingerprint} != {EXPECTED_TEXT_POOL_SHA256}"
        )

    import open_clip

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    chunks = [
        tokenizer(captions[start : start + 4_096]).cpu()
        for start in range(0, len(captions), 4_096)
    ]
    payload = {
        "tokens": torch.cat(chunks, dim=0),
        "dataset_sha256": fingerprint,
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "tokenizer": MODEL_NAME,
    }
    _validate_text_pool(payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    return payload


def _validate_text_pool(payload: dict[str, Any]) -> None:
    tokens = payload.get("tokens")
    if (
        payload.get("dataset_sha256") != EXPECTED_TEXT_POOL_SHA256
        or payload.get("data_seed") != DATA_SEED
        or payload.get("pool_size") != POOL_SIZE
        or not isinstance(tokens, Tensor)
        or tuple(tokens.shape) != (POOL_SIZE, N_TOKENS)
    ):
        raise RuntimeError("token cache does not match the text main experiment")


def build_main_paired_image_paths(
    annotations: Path = ANNOTATIONS,
    image_root: Path = IMAGE_ROOT,
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    """Strictly pair every item in the main 500k text pool with its image."""

    annotations = Path(annotations)
    image_root = Path(image_root)
    with annotations.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("ShareGPT4V annotations must contain a JSON list")
    random.Random(DATA_SEED).shuffle(records)

    paths = []
    text_digest = hashlib.sha256()
    path_digest = hashlib.sha256()
    for record in records:
        caption = extract_main_caption(record)
        if not caption:
            continue
        relative = record.get("image") if isinstance(record, dict) else None
        if not relative:
            raise RuntimeError("a main-experiment text sample has no paired image")
        relative = str(relative)
        path = image_root / relative
        if not path.is_file() and relative.startswith("llava/llava_pretrain/images/"):
            path = image_root / "llava" / relative.removeprefix("llava/llava_pretrain/images/")
        if not path.is_file():
            raise FileNotFoundError(
                f"unable to resolve ShareGPT4V image {relative!r} under {image_root}"
            )
        paths.append(path)
        text_digest.update(caption.encode("utf-8"))
        text_digest.update(b"\0")
        path_digest.update(relative.encode("utf-8"))
        path_digest.update(b"\0")
        if len(paths) >= POOL_SIZE:
            break

    metadata = {
        "data_seed": DATA_SEED,
        "pool_size": len(paths),
        "text_pool_sha256": text_digest.hexdigest(),
        "paired_path_pool_sha256": path_digest.hexdigest(),
    }
    if len(paths) != POOL_SIZE:
        raise RuntimeError(f"found only {len(paths):,} paired main-pool samples")
    if metadata["text_pool_sha256"] != EXPECTED_TEXT_POOL_SHA256:
        raise RuntimeError("paired captions differ from the text main experiment")
    if metadata["paired_path_pool_sha256"] != EXPECTED_PAIRED_PATH_POOL_SHA256:
        raise RuntimeError("paired paths differ from the text main-experiment records")
    return tuple(paths), metadata


def validate_image_feature_cache(payload: dict[str, Any]) -> None:
    features = payload.get("features")
    if (
        payload.get("data_seed") != DATA_SEED
        or payload.get("pool_size") != POOL_SIZE
        or payload.get("text_pool_sha256") != EXPECTED_TEXT_POOL_SHA256
        or payload.get("paired_path_pool_sha256") != EXPECTED_PAIRED_PATH_POOL_SHA256
        or payload.get("pretrained_sha256") != EXPECTED_PRETRAINED_SHA256
        or not isinstance(features, Tensor)
        or tuple(features.shape) != (POOL_SIZE, D_MODEL)
    ):
        raise RuntimeError("image-feature cache does not match the paired main pool")
    if features.dtype != torch.float16 or not torch.isfinite(features).all():
        raise RuntimeError("image-feature cache must contain finite float16 features")


def prepare_image_feature_cache(
    model: nn.Module,
    preprocess: Any,
    device: str,
    cache_path: Path = IMAGE_FEATURE_CACHE,
    annotations: Path = ANNOTATIONS,
    image_root: Path = IMAGE_ROOT,
    batch_size: int = 128,
    workers: int = 8,
) -> dict[str, Any]:
    """Encode all paired main-pool images once for paired text baselines."""

    cache_path = Path(cache_path)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid image-feature cache: {cache_path}")
        validate_image_feature_cache(payload)
        return payload
    if batch_size <= 0 or workers < 0:
        raise ValueError("image batch size must be positive and workers non-negative")

    paths, metadata = build_main_paired_image_paths(annotations, image_root)
    loader = DataLoader(
        EvaluationImages(paths, preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=workers > 0,
    )
    features = []
    with torch.inference_mode():
        for images in tqdm(loader, desc="cache all 500k paired image features"):
            encoded = model.encode_image(images.to(device, non_blocking=True))
            features.append(F.normalize(encoded, dim=-1).half().cpu())
    payload = {
        **metadata,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "model_name": MODEL_NAME,
        "features": torch.cat(features),
    }
    validate_image_feature_cache(payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    return payload


def full_pool_permutation(seed: int) -> Tensor:
    """Return all 500k indices in the main DataLoader's first-epoch order."""

    generator = torch.Generator().manual_seed(seed)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return torch.randperm(POOL_SIZE, generator=generator)


def iter_token_batches(
    tokens: Tensor,
    indices: Tensor,
    batch_size: int,
) -> Iterator[Tensor]:
    if tuple(tokens.shape) != (POOL_SIZE, N_TOKENS):
        raise ValueError("tokens must contain the complete 500k-example pool")
    if indices.numel() != POOL_SIZE:
        raise ValueError("indices must cover the complete text pool")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, POOL_SIZE, batch_size):
        selection = indices[start : start + batch_size].to(torch.int64)
        yield tokens[selection]


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
    """Load the exact 5k-image/25,014-caption COCO evaluation split."""

    with Path(annotations).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    id_to_file = {item["id"]: item["file_name"] for item in payload["images"]}
    image_ids = list(id_to_file)[:5_000]
    id_to_index = {image_id: index for index, image_id in enumerate(image_ids)}
    paths = tuple(Path(image_root) / id_to_file[image_id] for image_id in image_ids)
    captions = []
    mapping = []
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
    """Load the exact 1k-image/5k-caption Flickr30k test split."""

    paths = []
    captions = []
    mapping = []
    with Path(annotations).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "test":
                continue
            image_index = len(paths)
            paths.append(Path(image_root) / row["filename"])
            raw_captions = ast.literal_eval(row["raw"])
            for caption in raw_captions:
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
            features = model.encode_image(images.to(device, non_blocking=True))
            outputs.append(F.normalize(features, dim=-1).cpu())
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
            batch = tokenizer(captions[start : start + batch_size]).to(device)
            outputs.append(F.normalize(model.encode_text(batch), dim=-1).cpu())
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
            caption_image_indices[caption_index] in top_image[:, caption_index].tolist()
            for caption_index in range(len(caption_image_indices))
        )
        metrics[f"i2t_r{k}"] = 100.0 * image_to_text / image_features.shape[0]
        metrics[f"t2i_r{k}"] = 100.0 * text_to_image / len(caption_image_indices)
    return metrics

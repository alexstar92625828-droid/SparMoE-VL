"""Collect exact layer-wise Dense CLIP and SparMoE-VL representation metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from ...vision.encoder import SparMoEVisionEncoder
from ..vision_budget_sweep import inspect_checkpoint, load_encoder
from .protocol import (
    BATCH_SIZE,
    CAPACITY_FACTORS,
    CHECKPOINT,
    CHECKPOINT_EVERY,
    COCO_ANNOTATIONS,
    COCO_ANNOTATIONS_SHA256,
    COCO_IMAGES,
    COCO_IMAGES_TOTAL,
    COCO_MANIFEST_SHA256,
    CPU_THREADS,
    DATA_SEED,
    HISTORICAL_CHECKPOINT_SHA256,
    MODEL_DIM,
    MODEL_KEY,
    MODEL_NAME,
    NUM_LAYERS,
    NUM_WORKERS,
    OUTPUT_ROOT,
    PAPER_SCOPE,
    PATCHES_PER_IMAGE,
    PRETRAINED,
    PRETRAINED_SHA256,
    PROTOCOL,
    QUANTILES,
    ROUTING_MODE,
    RUN_SEED,
    STUDY_NAME,
    TARGET_RATIO,
    TOKENS_PER_IMAGE,
    TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
    protocol_manifest,
)


CACHE_NAMES = {
    "dense_cls": "dense_cls.f32.mmap",
    "sparse_cls": "sparse_cls.f32.mmap",
    "patch_cosine": "patch_cosine.f32.mmap",
    "progress": "progress.json",
    "analysis": "analysis.json",
    "table": "statistics.csv",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--coco-annotations", type=Path, default=COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=COCO_IMAGES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=min(CPU_THREADS, os.cpu_count() or 1),
    )
    parser.add_argument("--max-images", type=int, default=COCO_IMAGES_TOTAL)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--allow-partial-smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def load_coco_image_paths(annotation_path: Path, image_root: Path) -> tuple[Path, ...]:
    """Reproduce the result-generating lexicographic COCO file order."""

    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("images"), list):
        raise ValueError("COCO annotations must contain an images list")
    records = payload["images"]
    if not all(isinstance(item, Mapping) and "file_name" in item for item in records):
        raise ValueError("every COCO image record must contain file_name")
    names = [str(item["file_name"]) for item in records]
    if len(names) != len(set(names)):
        raise ValueError("COCO image file names must be unique")
    return tuple(image_root / name for name in sorted(names))


def validate_checkpoint(path: Path) -> dict[str, Any]:
    metadata = inspect_checkpoint(path)
    expected = {
        "target_ratio": TARGET_RATIO,
        "training_seed": RUN_SEED,
        "data_seed": DATA_SEED,
        "pool_size": TRAINING_POOL_SIZE,
        "dataset_sha256": TRAINING_POOL_SHA256,
        "capacity_factors": list(CAPACITY_FACTORS),
        "reuses_visual_main_experiment": True,
    }
    for field, wanted in expected.items():
        if metadata.get(field) != wanted:
            raise ValueError(f"checkpoint {field}={metadata.get(field)!r}; expected {wanted!r}")
    digest = file_sha256(path)
    if metadata["format"] == "historical_two_stage" and digest != HISTORICAL_CHECKPOINT_SHA256:
        raise ValueError("historical checkpoint differs from the result-generating weight")
    metadata["checkpoint_sha256"] = digest
    return metadata


def validate_args(
    args: argparse.Namespace,
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    for path, label in (
        (args.checkpoint, "visual Stage-2 checkpoint"),
        (args.pretrained, "Dense CLIP weights"),
        (args.coco_annotations, "COCO annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not args.coco_images.is_dir():
        raise FileNotFoundError(f"missing COCO image root: {args.coco_images}")
    if min(args.batch_size, args.max_images, args.checkpoint_every) <= 0:
        raise ValueError("batch size, image count, and checkpoint interval must be positive")
    if min(args.num_workers, args.cpu_threads) < 0:
        raise ValueError("worker and thread counts must be non-negative")
    if args.max_images > COCO_IMAGES_TOTAL:
        raise ValueError(f"max-images cannot exceed {COCO_IMAGES_TOTAL}")
    if not args.allow_partial_smoke:
        expected = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "max_images": COCO_IMAGES_TOTAL,
        }
        for field, wanted in expected.items():
            if getattr(args, field) != wanted:
                raise ValueError(
                    f"registered {field}={getattr(args, field)!r}; expected {wanted!r}"
                )
    if file_sha256(args.pretrained) != PRETRAINED_SHA256:
        raise ValueError("Dense CLIP checkpoint identity differs from the paper experiment")
    if file_sha256(args.coco_annotations) != COCO_ANNOTATIONS_SHA256:
        raise ValueError("COCO annotation identity differs from the paper experiment")
    metadata = validate_checkpoint(args.checkpoint)
    all_paths = load_coco_image_paths(args.coco_annotations, args.coco_images)
    if len(all_paths) != COCO_IMAGES_TOTAL:
        raise RuntimeError(f"expected {COCO_IMAGES_TOTAL} COCO images, found {len(all_paths)}")
    missing = [path for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"COCO image set is incomplete; first missing: {missing[0]}")
    manifest = sequence_sha256([path.name for path in all_paths])
    if manifest != COCO_MANIFEST_SHA256:
        raise ValueError("ordered COCO image manifest differs from the paper experiment")
    return all_paths[: args.max_images], metadata


class StrictImageDataset(Dataset[Tensor]):
    def __init__(self, paths: Sequence[Path], preprocess: Any) -> None:
        self.paths = tuple(paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                return self.preprocess(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"unable to read COCO image: {path}") from error


def visual_stem(clip_model: torch.nn.Module, images: Tensor) -> Tensor:
    """Return tokens immediately before the first visual Transformer block."""

    visual = clip_model.visual
    dtype = visual.transformer.get_cast_dtype()
    x = visual.conv1(images.to(dtype=dtype)).flatten(2).permute(0, 2, 1)
    class_token = visual.class_embedding.to(dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype
    )
    x = torch.cat([class_token, x], dim=1)
    x = x + visual.positional_embedding.to(dtype)
    x = visual.patch_dropout(x)
    x = visual.ln_pre(x)
    if not visual.transformer.batch_first:
        x = x.transpose(0, 1).contiguous()
    return x


@torch.inference_mode()
def dense_layer_states(clip_model: torch.nn.Module, images: Tensor) -> tuple[Tensor, ...]:
    """Collect post-block Dense CLIP token states on CPU in BTD layout."""

    visual = clip_model.visual
    batch_first = bool(visual.transformer.batch_first)
    x = visual_stem(clip_model, images)
    states = []
    for block in visual.transformer.resblocks:
        x = block(x, attn_mask=None)
        batch_tokens = x if batch_first else x.transpose(0, 1)
        states.append(batch_tokens.detach().float().cpu())
    if len(states) != NUM_LAYERS:
        raise RuntimeError(f"Dense CLIP returned {len(states)} layers; expected {NUM_LAYERS}")
    return tuple(states)


@torch.inference_mode()
def sparse_layer_states(model: SparMoEVisionEncoder, images: Tensor) -> tuple[Tensor, ...]:
    """Collect the exact learned-argmax Stage-2 post-block token states."""

    if model.training or model.training_stage != 2:
        raise RuntimeError("representation analysis requires an eval-mode Stage-2 encoder")
    visual = model.clip_model.visual
    batch_first = bool(visual.transformer.batch_first)
    x = visual_stem(model.clip_model, images)
    patterns = model.sparse_pattern_generator.all_layers(model.budget())
    states = []
    for layer_index, block in enumerate(visual.transformer.resblocks):
        sparse_position = model._sparse_position_by_layer.get(layer_index)
        if sparse_position is None:
            x = block(x, attn_mask=None)
        else:
            x = x + block.ls_1(block.attention(q_x=block.ln_1(x), attn_mask=None))
            normalized = block.ln_2(x)
            normalized_batch = normalized if batch_first else normalized.transpose(0, 1)
            batch_size, token_count, dimension = normalized_batch.shape
            weight_1 = block.mlp.c_fc.weight
            bias_1 = block.mlp.c_fc.bias
            weight_2 = block.mlp.c_proj.weight
            bias_2 = block.mlp.c_proj.bias
            activation = block.mlp.gelu
            class_output = (
                activation(normalized_batch[:, :1] @ weight_1.T + bias_1) @ weight_2.T + bias_2
            )
            patch = normalized_batch[:, 1:].reshape(-1, dimension)
            hidden = activation(patch @ weight_1.T + bias_1)
            logits = model.routers[sparse_position].projection(patch)
            expert_ids = logits.argmax(dim=-1)
            gates = F.one_hot(expert_ids, model.num_capacity_levels).to(patch.dtype)
            masks = patterns[sparse_position].masks
            patch_output = hidden * (gates.to(masks.dtype) @ masks).to(hidden.dtype)
            patch_output = patch_output @ weight_2.T + bias_2
            mlp_output = torch.cat(
                [
                    class_output,
                    patch_output.reshape(batch_size, token_count - 1, dimension),
                ],
                dim=1,
            )
            if not batch_first:
                mlp_output = mlp_output.transpose(0, 1)
            x = x + block.ls_2(mlp_output)
        batch_tokens = x if batch_first else x.transpose(0, 1)
        states.append(batch_tokens.detach().float().cpu())
    if len(states) != NUM_LAYERS:
        raise RuntimeError(f"SparMoE-VL returned {len(states)} layers; expected {NUM_LAYERS}")
    return tuple(states)


def patch_cosines(
    dense_states: Sequence[Tensor], sparse_states: Sequence[Tensor]
) -> np.ndarray:
    values = []
    for dense, sparse in zip(dense_states, sparse_states):
        cosine = F.cosine_similarity(dense[:, 1:, :], sparse[:, 1:, :], dim=-1)
        values.append(cosine.numpy())
    result = np.stack(values, axis=1).astype(np.float32, copy=False)
    expected = (result.shape[0], NUM_LAYERS, PATCHES_PER_IMAGE)
    if result.shape != expected:
        raise RuntimeError(f"unexpected patch cosine shape {result.shape}; expected {expected}")
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite patch-token cosine detected")
    return result


def batch_similarity_sums(
    dense_states: Sequence[Tensor], sparse_states: Sequence[Tensor]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_sums = np.zeros(NUM_LAYERS, dtype=np.float64)
    cls_sums = np.zeros(NUM_LAYERS, dtype=np.float64)
    all_counts = np.zeros(NUM_LAYERS, dtype=np.int64)
    cls_counts = np.zeros(NUM_LAYERS, dtype=np.int64)
    for layer, (dense, sparse) in enumerate(zip(dense_states, sparse_states)):
        cosine = F.cosine_similarity(dense, sparse, dim=-1)
        all_sums[layer] = cosine.double().sum().item()
        cls_sums[layer] = cosine[:, 0].double().sum().item()
        all_counts[layer] = cosine.numel()
        cls_counts[layer] = cosine.shape[0]
    return all_sums, cls_sums, all_counts, cls_counts


def centered_linear_cka(dense: np.ndarray, sparse: np.ndarray) -> float:
    """Exact centered linear CKA, matching the result-generating estimator."""

    if dense.shape != sparse.shape or dense.ndim != 2 or dense.shape[0] < 2:
        raise ValueError("CKA inputs must share a two-dimensional shape with at least 2 rows")
    x = torch.from_numpy(np.asarray(dense, dtype=np.float32).copy())
    y = torch.from_numpy(np.asarray(sparse, dtype=np.float32).copy())
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise FloatingPointError("non-finite CLS representation detected")
    x -= x.mean(dim=0, keepdim=True)
    y -= y.mean(dim=0, keepdim=True)
    cross = x.T @ y
    x_covariance = x.T @ x
    y_covariance = y.T @ y
    numerator = cross.square().sum()
    denominator = torch.linalg.matrix_norm(x_covariance, ord="fro") * torch.linalg.matrix_norm(
        y_covariance, ord="fro"
    )
    return float((numerator / denominator.clamp_min(1e-12)).clamp(0.0, 1.0).item())


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {key: output_dir / name for key, name in CACHE_NAMES.items()}


def expected_bytes(shape: Sequence[int], dtype: np.dtype[Any] = np.dtype(np.float32)) -> int:
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def cache_shapes(image_count: int) -> dict[str, tuple[int, ...]]:
    return {
        "dense_cls": (image_count, NUM_LAYERS, MODEL_DIM),
        "sparse_cls": (image_count, NUM_LAYERS, MODEL_DIM),
        "patch_cosine": (image_count, NUM_LAYERS, PATCHES_PER_IMAGE),
    }


def state_manifest(
    dataset_manifest: str,
    checkpoint_sha256: str,
    image_count: int,
) -> str:
    return sequence_sha256(
        [PROTOCOL, PRETRAINED_SHA256, dataset_manifest, checkpoint_sha256, str(image_count)]
    )


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def initial_progress(manifest: str, image_count: int) -> dict[str, Any]:
    return {
        "state_manifest": manifest,
        "image_count": image_count,
        "processed": 0,
        "all_token_cosine_sums": [0.0] * NUM_LAYERS,
        "cls_token_cosine_sums": [0.0] * NUM_LAYERS,
        "all_token_counts": [0] * NUM_LAYERS,
        "cls_token_counts": [0] * NUM_LAYERS,
    }


def validate_progress(
    progress: Mapping[str, Any], manifest: str, image_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if progress.get("state_manifest") != manifest:
        raise ValueError("existing progress belongs to a different data/model protocol")
    if progress.get("image_count") != image_count:
        raise ValueError("existing progress belongs to a different image count")
    processed = int(progress.get("processed", -1))
    if not 0 <= processed <= image_count:
        raise ValueError("progress has an invalid processed image count")
    arrays = (
        np.asarray(progress.get("all_token_cosine_sums"), dtype=np.float64),
        np.asarray(progress.get("cls_token_cosine_sums"), dtype=np.float64),
        np.asarray(progress.get("all_token_counts"), dtype=np.int64),
        np.asarray(progress.get("cls_token_counts"), dtype=np.int64),
    )
    if any(array.shape != (NUM_LAYERS,) for array in arrays):
        raise ValueError("progress contains malformed layer accumulators")
    if not all(np.isfinite(array).all() for array in arrays[:2]):
        raise ValueError("progress contains non-finite cosine sums")
    expected_all = processed * TOKENS_PER_IMAGE
    expected_cls = processed
    if not np.all(arrays[2] == expected_all) or not np.all(arrays[3] == expected_cls):
        raise ValueError("progress token counts disagree with processed images")
    return (*arrays, processed)


def open_caches(
    output_dir: Path,
    image_count: int,
    create: bool,
) -> tuple[np.memmap, np.memmap, np.memmap]:
    paths = artifact_paths(output_dir)
    shapes = cache_shapes(image_count)
    if not create:
        for key, shape in shapes.items():
            path = paths[key]
            if not path.is_file() or path.stat().st_size != expected_bytes(shape):
                raise ValueError(f"{key} cache is missing or has an unexpected size")
    mode = "w+" if create else "r+"
    return tuple(
        np.memmap(paths[key], dtype=np.float32, mode=mode, shape=shapes[key])
        for key in ("dense_cls", "sparse_cls", "patch_cosine")
    )  # type: ignore[return-value]


def save_progress(
    path: Path,
    manifest: str,
    image_count: int,
    processed: int,
    all_sums: np.ndarray,
    cls_sums: np.ndarray,
    all_counts: np.ndarray,
    cls_counts: np.ndarray,
) -> None:
    atomic_json(
        path,
        {
            "state_manifest": manifest,
            "image_count": image_count,
            "processed": processed,
            "all_token_cosine_sums": all_sums.tolist(),
            "cls_token_cosine_sums": cls_sums.tolist(),
            "all_token_counts": all_counts.tolist(),
            "cls_token_counts": cls_counts.tolist(),
        },
    )


def collect(
    model: SparMoEVisionEncoder,
    preprocess: Any,
    paths: Sequence[Path],
    args: argparse.Namespace,
    manifest: str,
) -> tuple[np.memmap, np.memmap, np.memmap, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    artifacts = artifact_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.restart:
        for path in artifacts.values():
            path.unlink(missing_ok=True)

    if artifacts["progress"].is_file():
        progress = json.loads(artifacts["progress"].read_text(encoding="utf-8"))
        all_sums, cls_sums, all_counts, cls_counts, processed = validate_progress(
            progress, manifest, len(paths)
        )
        create = False
    else:
        progress = initial_progress(manifest, len(paths))
        all_sums, cls_sums, all_counts, cls_counts, processed = validate_progress(
            progress, manifest, len(paths)
        )
        create = True
    dense_map, sparse_map, patch_map = open_caches(args.output_dir, len(paths), create)

    dataset = StrictImageDataset(paths, preprocess)
    loader = DataLoader(
        Subset(dataset, range(processed, len(dataset))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    for batch_index, images in enumerate(
        tqdm(loader, desc="COCO layer representations", unit="batch"), start=1
    ):
        images = images.to(args.device, non_blocking=str(args.device).startswith("cuda"))
        dense_states = dense_layer_states(model.clip_model, images)
        sparse_states = sparse_layer_states(model, images)
        batch_count = int(images.shape[0])
        stop = processed + batch_count
        dense_map[processed:stop] = np.stack(
            [state[:, 0, :].numpy() for state in dense_states], axis=1
        )
        sparse_map[processed:stop] = np.stack(
            [state[:, 0, :].numpy() for state in sparse_states], axis=1
        )
        patch_map[processed:stop] = patch_cosines(dense_states, sparse_states)
        batch_values = batch_similarity_sums(dense_states, sparse_states)
        all_sums += batch_values[0]
        cls_sums += batch_values[1]
        all_counts += batch_values[2]
        cls_counts += batch_values[3]
        processed = stop
        if batch_index % args.checkpoint_every == 0 or processed == len(paths):
            dense_map.flush()
            sparse_map.flush()
            patch_map.flush()
            save_progress(
                artifacts["progress"],
                manifest,
                len(paths),
                processed,
                all_sums,
                cls_sums,
                all_counts,
                cls_counts,
            )
    if processed != len(paths):
        raise RuntimeError(f"processed {processed} images; expected {len(paths)}")
    return dense_map, sparse_map, patch_map, all_sums, cls_sums, all_counts, cls_counts


def summarize_patch_distribution(patch_map: np.memmap) -> dict[str, list[float]]:
    statistics = {key: [] for key in ("mean", "std", "q10", "q25", "median", "q75", "q90")}
    for layer in tqdm(range(NUM_LAYERS), desc="Exact patch distribution", unit="layer"):
        values = np.asarray(patch_map[:, layer, :]).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise FloatingPointError(f"invalid patch-token values at layer {layer + 1}")
        quantiles = np.quantile(values, QUANTILES)
        statistics["mean"].append(float(np.mean(values, dtype=np.float64)))
        statistics["std"].append(float(np.std(values, dtype=np.float64)))
        for key, value in zip(("q10", "q25", "median", "q75", "q90"), quantiles):
            statistics[key].append(float(value))
    return statistics


def build_payload(
    paths: Sequence[Path],
    metadata: Mapping[str, Any],
    dense_map: np.memmap,
    sparse_map: np.memmap,
    patch_map: np.memmap,
    all_sums: np.ndarray,
    cls_sums: np.ndarray,
    all_counts: np.ndarray,
    cls_counts: np.ndarray,
    partial_smoke: bool,
) -> dict[str, Any]:
    cka = [
        centered_linear_cka(dense_map[:, layer, :], sparse_map[:, layer, :])
        for layer in tqdm(range(NUM_LAYERS), desc="Exact CLS CKA", unit="layer")
    ]
    distribution = summarize_patch_distribution(patch_map)
    shapes = cache_shapes(len(paths))
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL if not partial_smoke else f"{PROTOCOL}_smoke",
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "checkpoint_metadata": dict(metadata),
        "routing": ROUTING_MODE,
        "representation_point": "post_transformer_block",
        "dataset": "COCO-val2017",
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "dataset_manifest_sha256": sequence_sha256([path.name for path in paths]),
        "image_order": "ascending_file_name",
        "image_count": len(paths),
        "num_layers": NUM_LAYERS,
        "tokens_per_image": TOKENS_PER_IMAGE,
        "patches_per_image": PATCHES_PER_IMAGE,
        "all_tokens_per_layer": len(paths) * TOKENS_PER_IMAGE,
        "patch_tokens_per_layer": len(paths) * PATCHES_PER_IMAGE,
        "cka_estimator": "exact_centered_linear_cka_over_all_cls_states",
        "layers_one_based": list(range(1, NUM_LAYERS + 1)),
        "cls_linear_cka": cka,
        "all_token_cosine": np.clip(all_sums / all_counts, -1.0, 1.0).tolist(),
        "cls_token_cosine": np.clip(cls_sums / cls_counts, -1.0, 1.0).tolist(),
        "patch_token_cosine": distribution,
        "cache": {
            key: {
                "file": CACHE_NAMES[key],
                "dtype": "float32",
                "shape": list(shapes[key]),
            }
            for key in ("dense_cls", "sparse_cls", "patch_cosine")
        },
        "tf32": False,
    }


def validate_analysis(
    payload: Mapping[str, Any],
    source: str | Path,
    *,
    allow_partial_smoke: bool = False,
) -> None:
    if allow_partial_smoke:
        image_count = int(payload.get("image_count", 0))
        if not 2 <= image_count <= COCO_IMAGES_TOTAL:
            raise ValueError(f"{source}: smoke analysis requires 2..{COCO_IMAGES_TOTAL} images")
        dataset_manifest = payload.get("dataset_manifest_sha256")
        if not isinstance(dataset_manifest, str) or len(dataset_manifest) != 64:
            raise ValueError(f"{source}: smoke analysis has no valid dataset manifest")
        expected_protocol = f"{PROTOCOL}_smoke"
    else:
        image_count = COCO_IMAGES_TOTAL
        dataset_manifest = COCO_MANIFEST_SHA256
        expected_protocol = PROTOCOL
    expected = {
        "study": STUDY_NAME,
        "protocol": expected_protocol,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "routing": ROUTING_MODE,
        "representation_point": "post_transformer_block",
        "dataset": "COCO-val2017",
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "dataset_manifest_sha256": dataset_manifest,
        "image_order": "ascending_file_name",
        "image_count": image_count,
        "num_layers": NUM_LAYERS,
        "tokens_per_image": TOKENS_PER_IMAGE,
        "patches_per_image": PATCHES_PER_IMAGE,
        "all_tokens_per_layer": image_count * TOKENS_PER_IMAGE,
        "patch_tokens_per_layer": image_count * PATCHES_PER_IMAGE,
        "cka_estimator": "exact_centered_linear_cka_over_all_cls_states",
        "layers_one_based": list(range(1, NUM_LAYERS + 1)),
        "tf32": False,
    }
    for field, wanted in expected.items():
        if payload.get(field) != wanted:
            raise ValueError(f"{source}: {field}={payload.get(field)!r}; expected {wanted!r}")
    metadata = payload.get("checkpoint_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}: missing checkpoint metadata")
    checkpoint_expected = {
        "target_ratio": TARGET_RATIO,
        "training_seed": RUN_SEED,
        "data_seed": DATA_SEED,
        "pool_size": TRAINING_POOL_SIZE,
        "dataset_sha256": TRAINING_POOL_SHA256,
        "capacity_factors": list(CAPACITY_FACTORS),
        "reuses_visual_main_experiment": True,
    }
    for field, wanted in checkpoint_expected.items():
        if metadata.get(field) != wanted:
            raise ValueError(f"{source}: checkpoint metadata disagrees on {field}")
    for field in ("cls_linear_cka", "all_token_cosine", "cls_token_cosine"):
        values = np.asarray(payload.get(field), dtype=np.float64)
        if values.shape != (NUM_LAYERS,) or not np.isfinite(values).all():
            raise ValueError(f"{source}: invalid {field}")
        lower = 0.0 if field == "cls_linear_cka" else -1.0
        if np.any(values < lower) or np.any(values > 1.0):
            raise ValueError(f"{source}: {field} lies outside its valid range")
    distribution = payload.get("patch_token_cosine")
    if not isinstance(distribution, Mapping):
        raise ValueError(f"{source}: missing patch-token distribution")
    arrays = {}
    for field in ("mean", "std", "q10", "q25", "median", "q75", "q90"):
        values = np.asarray(distribution.get(field), dtype=np.float64)
        if values.shape != (NUM_LAYERS,) or not np.isfinite(values).all():
            raise ValueError(f"{source}: invalid patch-token {field}")
        arrays[field] = values
    if np.any(arrays["std"] < 0):
        raise ValueError(f"{source}: patch-token standard deviation cannot be negative")
    ordered = np.stack([arrays[key] for key in ("q10", "q25", "median", "q75", "q90")])
    if np.any(np.diff(ordered, axis=0) < 0) or np.any(ordered < -1) or np.any(ordered > 1):
        raise ValueError(f"{source}: patch-token quantiles are invalid")
    cache = payload.get("cache")
    expected_shapes = cache_shapes(image_count)
    if not isinstance(cache, Mapping):
        raise ValueError(f"{source}: missing cache contract")
    for key, shape in expected_shapes.items():
        spec = cache.get(key)
        if not isinstance(spec, Mapping):
            raise ValueError(f"{source}: missing {key} cache contract")
        if spec.get("file") != CACHE_NAMES[key] or spec.get("dtype") != "float32":
            raise ValueError(f"{source}: invalid {key} cache identity")
        if spec.get("shape") != list(shape):
            raise ValueError(f"{source}: invalid {key} cache shape")


def write_table(path: Path, payload: Mapping[str, Any]) -> None:
    distribution = payload["patch_token_cosine"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "layer",
                "cls_linear_cka",
                "cls_token_cosine",
                "all_token_cosine",
                "patch_mean",
                "patch_std",
                "patch_q10",
                "patch_q25",
                "patch_median",
                "patch_q75",
                "patch_q90",
            ]
        )
        for index, layer in enumerate(payload["layers_one_based"]):
            writer.writerow(
                [
                    layer,
                    *(
                        f"{float(values[index]):.8f}"
                        for values in (
                            payload["cls_linear_cka"],
                            payload["cls_token_cosine"],
                            payload["all_token_cosine"],
                            distribution["mean"],
                            distribution["std"],
                            distribution["q10"],
                            distribution["q25"],
                            distribution["median"],
                            distribution["q75"],
                            distribution["q90"],
                        )
                    ),
                ]
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths, metadata = validate_args(args)
    dataset_manifest = sequence_sha256([path.name for path in paths])
    check = {
        **protocol_manifest(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "checkpoint_format": metadata["format"],
        "pretrained": str(args.pretrained.resolve()),
        "pretrained_sha256": PRETRAINED_SHA256,
        "coco_annotations": str(args.coco_annotations.resolve()),
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "coco_images": str(args.coco_images.resolve()),
        "dataset_manifest_sha256": dataset_manifest,
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.check_only:
        print(json.dumps(check, indent=2))
        return check
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cpu" and args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    set_reproducible_seed(RUN_SEED)
    model, loaded_metadata, preprocess, _ = load_encoder(
        args.checkpoint, args.pretrained, device
    )
    if loaded_metadata["checkpoint_step"] != metadata["checkpoint_step"]:
        raise RuntimeError("checkpoint changed between validation and model loading")
    manifest = state_manifest(dataset_manifest, metadata["checkpoint_sha256"], len(paths))
    values = collect(model, preprocess, paths, args, manifest)
    payload = build_payload(paths, metadata, *values, args.allow_partial_smoke)
    artifacts = artifact_paths(args.output_dir)
    validate_analysis(
        payload,
        artifacts["analysis"],
        allow_partial_smoke=args.allow_partial_smoke,
    )
    atomic_json(artifacts["analysis"], payload)
    write_table(artifacts["table"], payload)
    print(f"analysis={artifacts['analysis']}")
    print(f"statistics={artifacts['table']}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

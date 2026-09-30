"""Measure Dense-to-sparse cross-modal similarity structure preservation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from ..layerwise_representation_consistency.analysis import (
    expected_bytes,
    validate_analysis as validate_layerwise_analysis,
)
from .protocol import (
    CAPACITY_FACTORS,
    CAPTION_COUNT,
    COCO_ANNOTATIONS,
    COCO_ANNOTATIONS_SHA256,
    COCO_CAPTION_ANNOTATION_ID_SHA256,
    COCO_CAPTION_IMAGE_INDEX_SHA256,
    COCO_CAPTION_ORDER_SHA256,
    COCO_FIRST_CAPTION_INDEX_SHA256,
    COCO_IMAGE_ORDER_SHA256,
    COCO_IMAGES,
    DATA_SEED,
    IMAGE_COUNT,
    LAYER_COUNT,
    LAYERWISE_OUTPUT,
    LAYERWISE_PROTOCOL,
    LAYERWISE_STUDY,
    MATRIX_BLOCK_SIZE,
    MODEL_DIM,
    MODEL_KEY,
    MODEL_NAME,
    OUTPUT_DIM,
    OUTPUT_ROOT,
    PAPER_SCOPE,
    PRETRAINED,
    PRETRAINED_SHA256,
    PROJECTION_BATCH_SIZE,
    PROTOCOL,
    RUN_SEED,
    SAMPLE_COUNT,
    SAMPLE_SEED,
    STUDY_NAME,
    TARGET_RATIO,
    TEXT_BATCH_SIZE,
    TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
    UNORDERED_SAMPLE_INDEX_SHA256,
    protocol_manifest,
)


FEATURE_FILE = "features.pt"
ANALYSIS_FILE = "analysis.json"
SAMPLE_FILE = "visualization.npz"
SAMPLE_MANIFEST_FILE = "visualization_manifest.json"
PLOT_FILES = (
    f"{STUDY_NAME}.pdf",
    f"{STUDY_NAME}.png",
)


@dataclass(frozen=True)
class PaperCOCO:
    image_paths: tuple[Path, ...]
    image_ids: tuple[int, ...]
    file_names: tuple[str, ...]
    captions: tuple[str, ...]
    caption_image_indices: np.ndarray
    caption_annotation_ids: tuple[int, ...]
    first_caption_indices: np.ndarray


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layerwise-analysis",
        type=Path,
        default=LAYERWISE_OUTPUT / "analysis.json",
    )
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--coco-annotations", type=Path, default=COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=COCO_IMAGES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-batch-size", type=int, default=TEXT_BATCH_SIZE)
    parser.add_argument(
        "--projection-batch-size",
        type=int,
        default=PROJECTION_BATCH_SIZE,
    )
    parser.add_argument("--matrix-block-size", type=int, default=MATRIX_BLOCK_SIZE)
    parser.add_argument("--max-images", type=int, default=IMAGE_COUNT)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
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


def sequence_sha256(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_paper_coco(annotation_path: Path, image_root: Path) -> PaperCOCO:
    """Recover the exact sorted-image/raw-caption ordering of the paper code."""

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("COCO annotations must contain a JSON object")
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO annotations must expose images and annotations lists")
    if len(images) != IMAGE_COUNT:
        raise ValueError(f"COCO contains {len(images)} images; expected {IMAGE_COUNT}")
    ordered = sorted(images, key=lambda item: str(item["file_name"]))
    image_ids = []
    file_names = []
    id_to_index: dict[int, int] = {}
    for index, item in enumerate(ordered):
        if not isinstance(item, Mapping) or "id" not in item or "file_name" not in item:
            raise ValueError(f"COCO image record {index} is malformed")
        image_id = int(item["id"])
        if image_id in id_to_index:
            raise ValueError(f"duplicate COCO image id: {image_id}")
        image_ids.append(image_id)
        file_names.append(str(item["file_name"]))
        id_to_index[image_id] = index

    captions = []
    caption_image_indices = []
    caption_annotation_ids = []
    first_caption_indices = np.full(IMAGE_COUNT, -1, dtype=np.int64)
    for annotation_index, item in enumerate(annotations):
        if not isinstance(item, Mapping):
            raise ValueError(f"COCO caption record {annotation_index} is malformed")
        image_id = int(item.get("image_id", -1))
        caption = item.get("caption")
        if image_id not in id_to_index or not isinstance(caption, str) or "id" not in item:
            raise ValueError(f"COCO caption record {annotation_index} is incomplete")
        image_index = id_to_index[image_id]
        caption_index = len(captions)
        captions.append(caption.strip())
        caption_image_indices.append(image_index)
        caption_annotation_ids.append(int(item["id"]))
        if first_caption_indices[image_index] < 0:
            first_caption_indices[image_index] = caption_index
    if len(captions) != CAPTION_COUNT:
        raise ValueError(f"COCO contains {len(captions)} captions; expected {CAPTION_COUNT}")
    if np.any(first_caption_indices < 0):
        raise ValueError("at least one COCO image has no caption")

    identities = {
        "image order": (sequence_sha256(file_names), COCO_IMAGE_ORDER_SHA256),
        "caption order": (sequence_sha256(captions), COCO_CAPTION_ORDER_SHA256),
        "caption-image mapping": (
            sequence_sha256(caption_image_indices),
            COCO_CAPTION_IMAGE_INDEX_SHA256,
        ),
        "caption annotation IDs": (
            sequence_sha256(caption_annotation_ids),
            COCO_CAPTION_ANNOTATION_ID_SHA256,
        ),
        "first-caption mapping": (
            sequence_sha256(first_caption_indices.tolist()),
            COCO_FIRST_CAPTION_INDEX_SHA256,
        ),
    }
    for label, (observed, expected) in identities.items():
        if observed != expected:
            raise ValueError(f"COCO {label} differs from the result-generating protocol")
    image_paths = tuple(image_root / name for name in file_names)
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing registered COCO image: {missing[0]}")
    return PaperCOCO(
        image_paths=image_paths,
        image_ids=tuple(image_ids),
        file_names=tuple(file_names),
        captions=tuple(captions),
        caption_image_indices=np.asarray(caption_image_indices, dtype=np.int64),
        caption_annotation_ids=tuple(caption_annotation_ids),
        first_caption_indices=first_caption_indices,
    )


def subset_coco(corpus: PaperCOCO, image_count: int) -> PaperCOCO:
    """Keep the first sorted images and their captions for explicit smoke runs."""

    if not 1 <= image_count <= len(corpus.image_paths):
        raise ValueError("image subset size lies outside the COCO corpus")
    if image_count == len(corpus.image_paths):
        return corpus
    keep = corpus.caption_image_indices < image_count
    old_caption_indices = np.flatnonzero(keep)
    captions = tuple(corpus.captions[index] for index in old_caption_indices)
    annotation_ids = tuple(
        corpus.caption_annotation_ids[index] for index in old_caption_indices
    )
    caption_image_indices = corpus.caption_image_indices[keep].copy()
    first_caption_indices = np.full(image_count, -1, dtype=np.int64)
    for caption_index, image_index in enumerate(caption_image_indices):
        if first_caption_indices[image_index] < 0:
            first_caption_indices[image_index] = caption_index
    if np.any(first_caption_indices < 0):
        raise ValueError("smoke image subset contains an image without captions")
    return PaperCOCO(
        image_paths=corpus.image_paths[:image_count],
        image_ids=corpus.image_ids[:image_count],
        file_names=corpus.file_names[:image_count],
        captions=captions,
        caption_image_indices=caption_image_indices,
        caption_annotation_ids=annotation_ids,
        first_caption_indices=first_caption_indices,
    )


def load_layerwise_cls_caches(
    analysis_path: Path,
    *,
    allow_partial_smoke: bool,
) -> tuple[dict[str, Any], np.memmap, np.memmap, str]:
    if not analysis_path.is_file():
        raise FileNotFoundError(f"missing layer-wise analysis: {analysis_path}")
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("layer-wise analysis must contain a JSON object")
    validate_layerwise_analysis(
        payload,
        analysis_path,
        allow_partial_smoke=allow_partial_smoke,
    )
    expected_protocol = (
        f"{LAYERWISE_PROTOCOL}_smoke" if allow_partial_smoke else LAYERWISE_PROTOCOL
    )
    if payload.get("study") != LAYERWISE_STUDY or payload.get("protocol") != expected_protocol:
        raise ValueError("image CLS cache comes from a different layer-wise study")
    image_count = int(payload["image_count"])
    maps = []
    for key in ("dense_cls", "sparse_cls"):
        spec = payload["cache"][key]
        shape = tuple(int(value) for value in spec["shape"])
        if shape != (image_count, LAYER_COUNT, MODEL_DIM):
            raise ValueError(f"{key} cache has invalid geometry")
        path = analysis_path.parent / str(spec["file"])
        if not path.is_file() or path.stat().st_size != expected_bytes(shape):
            raise ValueError(f"{key} cache is missing or has an unexpected size")
        maps.append(np.memmap(path, dtype=np.float32, mode="r", shape=shape))
    return payload, maps[0], maps[1], file_sha256(analysis_path)


def validate_args(
    args: argparse.Namespace,
) -> tuple[PaperCOCO, dict[str, Any], np.memmap, np.memmap, str]:
    for path, label in (
        (args.pretrained, "Dense CLIP ViT-L/14 weights"),
        (args.coco_annotations, "COCO val2017 annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not args.coco_images.is_dir():
        raise FileNotFoundError(f"missing COCO image root: {args.coco_images}")
    if (
        min(
            args.text_batch_size,
            args.projection_batch_size,
            args.matrix_block_size,
            args.max_images,
            args.sample_count,
        )
        <= 0
    ):
        raise ValueError("batch, block, image, and sample counts must be positive")
    if args.sample_seed != SAMPLE_SEED:
        raise ValueError(f"sample seed must remain {SAMPLE_SEED}")
    if args.sample_count > args.max_images:
        raise ValueError("visualization sample count cannot exceed the image count")
    if not args.allow_partial_smoke:
        expected = {
            "text_batch_size": TEXT_BATCH_SIZE,
            "projection_batch_size": PROJECTION_BATCH_SIZE,
            "matrix_block_size": MATRIX_BLOCK_SIZE,
            "max_images": IMAGE_COUNT,
            "sample_count": SAMPLE_COUNT,
        }
        for field, wanted in expected.items():
            if getattr(args, field) != wanted:
                raise ValueError(
                    f"registered {field}={getattr(args, field)!r}; expected {wanted!r}"
                )
    elif args.max_images < 2 or args.sample_count < 2:
        raise ValueError("partial smoke analysis requires at least two images and samples")
    if file_sha256(args.pretrained) != PRETRAINED_SHA256:
        raise ValueError("Dense CLIP checkpoint identity differs from the paper experiment")
    if file_sha256(args.coco_annotations) != COCO_ANNOTATIONS_SHA256:
        raise ValueError("COCO annotation identity differs from the paper experiment")
    full_corpus = load_paper_coco(args.coco_annotations, args.coco_images)
    corpus = subset_coco(full_corpus, args.max_images)
    layerwise, dense_map, sparse_map, analysis_sha = load_layerwise_cls_caches(
        args.layerwise_analysis,
        allow_partial_smoke=args.allow_partial_smoke,
    )
    if int(layerwise["image_count"]) != len(corpus.image_paths):
        raise ValueError("layer-wise CLS cache and requested COCO image count differ")
    return corpus, layerwise, dense_map, sparse_map, analysis_sha


def load_dense_clip(pretrained: Path, device: torch.device) -> tuple[Any, Any]:
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("install open_clip_torch before running this analysis") from error
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(pretrained),
        force_quick_gelu=True,
    )
    return model.to(device).eval(), open_clip.get_tokenizer(MODEL_NAME)


@torch.inference_mode()
def project_final_cls(
    model: Any,
    states: np.memmap,
    device: torch.device,
    batch_size: int,
    description: str,
) -> Tensor:
    features = []
    visual = model.visual
    for start in tqdm(range(0, states.shape[0], batch_size), desc=description):
        stop = min(start + batch_size, states.shape[0])
        cls = torch.from_numpy(np.array(states[start:stop, -1, :], copy=True)).to(device)
        cls = visual.ln_post(cls)
        if visual.proj is not None:
            cls = cls @ visual.proj
        features.append(F.normalize(cls.float(), dim=-1).cpu())
    return torch.cat(features, dim=0)


@torch.inference_mode()
def encode_dense_texts(
    model: Any,
    tokenizer: Any,
    captions: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Tensor:
    features = []
    for start in tqdm(range(0, len(captions), batch_size), desc="Dense COCO texts"):
        token_ids = tokenizer(list(captions[start : start + batch_size])).to(device)
        encoded = model.encode_text(token_ids)
        features.append(F.normalize(encoded.float(), dim=-1).cpu())
    return torch.cat(features, dim=0)


def corpus_identity(corpus: PaperCOCO) -> dict[str, Any]:
    return {
        "image_order_sha256": sequence_sha256(corpus.file_names),
        "caption_order_sha256": sequence_sha256(corpus.captions),
        "caption_image_index_sha256": sequence_sha256(corpus.caption_image_indices.tolist()),
        "caption_annotation_id_sha256": sequence_sha256(corpus.caption_annotation_ids),
        "first_caption_index_sha256": sequence_sha256(corpus.first_caption_indices.tolist()),
    }


def build_features(
    model: Any,
    tokenizer: Any,
    corpus: PaperCOCO,
    layerwise: Mapping[str, Any],
    dense_map: np.memmap,
    sparse_map: np.memmap,
    layerwise_analysis_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dense_images = project_final_cls(
        model,
        dense_map,
        torch.device(args.device),
        args.projection_batch_size,
        "Project Dense image CLS",
    )
    sparse_images = project_final_cls(
        model,
        sparse_map,
        torch.device(args.device),
        args.projection_batch_size,
        "Project SparMoE image CLS",
    )
    text_features = encode_dense_texts(
        model,
        tokenizer,
        corpus.captions,
        torch.device(args.device),
        args.text_batch_size,
    )
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL if not args.allow_partial_smoke else f"{PROTOCOL}_smoke",
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "image_count": len(corpus.image_paths),
        "caption_count": len(corpus.captions),
        "feature_dimension": OUTPUT_DIM,
        "text_encoder": "frozen_dense_clip",
        "pretrained_sha256": PRETRAINED_SHA256,
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        **corpus_identity(corpus),
        "layerwise_source": {
            "study": layerwise["study"],
            "protocol": layerwise["protocol"],
            "analysis_sha256": layerwise_analysis_sha256,
            "checkpoint_metadata": layerwise["checkpoint_metadata"],
        },
        "dense_image_features": dense_images,
        "sparse_image_features": sparse_images,
        "text_features": text_features,
        "caption_image_indices": torch.from_numpy(corpus.caption_image_indices.copy()),
        "first_caption_indices": torch.from_numpy(corpus.first_caption_indices.copy()),
        "image_ids": list(corpus.image_ids),
        "image_files": list(corpus.file_names),
    }


def validate_features(
    features: Mapping[str, Any],
    corpus: PaperCOCO,
    layerwise_analysis_sha256: str,
    *,
    allow_partial_smoke: bool,
) -> None:
    expected_protocol = PROTOCOL if not allow_partial_smoke else f"{PROTOCOL}_smoke"
    expected = {
        "study": STUDY_NAME,
        "protocol": expected_protocol,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "image_count": len(corpus.image_paths),
        "caption_count": len(corpus.captions),
        "feature_dimension": OUTPUT_DIM,
        "text_encoder": "frozen_dense_clip",
        "pretrained_sha256": PRETRAINED_SHA256,
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        **corpus_identity(corpus),
    }
    for field, wanted in expected.items():
        if features.get(field) != wanted:
            raise ValueError(f"feature cache disagrees on {field}")
    source = features.get("layerwise_source")
    if not isinstance(source, Mapping):
        raise ValueError("feature cache has no layer-wise source identity")
    if source.get("analysis_sha256") != layerwise_analysis_sha256:
        raise ValueError("feature cache belongs to a different layer-wise analysis")
    expected_shapes = {
        "dense_image_features": (len(corpus.image_paths), OUTPUT_DIM),
        "sparse_image_features": (len(corpus.image_paths), OUTPUT_DIM),
        "text_features": (len(corpus.captions), OUTPUT_DIM),
        "caption_image_indices": (len(corpus.captions),),
        "first_caption_indices": (len(corpus.image_paths),),
    }
    for key, shape in expected_shapes.items():
        tensor = features.get(key)
        if not isinstance(tensor, Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"feature cache {key} has an invalid shape")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"feature cache {key} contains non-finite values")
    for key in ("dense_image_features", "sparse_image_features", "text_features"):
        norms = torch.linalg.vector_norm(features[key].float(), dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=0.0):
            raise ValueError(f"feature cache {key} is not unit-normalized")
    if not torch.equal(
        features["caption_image_indices"].long(),
        torch.from_numpy(corpus.caption_image_indices),
    ):
        raise ValueError("feature cache caption-image mapping changed")
    if not torch.equal(
        features["first_caption_indices"].long(),
        torch.from_numpy(corpus.first_caption_indices),
    ):
        raise ValueError("feature cache first-caption mapping changed")
    if features.get("image_ids") != list(corpus.image_ids):
        raise ValueError("feature cache image IDs changed")
    if features.get("image_files") != list(corpus.file_names):
        raise ValueError("feature cache image filenames changed")


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_or_build_features(
    corpus: PaperCOCO,
    layerwise: Mapping[str, Any],
    dense_map: np.memmap,
    sparse_map: np.memmap,
    layerwise_analysis_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = args.output_dir / FEATURE_FILE
    if path.is_file():
        features = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(features, dict):
            raise ValueError("feature cache must contain a mapping")
    else:
        device = torch.device(args.device)
        model, tokenizer = load_dense_clip(args.pretrained, device)
        features = build_features(
            model,
            tokenizer,
            corpus,
            layerwise,
            dense_map,
            sparse_map,
            layerwise_analysis_sha256,
            args,
        )
        atomic_torch_save(path, features)
    validate_features(
        features,
        corpus,
        layerwise_analysis_sha256,
        allow_partial_smoke=args.allow_partial_smoke,
    )
    return features


@torch.inference_mode()
def full_matrix_statistics(
    dense_images: Tensor,
    sparse_images: Tensor,
    text_features: Tensor,
    *,
    device: torch.device,
    block_size: int,
) -> dict[str, float | int]:
    if dense_images.shape != sparse_images.shape or dense_images.ndim != 2:
        raise ValueError("Dense and sparse image features must share a matrix shape")
    if text_features.ndim != 2 or text_features.shape[1] != dense_images.shape[1]:
        raise ValueError("text and image feature dimensions must match")
    dense_images = dense_images.to(device)
    sparse_images = sparse_images.to(device)
    text_features = text_features.to(device)
    sums = {key: 0.0 for key in ("dense", "sparse", "dd", "ss", "ds", "abs", "sq")}
    count = 0
    for start in tqdm(
        range(0, dense_images.shape[0], block_size),
        desc="Full cross-modal matrix statistics",
    ):
        stop = min(start + block_size, dense_images.shape[0])
        dense = (dense_images[start:stop] @ text_features.T).double()
        sparse = (sparse_images[start:stop] @ text_features.T).double()
        difference = sparse - dense
        sums["dense"] += float(dense.sum())
        sums["sparse"] += float(sparse.sum())
        sums["dd"] += float(dense.square().sum())
        sums["ss"] += float(sparse.square().sum())
        sums["ds"] += float((dense * sparse).sum())
        sums["abs"] += float(difference.abs().sum())
        sums["sq"] += float(difference.square().sum())
        count += dense.numel()
    if count == 0:
        raise ValueError("cross-modal matrix contains no similarities")
    size = float(count)
    covariance = sums["ds"] - sums["dense"] * sums["sparse"] / size
    dense_variance = sums["dd"] - sums["dense"] ** 2 / size
    sparse_variance = sums["ss"] - sums["sparse"] ** 2 / size
    denominator = np.sqrt(max(dense_variance * sparse_variance, 0.0))
    if denominator <= 0 or sums["dd"] <= 0 or sums["ss"] <= 0:
        raise ValueError("cross-modal matrix has degenerate similarity variance")
    return {
        "pairwise_similarity_count": count,
        "pearson": float(covariance / denominator),
        "matrix_cosine": float(sums["ds"] / np.sqrt(sums["dd"] * sums["ss"])),
        "mae": float(sums["abs"] / size),
        "rmse": float(np.sqrt(sums["sq"] / size)),
    }


def fixed_unordered_sample(image_count: int, sample_count: int) -> np.ndarray:
    selected = np.random.default_rng(SAMPLE_SEED).choice(
        image_count,
        size=sample_count,
        replace=False,
    )
    selected = np.asarray(selected, dtype=np.int64)
    if image_count == IMAGE_COUNT and sample_count == SAMPLE_COUNT:
        if sequence_sha256(selected.tolist()) != UNORDERED_SAMPLE_INDEX_SHA256:
            raise RuntimeError("NumPy sampling differs from the result-generating selection")
    return selected


def build_visualization(
    features: Mapping[str, Any],
    corpus: PaperCOCO,
    sample_count: int,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    selected_unordered = fixed_unordered_sample(len(corpus.image_paths), sample_count)
    first_captions = corpus.first_caption_indices[selected_unordered]
    dense_images = features["dense_image_features"][selected_unordered]
    texts = features["text_features"][first_captions]
    joint = F.normalize(dense_images + texts, dim=-1).numpy()
    distance = np.clip(1.0 - joint @ joint.T, 0.0, 2.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    tree = linkage(
        squareform(distance, checks=False),
        method="average",
        optimal_ordering=True,
    )
    order = leaves_list(tree)
    selected = selected_unordered[order]
    first_captions = corpus.first_caption_indices[selected]
    dense_images = features["dense_image_features"][selected]
    sparse_images = features["sparse_image_features"][selected]
    texts = features["text_features"][first_captions]
    dense_matrix = (dense_images @ texts.T).numpy().astype(np.float32)
    sparse_matrix = (sparse_images @ texts.T).numpy().astype(np.float32)
    sample = {
        "dense": dense_matrix,
        "sparse": sparse_matrix,
        "image_indices": selected,
        "caption_indices": first_captions,
    }
    manifest = {
        "study": STUDY_NAME,
        "protocol": features["protocol"],
        "dataset": "COCO-val2017",
        "selection": "uniform_without_replacement_independent_of_model_scores",
        "sample_seed": SAMPLE_SEED,
        "sample_count": sample_count,
        "unordered_selection_sha256": sequence_sha256(selected_unordered.tolist()),
        "display_order": "average_linkage_dense_joint_image_text_embedding",
        "ordered_image_index_sha256": sequence_sha256(selected.tolist()),
        "ordered_caption_index_sha256": sequence_sha256(first_captions.tolist()),
        "entries": [
            {
                "image_index": int(image_index),
                "image_id": corpus.image_ids[int(image_index)],
                "file_name": corpus.file_names[int(image_index)],
                "caption_index": int(caption_index),
                "caption_annotation_id": corpus.caption_annotation_ids[int(caption_index)],
                "caption": corpus.captions[int(caption_index)],
            }
            for image_index, caption_index in zip(selected, first_captions)
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / SAMPLE_FILE
    temporary = sample_path.with_suffix(f"{sample_path.suffix}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **sample)
    temporary.replace(sample_path)
    atomic_json(output_dir / SAMPLE_MANIFEST_FILE, manifest)
    return sample, manifest


def validate_visualization(
    sample: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    corpus: PaperCOCO,
    sample_count: int,
    *,
    allow_partial_smoke: bool,
) -> None:
    expected_protocol = PROTOCOL if not allow_partial_smoke else f"{PROTOCOL}_smoke"
    expected = {
        "study": STUDY_NAME,
        "protocol": expected_protocol,
        "dataset": "COCO-val2017",
        "selection": "uniform_without_replacement_independent_of_model_scores",
        "sample_seed": SAMPLE_SEED,
        "sample_count": sample_count,
        "display_order": "average_linkage_dense_joint_image_text_embedding",
    }
    for field, wanted in expected.items():
        if manifest.get(field) != wanted:
            raise ValueError(f"visualization manifest disagrees on {field}")
    for key in ("dense", "sparse"):
        values = np.asarray(sample.get(key))
        if values.shape != (sample_count, sample_count) or not np.isfinite(values).all():
            raise ValueError(f"visualization {key} matrix is invalid")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError(f"visualization {key} cosine lies outside [-1, 1]")
    image_indices = np.asarray(sample.get("image_indices"), dtype=np.int64)
    caption_indices = np.asarray(sample.get("caption_indices"), dtype=np.int64)
    if image_indices.shape != (sample_count,) or len(np.unique(image_indices)) != sample_count:
        raise ValueError("visualization image indices are invalid")
    if caption_indices.shape != (sample_count,):
        raise ValueError("visualization caption indices are invalid")
    expected_set = np.sort(fixed_unordered_sample(len(corpus.image_paths), sample_count))
    if not np.array_equal(np.sort(image_indices), expected_set):
        raise ValueError("visualization does not contain the fixed sampled image set")
    expected_captions = corpus.first_caption_indices[image_indices]
    if not np.array_equal(caption_indices, expected_captions):
        raise ValueError("visualization does not use each image's first caption")
    if manifest.get("unordered_selection_sha256") != sequence_sha256(
        fixed_unordered_sample(len(corpus.image_paths), sample_count).tolist()
    ):
        raise ValueError("visualization selection hash changed")
    if manifest.get("ordered_image_index_sha256") != sequence_sha256(image_indices.tolist()):
        raise ValueError("visualization image order hash changed")
    if manifest.get("ordered_caption_index_sha256") != sequence_sha256(
        caption_indices.tolist()
    ):
        raise ValueError("visualization caption order hash changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != sample_count:
        raise ValueError("visualization manifest entries are incomplete")
    for position, (entry, image_index, caption_index) in enumerate(
        zip(entries, image_indices, caption_indices)
    ):
        if not isinstance(entry, Mapping):
            raise ValueError(f"visualization manifest entry {position} is malformed")
        expected_entry = {
            "image_index": int(image_index),
            "image_id": corpus.image_ids[int(image_index)],
            "file_name": corpus.file_names[int(image_index)],
            "caption_index": int(caption_index),
            "caption_annotation_id": corpus.caption_annotation_ids[int(caption_index)],
            "caption": corpus.captions[int(caption_index)],
        }
        if dict(entry) != expected_entry:
            raise ValueError(f"visualization manifest entry {position} changed")


def load_or_build_visualization(
    features: Mapping[str, Any],
    corpus: PaperCOCO,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_path = args.output_dir / SAMPLE_FILE
    manifest_path = args.output_dir / SAMPLE_MANIFEST_FILE
    if sample_path.is_file() and manifest_path.is_file():
        with np.load(sample_path, allow_pickle=False) as stored:
            sample = {key: np.array(stored[key], copy=True) for key in stored.files}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif sample_path.exists() or manifest_path.exists():
        raise ValueError("visualization cache is incomplete; use --restart")
    else:
        sample, manifest = build_visualization(
            features,
            corpus,
            args.sample_count,
            args.output_dir,
        )
    validate_visualization(
        sample,
        manifest,
        corpus,
        args.sample_count,
        allow_partial_smoke=args.allow_partial_smoke,
    )
    return sample, manifest


def build_analysis(
    corpus: PaperCOCO,
    layerwise: Mapping[str, Any],
    layerwise_analysis_sha256: str,
    statistics: Mapping[str, float | int],
    sample_manifest: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL if not args.allow_partial_smoke else f"{PROTOCOL}_smoke",
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "visual_checkpoint_metadata": layerwise["checkpoint_metadata"],
        "text_encoder": "frozen_dense_clip",
        "pretrained_sha256": PRETRAINED_SHA256,
        "evaluation": {
            "dataset": "COCO-val2017",
            "annotation_sha256": COCO_ANNOTATIONS_SHA256,
            "images": len(corpus.image_paths),
            "captions": len(corpus.captions),
            **corpus_identity(corpus),
        },
        "image_feature_source": {
            "study": layerwise["study"],
            "protocol": layerwise["protocol"],
            "analysis_sha256": layerwise_analysis_sha256,
            "layer": LAYER_COUNT,
            "representation": "projected_final_cls",
        },
        "statistics": dict(statistics),
        "visualization": {
            "sample_seed": SAMPLE_SEED,
            "sample_count": args.sample_count,
            "unordered_selection_sha256": sample_manifest["unordered_selection_sha256"],
            "ordering": sample_manifest["display_order"],
            "sample_file": SAMPLE_FILE,
            "manifest_file": SAMPLE_MANIFEST_FILE,
        },
        "feature_file": FEATURE_FILE,
    }


def validate_analysis(
    payload: Mapping[str, Any],
    source: str | Path,
    *,
    allow_partial_smoke: bool = False,
) -> None:
    expected_protocol = PROTOCOL if not allow_partial_smoke else f"{PROTOCOL}_smoke"
    expected = {
        "study": STUDY_NAME,
        "protocol": expected_protocol,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "text_encoder": "frozen_dense_clip",
        "pretrained_sha256": PRETRAINED_SHA256,
        "feature_file": FEATURE_FILE,
    }
    for field, wanted in expected.items():
        if payload.get(field) != wanted:
            raise ValueError(f"{source}: {field}={payload.get(field)!r}; expected {wanted!r}")
    metadata = payload.get("visual_checkpoint_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}: missing visual checkpoint metadata")
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
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{source}: missing evaluation identity")
    if allow_partial_smoke:
        image_count = int(evaluation.get("images", 0))
        caption_count = int(evaluation.get("captions", 0))
        if image_count < 2 or caption_count < 2:
            raise ValueError(f"{source}: smoke evaluation is too small")
    else:
        image_count = IMAGE_COUNT
        caption_count = CAPTION_COUNT
        evaluation_expected = {
            "image_order_sha256": COCO_IMAGE_ORDER_SHA256,
            "caption_order_sha256": COCO_CAPTION_ORDER_SHA256,
            "caption_image_index_sha256": COCO_CAPTION_IMAGE_INDEX_SHA256,
            "caption_annotation_id_sha256": COCO_CAPTION_ANNOTATION_ID_SHA256,
            "first_caption_index_sha256": COCO_FIRST_CAPTION_INDEX_SHA256,
        }
        for field, wanted in evaluation_expected.items():
            if evaluation.get(field) != wanted:
                raise ValueError(f"{source}: evaluation disagrees on {field}")
    base_evaluation = {
        "dataset": "COCO-val2017",
        "annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "images": image_count,
        "captions": caption_count,
    }
    for field, wanted in base_evaluation.items():
        if evaluation.get(field) != wanted:
            raise ValueError(f"{source}: evaluation {field} changed")
    image_source = payload.get("image_feature_source")
    if not isinstance(image_source, Mapping):
        raise ValueError(f"{source}: missing image feature source")
    expected_layerwise_protocol = (
        LAYERWISE_PROTOCOL if not allow_partial_smoke else f"{LAYERWISE_PROTOCOL}_smoke"
    )
    image_source_expected = {
        "study": LAYERWISE_STUDY,
        "protocol": expected_layerwise_protocol,
        "layer": LAYER_COUNT,
        "representation": "projected_final_cls",
    }
    for field, wanted in image_source_expected.items():
        if image_source.get(field) != wanted:
            raise ValueError(f"{source}: image feature source disagrees on {field}")
    statistics = payload.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError(f"{source}: missing full-matrix statistics")
    if statistics.get("pairwise_similarity_count") != image_count * caption_count:
        raise ValueError(f"{source}: pairwise similarity count is incomplete")
    ranges = {
        "pearson": (-1.0, 1.0),
        "matrix_cosine": (-1.0, 1.0),
        "mae": (0.0, 2.0),
        "rmse": (0.0, 2.0),
    }
    for field, (lower, upper) in ranges.items():
        value = statistics.get(field)
        if not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError(f"{source}: invalid {field}")
        if not lower <= float(value) <= upper:
            raise ValueError(f"{source}: {field} lies outside its valid range")
    visualization = payload.get("visualization")
    if not isinstance(visualization, Mapping):
        raise ValueError(f"{source}: missing visualization contract")
    sample_count = int(visualization.get("sample_count", 0))
    if not allow_partial_smoke and sample_count != SAMPLE_COUNT:
        raise ValueError(f"{source}: visualization does not contain {SAMPLE_COUNT} pairs")
    if allow_partial_smoke and not 2 <= sample_count <= image_count:
        raise ValueError(f"{source}: invalid smoke visualization size")
    if visualization.get("sample_seed") != SAMPLE_SEED:
        raise ValueError(f"{source}: visualization sample seed changed")
    if visualization.get("ordering") != ("average_linkage_dense_joint_image_text_embedding"):
        raise ValueError(f"{source}: visualization ordering changed")
    if visualization.get("sample_file") != SAMPLE_FILE:
        raise ValueError(f"{source}: visualization sample filename changed")
    if visualization.get("manifest_file") != SAMPLE_MANIFEST_FILE:
        raise ValueError(f"{source}: visualization manifest filename changed")


def clear_outputs(output_dir: Path) -> None:
    for name in (
        FEATURE_FILE,
        ANALYSIS_FILE,
        SAMPLE_FILE,
        SAMPLE_MANIFEST_FILE,
        *PLOT_FILES,
    ):
        (output_dir / name).unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    corpus, layerwise, dense_map, sparse_map, layerwise_sha = validate_args(args)
    check = {
        **protocol_manifest(),
        "layerwise_analysis": str(args.layerwise_analysis.resolve()),
        "layerwise_analysis_sha256": layerwise_sha,
        "pretrained": str(args.pretrained.resolve()),
        "pretrained_sha256": PRETRAINED_SHA256,
        "coco_annotations": str(args.coco_annotations.resolve()),
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "coco_images": str(args.coco_images.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.check_only:
        print(json.dumps(check, indent=2))
        return check
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.restart:
        clear_outputs(args.output_dir)
    features = load_or_build_features(
        corpus,
        layerwise,
        dense_map,
        sparse_map,
        layerwise_sha,
        args,
    )
    statistics = full_matrix_statistics(
        features["dense_image_features"],
        features["sparse_image_features"],
        features["text_features"],
        device=device,
        block_size=args.matrix_block_size,
    )
    _, sample_manifest = load_or_build_visualization(features, corpus, args)
    payload = build_analysis(
        corpus,
        layerwise,
        layerwise_sha,
        statistics,
        sample_manifest,
        args,
    )
    validate_analysis(
        payload,
        args.output_dir / ANALYSIS_FILE,
        allow_partial_smoke=args.allow_partial_smoke,
    )
    atomic_json(args.output_dir / ANALYSIS_FILE, payload)
    print(f"features={args.output_dir / FEATURE_FILE}")
    print(f"analysis={args.output_dir / ANALYSIS_FILE}")
    print(f"visualization={args.output_dir / SAMPLE_FILE}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

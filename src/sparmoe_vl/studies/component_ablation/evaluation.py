"""Evaluate one Table-8 component-ablation row on COCO and Flickr30k."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...baselines.retrieval import EvaluationImages, RetrievalDataset, prepare_flickr30k
from ...common.data import load_coco_retrieval
from ..generalization.protocol import load_main_vision_encoder
from .checkpoints import (
    checkpoint_metadata,
    inspect_main_checkpoint,
    torch_load,
)
from .model import CLIPSparMoE, build_model, load_controller
from .protocol import (
    CHECKPOINT_ROOT,
    COCO_ANNOTATIONS,
    COCO_IMAGES,
    EVAL_BATCH_SIZE,
    EVALUATION_COUNTS,
    EVALUATION_SEED,
    EVALUATION_SHA256,
    FLICKR_ANNOTATIONS,
    FLICKR_IMAGES,
    MAIN_OUTPUT_ROOT,
    METHOD_LABELS,
    METHODS,
    MODEL_KEY,
    MODEL_NAME,
    NUM_WORKERS,
    OUTPUT_ROOT,
    PRETRAINED,
    REPLACEMENTS,
    STUDY_NAME,
    TRAINED_METHODS,
    validate_method_seed,
)


@dataclass(frozen=True)
class LoadedModel:
    encoder: Any
    backbone: Any
    preprocess: Any
    tokenizer: Any
    metadata: dict[str, Any]
    legacy: bool


def default_checkpoint(method: str, seed: int) -> Path:
    if method in ("without_token_router", "sparmoe_vl"):
        return MAIN_OUTPUT_ROOT / f"seed_{seed}" / "stage2" / "best.pt"
    return CHECKPOINT_ROOT / method / f"seed_{seed}" / "stage2" / "best.pt"


def default_output(method: str, seed: int) -> Path:
    return OUTPUT_ROOT / "evaluation" / method / f"seed_{seed}" / "result.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--coco-annotations", type=Path, default=COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=COCO_IMAGES)
    parser.add_argument("--flickr-annotations", type=Path, default=FLICKR_ANNOTATIONS)
    parser.add_argument("--flickr-images", type=Path, default=FLICKR_IMAGES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    validate_method_seed(args.method, args.seed)
    args.checkpoint = args.checkpoint or default_checkpoint(args.method, args.seed)
    args.output = args.output or default_output(args.method, args.seed)
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_args(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], RetrievalDataset, RetrievalDataset]:
    for path, label in (
        (args.checkpoint, "Table-8 checkpoint"),
        (args.pretrained, "Dense CLIP ViT-L/14 weights"),
        (args.coco_annotations, "COCO val2017 annotations"),
        (args.flickr_annotations, "Flickr30k annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    for path, label in (
        (args.coco_images, "COCO val2017 images"),
        (args.flickr_images, "Flickr30k images"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"missing {label}: {path}")
    identities = {
        "coco": file_sha256(args.coco_annotations),
        "flickr30k": file_sha256(args.flickr_annotations),
    }
    if identities != EVALUATION_SHA256:
        raise RuntimeError("retrieval annotations differ from the Table-8 protocol")
    if args.method in TRAINED_METHODS:
        metadata = checkpoint_metadata(torch_load(args.checkpoint), args.method)
    else:
        metadata = inspect_main_checkpoint(args.checkpoint, args.seed)
    if metadata["training_seed"] != args.seed:
        raise ValueError(
            f"checkpoint seed={metadata['training_seed']}; requested seed={args.seed}"
        )
    coco_raw = load_coco_retrieval(args.coco_annotations, args.coco_images, 5_000)
    coco = RetrievalDataset(
        coco_raw.image_paths,
        coco_raw.captions,
        coco_raw.caption_image_indices,
    )
    flickr = prepare_flickr30k(args.flickr_annotations, args.flickr_images)
    counts = {
        "coco": {"images": len(coco.image_paths), "texts": len(coco.captions)},
        "flickr30k": {
            "images": len(flickr.image_paths),
            "texts": len(flickr.captions),
        },
    }
    if counts != EVALUATION_COUNTS:
        raise RuntimeError(f"Table-8 retrieval counts changed: {counts}")
    metadata["evaluation_sha256"] = identities
    metadata["evaluation_counts"] = counts
    return metadata, coco, flickr


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_model(args: argparse.Namespace, device: torch.device) -> LoadedModel:
    if args.method not in TRAINED_METHODS:
        encoder, metadata, preprocess, tokenizer = load_main_vision_encoder(
            args.checkpoint,
            args.pretrained,
            device,
        )
        return LoadedModel(
            encoder,
            encoder.clip_model,
            preprocess,
            tokenizer,
            metadata,
            False,
        )
    checkpoint = torch_load(args.checkpoint)
    metadata = checkpoint_metadata(checkpoint, args.method)
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("open_clip_torch is required for Table-8 evaluation") from error
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(args.pretrained),
        force_quick_gelu=True,
    )
    clip_model = clip_model.to(device).eval()
    model, _ = build_model(clip_model, args.method, args.seed, training_stage=2)
    model = model.to(device)
    load_controller(model, checkpoint, args.method)
    model.eval()
    model.set_routing_mode("learned")
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return LoadedModel(model, clip_model, preprocess, tokenizer, metadata, True)


def image_loader(
    dataset: RetrievalDataset,
    preprocess: Any,
    device: torch.device,
) -> DataLoader[Tensor]:
    return DataLoader(
        EvaluationImages(dataset.image_paths, preprocess),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )


@torch.inference_mode()
def encode_texts(
    bundle: LoadedModel,
    captions: Sequence[str],
    device: torch.device,
    description: str,
) -> Tensor:
    features = []
    for start in tqdm(range(0, len(captions), EVAL_BATCH_SIZE), desc=description):
        tokens = bundle.tokenizer(list(captions[start : start + EVAL_BATCH_SIZE])).to(device)
        features.append(F.normalize(bundle.backbone.encode_text(tokens), dim=-1).cpu())
    return torch.cat(features)


def sparse_batch(
    bundle: LoadedModel,
    images: Tensor,
    routing_mode: str,
) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
    if bundle.legacy:
        model: CLIPSparMoE = bundle.encoder
        model.set_routing_mode(routing_mode)
        features, auxiliary = model.encode_sparse(images)
        routing = [(item["G"], item["masks"]) for item in auxiliary]
        return features, routing
    output = bundle.encoder(images, routing_mode=routing_mode)
    routing = [(layer.routing.gates, layer.sparse_pattern.masks) for layer in output.layers]
    return output.features, routing


@torch.inference_mode()
def encode_images(
    bundle: LoadedModel,
    dataset: RetrievalDataset,
    device: torch.device,
    routing_mode: str,
    description: str,
) -> tuple[Tensor, Tensor, Optional[list[float]]]:
    sparse_features = []
    dense_features = []
    ratio_sum = torch.zeros(24, dtype=torch.float64)
    image_count = 0
    for images in tqdm(image_loader(dataset, bundle.preprocess, device), desc=description):
        current = len(images)
        images = images.to(device, non_blocking=device.type == "cuda")
        dense = bundle.backbone.encode_image(images)
        encoded, routing = sparse_batch(bundle, images, routing_mode)
        dense_features.append(F.normalize(dense, dim=-1).cpu())
        sparse_features.append(encoded.cpu())
        image_count += current
        for layer_index, (gates, masks) in enumerate(routing):
            widths = masks.float().mean(-1)
            usage = gates.float().mean(0)
            ratio_sum[layer_index] += float((usage * widths).sum()) * current
    ratios = (ratio_sum / image_count).tolist()
    return torch.cat(sparse_features), torch.cat(dense_features), ratios


def retrieval_r1(
    images: Tensor,
    texts: Tensor,
    caption_image_indices: Sequence[int],
) -> tuple[float, float]:
    similarity = images.float() @ texts.float().T
    ground_truth = torch.tensor(caption_image_indices, dtype=torch.long)
    top_text = similarity.topk(1, dim=1).indices
    top_image = similarity.topk(1, dim=0).indices.T
    image_ids = torch.arange(len(images))[:, None]
    i2t = (ground_truth[top_text] == image_ids).any(1).float().mean()
    t2i = (top_image == ground_truth[:, None]).any(1).float().mean()
    return float(i2t * 100), float(t2i * 100)


def ffn_macs(layer_ratios: Sequence[float]) -> float:
    if len(layer_ratios) != 24:
        raise ValueError(f"expected 24 layer ratios, got {len(layer_ratios)}")
    return (24 + 256 * sum(float(value) for value in layer_ratios)) * 2 * 1024 * 4096 / 1e9


def seed_random_routing(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> dict[str, Any]:
    metadata, coco, flickr = validate_args(args)
    if args.check_only:
        result = {
            "study": STUDY_NAME,
            "paper_scope": "Table 8",
            "method": args.method,
            "label": METHOD_LABELS[args.method],
            "replacement": REPLACEMENTS[args.method],
            "seed": args.seed,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_metadata": metadata,
            "evaluation_seed": EVALUATION_SEED,
            "output": str(args.output.resolve()),
        }
        print(json.dumps(result, indent=2))
        return result
    device = resolve_device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    bundle = load_model(args, device)
    routing_mode = "random" if args.method == "without_token_router" else "learned"
    text_features = {
        "coco": encode_texts(bundle, coco.captions, device, "COCO Dense texts"),
        "flickr30k": encode_texts(
            bundle,
            flickr.captions,
            device,
            "Flickr30k Dense texts",
        ),
    }
    if routing_mode == "random":
        seed_random_routing(EVALUATION_SEED, device)
    coco_images, dense_coco_images, ratios = encode_images(
        bundle,
        coco,
        device,
        routing_mode,
        "COCO sparse images",
    )
    if ratios is None:
        raise RuntimeError("COCO routing ratios were not collected")
    if routing_mode == "random":
        seed_random_routing(EVALUATION_SEED + 1, device)
    flickr_images, dense_flickr_images, _ = encode_images(
        bundle,
        flickr,
        device,
        routing_mode,
        "Flickr30k sparse images",
    )
    coco_i2t, coco_t2i = retrieval_r1(
        coco_images,
        text_features["coco"],
        coco.caption_image_indices,
    )
    flickr_i2t, flickr_t2i = retrieval_r1(
        flickr_images,
        text_features["flickr30k"],
        flickr.caption_image_indices,
    )
    dense_coco_i2t, dense_coco_t2i = retrieval_r1(
        dense_coco_images,
        text_features["coco"],
        coco.caption_image_indices,
    )
    dense_flickr_i2t, dense_flickr_t2i = retrieval_r1(
        dense_flickr_images,
        text_features["flickr30k"],
        flickr.caption_image_indices,
    )
    dense_reference = {
        "ffn_macs_v_g": ffn_macs([1.0] * 24),
        "coco_i2t_r1": dense_coco_i2t,
        "coco_t2i_r1": dense_coco_t2i,
        "flickr_i2t_r1": dense_flickr_i2t,
        "flickr_t2i_r1": dense_flickr_t2i,
    }
    metrics = {
        "ffn_macs_v_g": ffn_macs(ratios),
        "coco_i2t_r1": coco_i2t,
        "coco_t2i_r1": coco_t2i,
        "flickr_i2t_r1": flickr_i2t,
        "flickr_t2i_r1": flickr_t2i,
    }
    metrics["retention_percent"] = 25.0 * sum(
        metrics[key] / dense_reference[key]
        for key in (
            "coco_i2t_r1",
            "coco_t2i_r1",
            "flickr_i2t_r1",
            "flickr_t2i_r1",
        )
    )
    result = {
        "format_version": 1,
        "study": STUDY_NAME,
        "paper_scope": "Table 8",
        "method": args.method,
        "label": METHOD_LABELS[args.method],
        "replacement": REPLACEMENTS[args.method],
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "run_seed": args.seed,
        "data_seed": metadata["data_seed"],
        "dataset_sha256": metadata["dataset_sha256"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": metadata["checkpoint_step"],
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_batch_size": EVAL_BATCH_SIZE,
        "evaluation_num_workers": NUM_WORKERS,
        "evaluation_sha256": metadata["evaluation_sha256"],
        "counts": metadata["evaluation_counts"],
        "routing": (
            "uniform_random_per_patch_token" if routing_mode == "random" else "learned_argmax"
        ),
        "dense_reference": dense_reference,
        "metrics": metrics,
        "layer_routed_ratios": ratios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"result={args.output}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))

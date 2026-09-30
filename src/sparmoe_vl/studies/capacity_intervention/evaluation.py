"""Evaluate all token- and layer-capacity interventions for one Table-7 seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...common.data import ImagePathDataset, RetrievalCorpus, load_coco_retrieval
from ...common.training import set_reproducible_seed
from .checkpoints import checkpoint_metadata, torch_load
from .model import (
    CLIPSparMoE,
    build_model,
    layer_base_logits,
    load_controller,
    set_layer_allocation,
)
from .protocol import (
    ALLOCATIONS,
    CHECKPOINT_ROOT,
    COCO_ANNOTATIONS,
    COCO_ANNOTATIONS_SHA256,
    COCO_IMAGES,
    EVAL_BATCH_SIZE,
    EVAL_IMAGES,
    MODEL_KEY,
    MODEL_NAME,
    NUM_WORKERS,
    OUTPUT_ROOT,
    PRETRAINED,
    RECOVERY_COSINE_THRESHOLD,
    STUDY_NAME,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--coco-annotations", type=Path, default=COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=COCO_IMAGES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint or CHECKPOINT_ROOT / f"seed_{args.seed}.pt"
    args.output = (
        args.output or OUTPUT_ROOT / "evaluation" / f"seed_{args.seed}" / "result.json"
    )
    args.batch_size = EVAL_BATCH_SIZE
    args.num_workers = NUM_WORKERS
    args.max_images = EVAL_IMAGES
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_args(args: argparse.Namespace) -> tuple[dict[str, Any], RetrievalCorpus]:
    for path, label in (
        (args.checkpoint, "Table-7 N=8 checkpoint"),
        (args.pretrained, "Dense CLIP ViT-L/14 weights"),
        (args.coco_annotations, "COCO val2017 annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not args.coco_images.is_dir():
        raise FileNotFoundError(f"missing COCO val2017 image root: {args.coco_images}")
    annotation_sha256 = file_sha256(args.coco_annotations)
    if annotation_sha256 != COCO_ANNOTATIONS_SHA256:
        raise RuntimeError("COCO annotations differ from the Table-7 evaluation set")
    metadata = checkpoint_metadata(torch_load(args.checkpoint))
    if metadata["training_seed"] != args.seed:
        raise ValueError(
            f"checkpoint seed={metadata['training_seed']}; requested seed={args.seed}"
        )
    corpus = load_coco_retrieval(
        args.coco_annotations,
        args.coco_images,
        max_images=EVAL_IMAGES,
    )
    if len(corpus.image_paths) != EVAL_IMAGES or len(corpus.captions) != 25_014:
        raise RuntimeError(
            "unexpected COCO Table-7 counts: "
            f"images={len(corpus.image_paths)}, captions={len(corpus.captions)}"
        )
    metadata["evaluation_annotation_sha256"] = annotation_sha256
    return metadata, corpus


def image_loader(
    corpus: RetrievalCorpus,
    preprocess: Any,
    device: torch.device,
) -> DataLoader[Tensor]:
    return DataLoader(
        ImagePathDataset(corpus.image_paths, preprocess),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )


def activated_ffn_macs(layer_ratios: Sequence[float]) -> float:
    """Return the paper's visual FFN-only MAC convention, including dense CLS."""

    if len(layer_ratios) != 24:
        raise ValueError(f"expected 24 layer ratios, got {len(layer_ratios)}")
    return (24 + 256 * sum(float(value) for value in layer_ratios)) * 2 * 1024 * 4096 / 1e9


@torch.inference_mode()
def encode_dense(
    model: CLIPSparMoE,
    corpus: RetrievalCorpus,
    preprocess: Any,
    device: torch.device,
) -> Tensor:
    features = []
    for images in tqdm(
        image_loader(corpus, preprocess, device),
        desc="Table 7 Dense",
    ):
        images = images.to(device, non_blocking=device.type == "cuda")
        features.append(model.dense_features(images).cpu())
    return torch.cat(features)


@torch.inference_mode()
def encode_intervention(
    model: CLIPSparMoE,
    corpus: RetrievalCorpus,
    preprocess: Any,
    device: torch.device,
    description: str,
) -> tuple[Tensor, float, list[float]]:
    features = []
    ratio_sums = torch.zeros(len(model.layers), dtype=torch.float64)
    image_count = 0
    for images in tqdm(
        image_loader(corpus, preprocess, device),
        desc=description,
    ):
        batch_size = len(images)
        images = images.to(device, non_blocking=device.type == "cuda")
        encoded, auxiliary = model.encode_sparse(images)
        features.append(encoded.cpu())
        image_count += batch_size
        for layer_index, item in enumerate(auxiliary):
            measured_widths = item["masks"].float().mean(-1)
            usage = item["G"].float().mean(0)
            ratio_sums[layer_index] += float((usage * measured_widths).sum()) * batch_size
    if image_count != EVAL_IMAGES:
        raise RuntimeError(f"evaluated {image_count} images; expected {EVAL_IMAGES}")
    layer_ratios = ratio_sums / image_count
    ffn_macs = activated_ffn_macs(layer_ratios.tolist())
    return torch.cat(features), ffn_macs, layer_ratios.tolist()


def intervention_metrics(
    allocation: str,
    sparse: Tensor,
    dense: Tensor,
    ffn_macs: float,
    layer_ratios: list[float],
) -> dict[str, Any]:
    cosine = F.cosine_similarity(sparse, dense, dim=-1)
    nre = (sparse - dense).norm(dim=-1).mean() / dense.norm(dim=-1).mean()
    return {
        "Allocation": allocation,
        "Activated FFN MACs (G)": ffn_macs,
        "Cosine ↑": float(cosine.mean()),
        "NRE ↓": float(nre),
        "Recovery Rate ↑": float((cosine >= RECOVERY_COSINE_THRESHOLD).float().mean()),
        "layer_routed_ratios": layer_ratios,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    metadata, corpus = validate_args(args)
    if args.check_only:
        result = {
            "study": STUDY_NAME,
            "model_name": MODEL_NAME,
            "model_key": MODEL_KEY,
            "checkpoint": metadata,
            "evaluation": {
                "images": len(corpus.image_paths),
                "captions": len(corpus.captions),
                "annotation_sha256": metadata["evaluation_annotation_sha256"],
                "batch_size": EVAL_BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "recovery_definition": (
                    f"per-image feature cosine >= {RECOVERY_COSINE_THRESHOLD}"
                ),
                "allocations": list(ALLOCATIONS),
            },
            "output": str(args.output.resolve()),
        }
        print(json.dumps(result, indent=2))
        return result
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_reproducible_seed(42)
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("open_clip_torch is required for Table-7 evaluation") from error
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(args.pretrained),
        force_quick_gelu=True,
    )
    clip_model = clip_model.to(device).eval()
    model = build_model(clip_model).to(device)
    checkpoint = torch_load(args.checkpoint)
    load_controller(model, checkpoint)
    del checkpoint
    model.eval()
    original_logits = layer_base_logits(model)
    dense = encode_dense(model, corpus, preprocess, device)
    configurations = (
        ("Self", "learned", "self"),
        ("Uniform", "uniform", "self"),
        ("Shuffled", "shuffled", "self"),
        ("Layer-Uniform", "learned", "uniform"),
        ("Layer-Shuffled", "learned", "shuffled"),
    )
    rows = {}
    for allocation, routing_mode, layer_mode in configurations:
        set_layer_allocation(model, layer_mode, original_logits)
        model.set_routing_mode(routing_mode)
        sparse, ffn_macs, layer_ratios = encode_intervention(
            model,
            corpus,
            preprocess,
            device,
            f"Table 7 {allocation}",
        )
        rows[allocation] = intervention_metrics(
            allocation,
            sparse,
            dense,
            ffn_macs,
            layer_ratios,
        )
    set_layer_allocation(model, "self", original_logits)
    result = {
        "format_version": 1,
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": metadata["checkpoint_step"],
        "run_seed": metadata["training_seed"],
        "data_seed": metadata["data_seed"],
        "dataset_sha256": metadata["dataset_sha256"],
        "training_protocol": metadata["protocol"],
        "images": EVAL_IMAGES,
        "batch_size": EVAL_BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "evaluation_annotation_sha256": metadata["evaluation_annotation_sha256"],
        "recovery_definition": (f"per-image feature cosine >= {RECOVERY_COSINE_THRESHOLD}"),
        "completed": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"result={args.output}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))

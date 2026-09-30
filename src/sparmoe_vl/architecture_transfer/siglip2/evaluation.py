"""COCO/Flickr30k retrieval evaluation for the SigLIP2 Table-6 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from ..siglip.evaluation import (
    EVALUATION_SHA256,
    _corpora,
    _dense_images,
    _dense_texts,
    _sparse_images,
    _sparse_texts,
    retrieval_r1,
)
from ..siglip.training import LocalSigLIPTokenizer
from .checkpoints import checkpoint_metadata, torch_load
from .model import build_model, load_stage2_controller
from .protocol import (
    CAPACITY_FACTORS,
    COCO_ANNOTATIONS,
    COCO_IMAGES,
    FLICKR_ANNOTATIONS,
    FLICKR_IMAGES,
    PROJECT_ROOT,
    TOKENIZER_JSON_SHA256,
    get_spec,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_checkpoint(model: str, modality: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "checkpoints"
        / "architecture_transfer"
        / "siglip2"
        / model
        / modality
        / f"seed_{seed}.pt"
    )


def default_output(model: str, modality: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "architecture_transfer"
        / "siglip2"
        / model
        / modality
        / f"seed_{seed}"
        / "retrieval.json"
    )


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    model: str,
) -> argparse.Namespace:
    spec = get_spec(model)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("vision", "text"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pretrained", type=Path, default=spec.pretrained)
    parser.add_argument("--tokenizer-path", type=Path, default=spec.tokenizer)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--coco-annotations", type=Path, default=COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=COCO_IMAGES)
    parser.add_argument("--flickr-annotations", type=Path, default=FLICKR_ANNOTATIONS)
    parser.add_argument("--flickr-images", type=Path, default=FLICKR_IMAGES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    args.model = model
    args.checkpoint = args.checkpoint or default_checkpoint(model, args.modality, args.seed)
    args.batch_size = args.batch_size or (32 if args.modality == "vision" else 128)
    args.output = args.output or default_output(model, args.modality, args.seed)
    return args


def validate_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    for path, label in (
        (args.checkpoint, "Stage-2 checkpoint"),
        (args.pretrained, "SigLIP2 pretrained weights"),
        (args.coco_annotations, "COCO annotations"),
        (args.flickr_annotations, "Flickr30k annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    for path, label in (
        (args.tokenizer_path, "local SigLIP2 tokenizer"),
        (args.coco_images, "COCO image root"),
        (args.flickr_images, "Flickr30k image root"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"missing {label}: {path}")
    if _file_sha256(args.tokenizer_path / "tokenizer.json") != TOKENIZER_JSON_SHA256:
        raise RuntimeError("SigLIP2 tokenizer differs from the result-generating tokenizer")
    identities = {
        "coco": _file_sha256(args.coco_annotations),
        "flickr30k": _file_sha256(args.flickr_annotations),
    }
    if identities != EVALUATION_SHA256:
        raise RuntimeError("retrieval annotations differ from the paper evaluation protocol")
    checkpoint = torch_load(args.checkpoint)
    metadata = checkpoint_metadata(
        checkpoint,
        get_spec(args.model),
        args.modality,
        2,
    )
    if metadata["training_seed"] != args.seed:
        raise ValueError(
            f"checkpoint seed={metadata['training_seed']}; requested seed={args.seed}"
        )
    metadata["evaluation_sha256"] = identities
    return metadata


def dense_macs(model_name: str, modality: str) -> dict[str, float]:
    """Static MAC convention used by the result-generating SigLIP2 evaluators."""

    spec = get_spec(model_name)
    if modality == "vision":
        tokens = spec.num_patches
        head_dim = spec.model_dim // spec.num_heads
        patch_embedding = (
            spec.num_patches * (3 * spec.patch_size * spec.patch_size) * spec.model_dim
        )
        attention = spec.num_layers * (
            tokens * spec.model_dim * (3 * spec.model_dim)
            + spec.num_heads * tokens * tokens * head_dim * 2
            + tokens * spec.model_dim * spec.model_dim
        )
        ffn = spec.num_layers * tokens * 2 * spec.model_dim * spec.ffn_dim
        projection = spec.num_patches * spec.model_dim * spec.model_dim
        non_ffn = patch_embedding + attention + projection
    elif modality == "text":
        tokens = spec.context_length
        head_dim = spec.model_dim // spec.num_heads
        non_ffn = spec.num_layers * (
            3 * tokens * spec.model_dim * spec.model_dim
            + spec.num_heads * tokens * tokens * head_dim * 2
            + tokens * spec.model_dim * spec.model_dim
        )
        ffn = spec.num_layers * 2 * tokens * spec.model_dim * spec.ffn_dim
    else:
        raise ValueError("modality must be 'vision' or 'text'")
    return {
        "non_ffn_g": non_ffn / 1e9,
        "ffn_g": ffn / 1e9,
        "total_g": (non_ffn + ffn) / 1e9,
    }


def sparse_macs(
    model_name: str,
    modality: str,
    ratios: Sequence[float],
) -> dict[str, float]:
    spec = get_spec(model_name)
    if len(ratios) != spec.num_layers:
        raise ValueError(f"expected {spec.num_layers} layer ratios, got {len(ratios)}")
    dense = dense_macs(model_name, modality)
    if modality == "vision":
        ffn = spec.num_patches * sum(ratios) * 2 * spec.model_dim * spec.ffn_dim
    elif modality == "text":
        ffn = sum(
            (1 + (spec.context_length - 1) * ratio) * 2 * spec.model_dim * spec.ffn_dim
            for ratio in ratios
        )
    else:
        raise ValueError("modality must be 'vision' or 'text'")
    ffn_g = ffn / 1e9
    return {
        "ffn_g": ffn_g,
        "total_g": dense["non_ffn_g"] + ffn_g,
        "reduction_pct": 100.0 * (1.0 - ffn_g / dense["ffn_g"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    metadata = validate_args(args)
    if args.check_only:
        result = {
            "family": "SigLIP2",
            "version": args.model,
            "modality": args.modality,
            "checkpoint": metadata,
            "batch_size": args.batch_size,
            "output": str(args.output.resolve()),
        }
        print(json.dumps(result, indent=2))
        return result
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    spec = get_spec(args.model)
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("open_clip_torch is required for evaluation") from error
    backbone, _, preprocess = open_clip.create_model_and_transforms(
        spec.model_name,
        pretrained=str(args.pretrained),
    )
    backbone = backbone.to(device).eval()
    tokenizer = LocalSigLIPTokenizer(args.tokenizer_path, spec.context_length)
    coco, flickr = _corpora(args)
    checkpoint = torch_load(args.checkpoint)
    model = build_model(
        backbone,
        args.modality,
        2,
        spec.target_ratio(args.modality),
        CAPACITY_FACTORS,
        0.4,
    ).to(device)
    load_stage2_controller(model, checkpoint)
    del checkpoint
    model.eval()
    model.set_routing_mode("learned")

    if args.modality == "vision":
        coco_text = _dense_texts(
            backbone, tokenizer, coco.captions, device, args.batch_size, "COCO text"
        )
        flickr_text = _dense_texts(
            backbone,
            tokenizer,
            flickr.captions,
            device,
            args.batch_size,
            "Flickr30k text",
        )
        coco_dense = _dense_images(
            backbone, coco, preprocess, device, args.batch_size, args.num_workers, "COCO dense"
        )
        flickr_dense = _dense_images(
            backbone,
            flickr,
            preprocess,
            device,
            args.batch_size,
            args.num_workers,
            "Flickr30k dense",
        )
        dense_coco = retrieval_r1(coco_dense, coco_text, coco.caption_image_indices)
        dense_flickr = retrieval_r1(flickr_dense, flickr_text, flickr.caption_image_indices)
        coco_sparse, coco_sum, coco_count = _sparse_images(
            model, coco, preprocess, device, args.batch_size, args.num_workers, "COCO sparse"
        )
        flickr_sparse, flickr_sum, flickr_count = _sparse_images(
            model,
            flickr,
            preprocess,
            device,
            args.batch_size,
            args.num_workers,
            "Flickr30k sparse",
        )
        sparse_coco = retrieval_r1(coco_sparse, coco_text, coco.caption_image_indices)
        sparse_flickr = retrieval_r1(flickr_sparse, flickr_text, flickr.caption_image_indices)
        layer_ratios = ((coco_sum + flickr_sum) / (coco_count + flickr_count)).tolist()
        ratio_basis = "dataset_weighted_COCO_and_Flickr_learned_patch_routes"
    else:
        coco_images = _dense_images(
            backbone, coco, preprocess, device, args.batch_size, args.num_workers, "COCO images"
        )
        flickr_images = _dense_images(
            backbone,
            flickr,
            preprocess,
            device,
            args.batch_size,
            args.num_workers,
            "Flickr30k images",
        )
        coco_dense = _dense_texts(
            backbone, tokenizer, coco.captions, device, args.batch_size, "COCO dense text"
        )
        flickr_dense = _dense_texts(
            backbone,
            tokenizer,
            flickr.captions,
            device,
            args.batch_size,
            "Flickr30k dense text",
        )
        dense_coco = retrieval_r1(coco_images, coco_dense, coco.caption_image_indices)
        dense_flickr = retrieval_r1(flickr_images, flickr_dense, flickr.caption_image_indices)
        coco_sparse, layer_ratios = _sparse_texts(
            model,
            tokenizer,
            coco.captions,
            device,
            args.batch_size,
            "COCO sparse text",
            collect_ratios=True,
        )
        flickr_sparse, _ = _sparse_texts(
            model,
            tokenizer,
            flickr.captions,
            device,
            args.batch_size,
            "Flickr30k sparse text",
            collect_ratios=False,
        )
        sparse_coco = retrieval_r1(coco_images, coco_sparse, coco.caption_image_indices)
        sparse_flickr = retrieval_r1(flickr_images, flickr_sparse, flickr.caption_image_indices)
        ratio_basis = "COCO_learned_token_routes"

    dense_compute = dense_macs(args.model, args.modality)
    sparse_compute = sparse_macs(args.model, args.modality, layer_ratios)
    retention = 25.0 * sum(
        (
            sparse_coco["I2T_R1"] / dense_coco["I2T_R1"],
            sparse_coco["T2I_R1"] / dense_coco["T2I_R1"],
            sparse_flickr["I2T_R1"] / dense_flickr["I2T_R1"],
            sparse_flickr["T2I_R1"] / dense_flickr["T2I_R1"],
        )
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": metadata["checkpoint_step"],
        "run_seed": metadata["training_seed"],
        "data_seed": metadata["data_seed"],
        "dataset_sha256": metadata["dataset_sha256"],
        "protocol": metadata["protocol"],
        "model_key": spec.key,
        "model_name": spec.model_name,
        "modality": args.modality,
        "p_target": metadata["target_ratio"],
        "levels": metadata["levels"],
        "evaluation": {
            "coco_images": len(coco.image_paths),
            "coco_captions": len(coco.captions),
            "flickr_images": len(flickr.image_paths),
            "flickr_captions": len(flickr.captions),
            "annotation_sha256": metadata["evaluation_sha256"],
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "ffn_macs_basis": ratio_basis,
            "retention_basis": "mean_of_four_sparse_over_dense_R1_ratios",
        },
        "dense": {
            "total_macs_g": dense_compute["total_g"],
            "ffn_macs_g": dense_compute["ffn_g"],
            "COCO_I2T_R1": dense_coco["I2T_R1"],
            "COCO_T2I_R1": dense_coco["T2I_R1"],
            "Flickr_I2T_R1": dense_flickr["I2T_R1"],
            "Flickr_T2I_R1": dense_flickr["T2I_R1"],
        },
        "sparse": {
            "total_macs_g": sparse_compute["total_g"],
            "ffn_macs_g": sparse_compute["ffn_g"],
            "ffn_reduction_pct": sparse_compute["reduction_pct"],
            "COCO_I2T_R1": sparse_coco["I2T_R1"],
            "COCO_T2I_R1": sparse_coco["T2I_R1"],
            "Flickr_I2T_R1": sparse_flickr["I2T_R1"],
            "Flickr_T2I_R1": sparse_flickr["T2I_R1"],
            "average_retention_pct": retention,
            "layer_routed_ratios": layer_ratios,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["sparse"], indent=2))
    print(f"result={args.output}")
    return result


def main(model: str, argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv, model=model))

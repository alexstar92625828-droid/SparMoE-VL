"""Measure patch-token FFN capacity requirements in frozen Dense CLIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from ...common.data import load_coco_retrieval
from .protocol import (
    BATCH_SIZE,
    CALIBRATION_IMAGES,
    CAPACITY_LEVELS,
    CHANNEL_RANKING,
    COCO_ANNOTATIONS,
    COCO_ANNOTATIONS_SHA256,
    COCO_IMAGES,
    COCO_IMAGES_TOTAL,
    COMPARISON,
    COSINE_THRESHOLD,
    EVALUATION_IMAGES,
    FFN_DIM,
    MODEL_KEY,
    MODEL_NAME,
    NRE_THRESHOLD,
    NUM_LAYERS,
    OUTPUT_ROOT,
    PAPER_SCOPE,
    PATCH_TOKENS,
    PRETRAINED,
    PRETRAINED_SHA256,
    PROTOCOL,
    SEED,
    SPLIT_METHOD,
    STUDY_NAME,
    protocol_manifest,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--coco-annotations", type=Path, default=COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=COCO_IMAGES)
    parser.add_argument("--calibration-images", type=int, default=CALIBRATION_IMAGES)
    parser.add_argument("--evaluation-images", type=int, default=EVALUATION_IMAGES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--cosine-threshold", type=float, default=COSINE_THRESHOLD)
    parser.add_argument("--nre-threshold", type=float, default=NRE_THRESHOLD)
    parser.add_argument("--layer-start", type=int, default=1)
    parser.add_argument("--layer-end", type=int, default=NUM_LAYERS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "shards")
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


def build_split(
    num_images: int,
    seed: int,
    calibration_images: int,
    evaluation_images: int,
    *,
    require_full: bool = True,
) -> tuple[list[int], list[int]]:
    requested = calibration_images + evaluation_images
    if min(num_images, calibration_images, evaluation_images) <= 0:
        raise ValueError("image counts must be positive")
    if require_full and requested != num_images:
        raise ValueError(
            "strict protocol must consume every COCO image: "
            f"requested={requested}, available={num_images}"
        )
    if requested > num_images:
        raise ValueError(f"requested {requested} images, but only {num_images} are available")
    permutation = torch.randperm(
        num_images,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    calibration = permutation[:calibration_images]
    evaluation = permutation[calibration_images:requested]
    if set(calibration) & set(evaluation):
        raise AssertionError("calibration and evaluation splits overlap")
    if require_full and len(set(calibration + evaluation)) != num_images:
        raise AssertionError("strict split does not cover every image exactly once")
    return calibration, evaluation


def split_identity(indices: Sequence[int], image_paths: Sequence[Path]) -> str:
    return sequence_sha256([image_paths[index].name for index in indices])


def validate_args(args: argparse.Namespace) -> tuple[tuple[Path, ...], list[int], list[int]]:
    if not args.pretrained.is_file():
        raise FileNotFoundError(f"missing Dense CLIP weights: {args.pretrained}")
    if not args.coco_annotations.is_file():
        raise FileNotFoundError(f"missing COCO annotations: {args.coco_annotations}")
    if not args.coco_images.is_dir():
        raise FileNotFoundError(f"missing COCO image root: {args.coco_images}")
    if file_sha256(args.pretrained) != PRETRAINED_SHA256:
        raise ValueError("Dense CLIP checkpoint identity differs from Figure 3")
    if file_sha256(args.coco_annotations) != COCO_ANNOTATIONS_SHA256:
        raise ValueError("COCO annotation identity differs from Figure 3")
    if args.seed != SEED:
        raise ValueError(f"Figure 3 is fixed to seed {SEED}")
    if not 1 <= args.layer_start <= args.layer_end <= NUM_LAYERS:
        raise ValueError(f"layer range must lie within 1..{NUM_LAYERS}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not args.allow_partial_smoke:
        expected = {
            "calibration_images": CALIBRATION_IMAGES,
            "evaluation_images": EVALUATION_IMAGES,
            "batch_size": BATCH_SIZE,
            "cosine_threshold": COSINE_THRESHOLD,
            "nre_threshold": NRE_THRESHOLD,
        }
        for field, wanted in expected.items():
            if getattr(args, field) != wanted:
                raise ValueError(
                    f"Figure-3 {field}={getattr(args, field)!r}; expected {wanted!r}"
                )
    corpus = load_coco_retrieval(
        args.coco_annotations,
        args.coco_images,
        max_images=COCO_IMAGES_TOTAL,
    )
    image_paths = tuple(corpus.image_paths)
    if len(image_paths) != COCO_IMAGES_TOTAL:
        raise RuntimeError(
            f"Figure 3 requires {COCO_IMAGES_TOTAL} COCO images, got {len(image_paths)}"
        )
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"COCO image files are incomplete; first missing: {missing[0]}")
    calibration, evaluation = build_split(
        len(image_paths),
        args.seed,
        args.calibration_images,
        args.evaluation_images,
        require_full=not args.allow_partial_smoke,
    )
    return image_paths, calibration, evaluation


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_images(
    indices: Sequence[int],
    image_paths: Sequence[Path],
    preprocess: Any,
    device: torch.device,
) -> Tensor:
    images = []
    for index in indices:
        path = image_paths[index]
        try:
            with Image.open(path) as image:
                images.append(preprocess(image.convert("RGB")))
        except Exception as error:
            raise RuntimeError(f"unable to read COCO image: {path}") from error
    return torch.stack(images).to(device, non_blocking=device.type == "cuda")


def register_captures(
    model: Any,
    layer_indices: Sequence[int],
) -> tuple[dict[int, Tensor], list[Any]]:
    captures: dict[int, Tensor] = {}
    hooks = []
    for layer in layer_indices:
        module = model.visual.transformer.resblocks[layer].mlp.c_fc

        def capture(_module: Any, _inputs: Any, output: Tensor, layer: int = layer) -> None:
            captures[layer] = output.detach()

        hooks.append(module.register_forward_hook(capture))
    return captures, hooks


def validate_capture(capture: Tensor, batch_size: int) -> None:
    expected = (batch_size, PATCH_TOKENS + 1, FFN_DIM)
    if tuple(capture.shape) != expected:
        raise RuntimeError(
            "unexpected Dense FFN tensor shape; refusing to guess token axes: "
            f"observed={tuple(capture.shape)}, expected={expected}"
        )


def calibrate_rankings(
    model: Any,
    preprocess: Any,
    image_paths: Sequence[Path],
    calibration_indices: Sequence[int],
    layer_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    captures, hooks = register_captures(model, layer_indices)
    scores = {
        layer: torch.zeros(FFN_DIM, dtype=torch.float64, device=device)
        for layer in layer_indices
    }
    token_count = 0
    try:
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(calibration_indices), batch_size),
                desc=f"Calibrate L{layer_indices[0] + 1:02d}-{layer_indices[-1] + 1:02d}",
            ):
                batch_indices = calibration_indices[start : start + batch_size]
                images = load_images(batch_indices, image_paths, preprocess, device)
                model.encode_image(images)
                token_count += len(batch_indices) * PATCH_TOKENS
                for layer in layer_indices:
                    capture = captures[layer]
                    validate_capture(capture, len(batch_indices))
                    block = model.visual.transformer.resblocks[layer]
                    hidden = block.mlp.gelu(capture.float())[:, 1:].reshape(-1, FFN_DIM)
                    projection = block.mlp.c_proj.weight.detach().float()
                    projection_norm_squared = projection.square().sum(dim=0)
                    scores[layer] += (
                        (hidden.square() * projection_norm_squared).sum(dim=0).double()
                    )
    finally:
        for hook in hooks:
            hook.remove()
    expected = len(calibration_indices) * PATCH_TOKENS
    if token_count != expected:
        raise AssertionError(f"calibration tokens={token_count}; expected {expected}")
    rankings = {
        layer: torch.argsort(scores[layer] / token_count, descending=True).cpu()
        for layer in layer_indices
    }
    return rankings, {layer: scores[layer].cpu() for layer in layer_indices}


def measure_batch_requirements(
    hidden: Tensor,
    projection: Tensor,
    bias: Tensor,
    order: Tensor,
    cosine_threshold: float,
    nre_threshold: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    if hidden.ndim != 2 or projection.ndim != 2 or bias.ndim != 1:
        raise ValueError("hidden, projection, and bias must be matrices, matrix, vector")
    ffn_dim = hidden.shape[1]
    if projection.shape[1] != ffn_dim or projection.shape[0] != bias.shape[0]:
        raise ValueError("projection and bias dimensions do not match hidden states")
    if order.shape != (ffn_dim,) or not torch.equal(
        torch.sort(order.detach().cpu()).values,
        torch.arange(ffn_dim),
    ):
        raise ValueError("channel order must be a permutation of every FFN channel")
    level_count = len(CAPACITY_LEVELS)
    dense_output = hidden @ projection.T + bias
    dense_norm = dense_output.norm(dim=-1).clamp_min(1e-8)
    assignments = torch.full(
        (len(hidden),),
        level_count - 1,
        dtype=torch.long,
        device=hidden.device,
    )
    assigned = torch.zeros(len(hidden), dtype=torch.bool, device=hidden.device)
    cosine_pass = torch.zeros(level_count, dtype=torch.int64)
    nre_pass = torch.zeros(level_count, dtype=torch.int64)
    joint_pass = torch.zeros(level_count, dtype=torch.int64)
    cosine_sum = torch.zeros(level_count, dtype=torch.float64)
    nre_sum = torch.zeros(level_count, dtype=torch.float64)
    order = order.to(hidden.device)
    sparse_output = bias.unsqueeze(0).expand(len(hidden), -1).clone()
    previous_width = 0
    for level_index, level in enumerate(CAPACITY_LEVELS[:-1]):
        width = round(level * ffn_dim)
        selected = order[previous_width:width]
        sparse_output += hidden[:, selected] @ projection[:, selected].T
        previous_width = width
        cosine = F.cosine_similarity(sparse_output, dense_output, dim=-1)
        nre = (sparse_output - dense_output).norm(dim=-1) / dense_norm
        passes_cosine = cosine >= cosine_threshold
        passes_nre = nre <= nre_threshold
        passes_joint = passes_cosine & passes_nre
        newly_assigned = (~assigned) & passes_joint
        assignments[newly_assigned] = level_index
        assigned |= passes_joint
        cosine_pass[level_index] = int(passes_cosine.sum().cpu())
        nre_pass[level_index] = int(passes_nre.sum().cpu())
        joint_pass[level_index] = int(passes_joint.sum().cpu())
        cosine_sum[level_index] = float(cosine.double().sum().cpu())
        nre_sum[level_index] = float(nre.double().sum().cpu())
    final = level_count - 1
    tokens = len(hidden)
    cosine_pass[final] = tokens
    nre_pass[final] = tokens
    joint_pass[final] = tokens
    cosine_sum[final] = tokens
    return assignments.cpu(), {
        "cosine_pass_counts": cosine_pass,
        "nre_pass_counts": nre_pass,
        "joint_pass_counts": joint_pass,
        "cosine_sum": cosine_sum,
        "nre_sum": nre_sum,
    }


def evaluate_requirements(
    model: Any,
    preprocess: Any,
    image_paths: Sequence[Path],
    evaluation_indices: Sequence[int],
    layer_indices: Sequence[int],
    rankings: dict[int, Tensor],
    batch_size: int,
    cosine_threshold: float,
    nre_threshold: float,
    device: torch.device,
) -> dict[int, dict[str, Any]]:
    captures, hooks = register_captures(model, layer_indices)
    level_count = len(CAPACITY_LEVELS)
    counts = {layer: torch.zeros(level_count, dtype=torch.int64) for layer in layer_indices}
    cosine_pass = {
        layer: torch.zeros(level_count, dtype=torch.int64) for layer in layer_indices
    }
    nre_pass = {layer: torch.zeros(level_count, dtype=torch.int64) for layer in layer_indices}
    joint_pass = {layer: torch.zeros(level_count, dtype=torch.int64) for layer in layer_indices}
    cosine_sum = {
        layer: torch.zeros(level_count, dtype=torch.float64) for layer in layer_indices
    }
    nre_sum = {layer: torch.zeros(level_count, dtype=torch.float64) for layer in layer_indices}
    try:
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(evaluation_indices), batch_size),
                desc=f"Evaluate L{layer_indices[0] + 1:02d}-{layer_indices[-1] + 1:02d}",
            ):
                batch_indices = evaluation_indices[start : start + batch_size]
                images = load_images(batch_indices, image_paths, preprocess, device)
                model.encode_image(images)
                for layer in layer_indices:
                    block = model.visual.transformer.resblocks[layer]
                    capture = captures[layer]
                    validate_capture(capture, len(batch_indices))
                    hidden = block.mlp.gelu(capture.float())[:, 1:].reshape(-1, FFN_DIM)
                    projection = block.mlp.c_proj.weight.detach().float()
                    bias = block.mlp.c_proj.bias.detach().float()
                    order = rankings[layer].to(device)
                    assignments, statistics = measure_batch_requirements(
                        hidden,
                        projection,
                        bias,
                        order,
                        cosine_threshold,
                        nre_threshold,
                    )
                    cosine_pass[layer] += statistics["cosine_pass_counts"]
                    nre_pass[layer] += statistics["nre_pass_counts"]
                    joint_pass[layer] += statistics["joint_pass_counts"]
                    cosine_sum[layer] += statistics["cosine_sum"]
                    nre_sum[layer] += statistics["nre_sum"]
                    counts[layer] += torch.bincount(
                        assignments,
                        minlength=level_count,
                    )
    finally:
        for hook in hooks:
            hook.remove()

    expected = len(evaluation_indices) * PATCH_TOKENS
    results: dict[int, dict[str, Any]] = {}
    for layer in layer_indices:
        observed = int(counts[layer].sum())
        if observed != expected:
            raise AssertionError(
                f"layer {layer + 1}: observations={observed}; expected {expected}"
            )
        layer_counts = counts[layer].tolist()
        results[layer] = {
            "observations": observed,
            "required_capacity_counts": layer_counts,
            "required_capacity_proportions": (counts[layer].double() / observed).tolist(),
            "layer_mean": float(
                sum(level * count for level, count in zip(CAPACITY_LEVELS, layer_counts))
                / observed
            ),
            "cosine_pass_counts": cosine_pass[layer].tolist(),
            "nre_pass_counts": nre_pass[layer].tolist(),
            "joint_pass_counts": joint_pass[layer].tolist(),
            "mean_cosine_by_capacity": (cosine_sum[layer] / observed).tolist(),
            "mean_nre_by_capacity": (nre_sum[layer] / observed).tolist(),
        }
    return results


def build_payload(
    args: argparse.Namespace,
    image_paths: Sequence[Path],
    calibration_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    ranking_path: Path,
    layer_results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL if not args.allow_partial_smoke else f"{PROTOCOL}_smoke",
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "backbone": "frozen_dense_clip",
        "pretrained_sha256": PRETRAINED_SHA256,
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "seed": args.seed,
        "split_method": SPLIT_METHOD,
        "calibration_images": len(calibration_indices),
        "evaluation_images": len(evaluation_indices),
        "calibration_split_sha256": split_identity(calibration_indices, image_paths),
        "evaluation_split_sha256": split_identity(evaluation_indices, image_paths),
        "split_disjoint": True,
        "patch_tokens_per_image": PATCH_TOKENS,
        "ffn_dim": FFN_DIM,
        "capacity_levels": list(CAPACITY_LEVELS),
        "cosine_threshold": args.cosine_threshold,
        "nre_threshold": args.nre_threshold,
        "channel_ranking": CHANNEL_RANKING,
        "comparison": COMPARISON,
        "batch_first_verified": True,
        "tf32": False,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "ranking_file": str(ranking_path.resolve()),
        "layers": {
            str(layer + 1): layer_results[layer]
            for layer in range(args.layer_start - 1, args.layer_end)
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    image_paths, calibration_indices, evaluation_indices = validate_args(args)
    if args.check_only:
        result = {
            **protocol_manifest(),
            "pretrained": str(args.pretrained.resolve()),
            "pretrained_sha256": PRETRAINED_SHA256,
            "coco_annotations": str(args.coco_annotations.resolve()),
            "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
            "coco_images": str(args.coco_images.resolve()),
            "calibration_split_sha256": split_identity(calibration_indices, image_paths),
            "evaluation_split_sha256": split_identity(evaluation_indices, image_paths),
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
            "output_dir": str(args.output_dir.resolve()),
        }
        print(json.dumps(result, indent=2))
        return result
    device = resolve_device(args.device)
    set_reproducible_seed(args.seed)
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("open_clip_torch is required for Figure 3") from error
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(args.pretrained),
        force_quick_gelu=True,
    )
    model = model.to(device).eval()
    if not model.visual.transformer.batch_first:
        raise RuntimeError("Figure 3 requires batch-first OpenCLIP tensors")
    layer_indices = list(range(args.layer_start - 1, args.layer_end))
    rankings, ranking_scores = calibrate_rankings(
        model,
        preprocess,
        image_paths,
        calibration_indices,
        layer_indices,
        args.batch_size,
        device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"layers_{args.layer_start:02d}_{args.layer_end:02d}"
    ranking_path = args.output_dir / f"{stem}_rankings.pt"
    torch.save(
        {
            "format_version": 1,
            "study": STUDY_NAME,
            "protocol": PROTOCOL,
            "seed": args.seed,
            "layers": [layer + 1 for layer in layer_indices],
            "rankings": {layer + 1: rankings[layer] for layer in layer_indices},
            "scores": {layer + 1: ranking_scores[layer] for layer in layer_indices},
        },
        ranking_path,
    )
    layer_results = evaluate_requirements(
        model,
        preprocess,
        image_paths,
        evaluation_indices,
        layer_indices,
        rankings,
        args.batch_size,
        args.cosine_threshold,
        args.nre_threshold,
        device,
    )
    payload = build_payload(
        args,
        image_paths,
        calibration_indices,
        evaluation_indices,
        ranking_path,
        layer_results,
    )
    result_path = args.output_dir / f"{stem}_result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"rankings={ranking_path}")
    print(f"result={result_path}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))

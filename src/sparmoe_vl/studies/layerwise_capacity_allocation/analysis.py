"""Collect layer-wise expert usage from the visual main Stage-2 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
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
    COCO_ANNOTATIONS,
    COCO_ANNOTATIONS_SHA256,
    COCO_IMAGES,
    COCO_IMAGES_TOTAL,
    COCO_MANIFEST_SHA256,
    CPU_THREADS,
    DATA_SEED,
    HISTORICAL_CHECKPOINT_SHA256,
    MODEL_KEY,
    MODEL_NAME,
    NUM_EXPERTS,
    NUM_LAYERS,
    NUM_WORKERS,
    OUTPUT_ROOT,
    PAPER_SCOPE,
    PATCHES_PER_IMAGE,
    PRETRAINED,
    PRETRAINED_SHA256,
    PROTOCOL,
    ROUTING_MODE,
    RUN_SEED,
    STUDY_NAME,
    TARGET_RATIO,
    TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
    protocol_manifest,
)


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
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "analysis.json")
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
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("images"), list):
        raise ValueError("COCO annotations must contain an images list")
    records = payload["images"]
    if not all(
        isinstance(record, Mapping) and "id" in record and "file_name" in record
        for record in records
    ):
        raise ValueError("COCO image records must contain id and file_name")
    records = sorted(records, key=lambda record: int(record["id"]))
    ids = [int(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("COCO image identifiers must be unique")
    return tuple(image_root / str(record["file_name"]) for record in records)


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


@torch.inference_mode()
def forward_and_count(model: SparMoEVisionEncoder, images: Tensor) -> Tensor:
    """Run the exact Stage-2 sparse path while retaining only route counts."""

    visual = model.clip_model.visual
    x = visual._embeds(images.to(dtype=model.backbone_dtype))
    batch_first = bool(visual.transformer.batch_first)
    if not batch_first:
        x = x.transpose(0, 1).contiguous()
    retention_ratios = model.budget()
    patterns = model.sparse_pattern_generator.all_layers(retention_ratios)
    sparse_positions = {
        layer_index: position for position, layer_index in enumerate(model.sparse_layers)
    }
    counts = torch.zeros(
        NUM_LAYERS,
        NUM_EXPERTS,
        dtype=torch.int64,
        device=x.device,
    )
    for layer_index, block in enumerate(visual.transformer.resblocks):
        sparse_position = sparse_positions.get(layer_index)
        if sparse_position is None:
            x = block(x, attn_mask=None)
            continue
        x = x + block.ls_1(block.attention(q_x=block.ln_1(x), attn_mask=None))
        normalized = block.ln_2(x)
        normalized_batch_first = normalized if batch_first else normalized.transpose(0, 1)
        batch_size, token_count, dimension = normalized_batch_first.shape
        patch = normalized_batch_first[:, 1:].reshape(-1, dimension)
        router = model.routers[sparse_position]
        logits = router.projection(patch.to(dtype=router.projection.weight.dtype))
        expert_ids = logits.argmax(dim=-1)
        counts[sparse_position] = torch.bincount(
            expert_ids,
            minlength=NUM_EXPERTS,
        )

        weight_1 = block.mlp.c_fc.weight
        bias_1 = block.mlp.c_fc.bias
        weight_2 = block.mlp.c_proj.weight
        bias_2 = block.mlp.c_proj.bias
        activation = block.mlp.gelu
        class_output = (
            activation(normalized_batch_first[:, :1] @ weight_1.T + bias_1) @ weight_2.T
            + bias_2
        )
        hidden = activation(patch @ weight_1.T + bias_1)
        intermediate_norm = getattr(block.mlp, "ln", None)
        if intermediate_norm is not None:
            hidden = intermediate_norm(hidden)
        masks = patterns[sparse_position].masks
        patch_output = (hidden * masks[expert_ids].to(hidden.dtype)) @ weight_2.T + bias_2
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
    return counts.cpu()


def partial_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.partial.npz")


def partial_manifest(
    dataset_manifest: str,
    checkpoint_sha256: str,
    image_count: int,
) -> str:
    return sequence_sha256([PROTOCOL, dataset_manifest, checkpoint_sha256, str(image_count)])


def save_partial(
    path: Path,
    counts: Tensor,
    processed: int,
    manifest: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            counts=counts.numpy(),
            processed=np.asarray(processed, dtype=np.int64),
            manifest=np.asarray(manifest),
        )
    temporary.replace(path)


def restore_partial(path: Path, manifest: str, image_count: int) -> tuple[Tensor, int]:
    if not path.is_file():
        return torch.zeros(NUM_LAYERS, NUM_EXPERTS, dtype=torch.int64), 0
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["manifest"].item()) != manifest:
            raise ValueError("partial routing count belongs to a different protocol")
        counts = torch.from_numpy(cached["counts"].astype(np.int64, copy=True))
        processed = int(cached["processed"].item())
    if counts.shape != (NUM_LAYERS, NUM_EXPERTS):
        raise ValueError(f"partial routing count has invalid shape {tuple(counts.shape)}")
    if not 0 <= processed <= image_count:
        raise ValueError("partial routing count has an invalid processed image count")
    expected = processed * PATCHES_PER_IMAGE
    if not torch.equal(counts.sum(dim=1), torch.full((NUM_LAYERS,), expected)):
        raise ValueError("partial routing count does not conserve processed patch tokens")
    return counts, processed


def collect_counts(
    model: SparMoEVisionEncoder,
    preprocess: Any,
    paths: Sequence[Path],
    args: argparse.Namespace,
    manifest: str,
) -> tuple[Tensor, float]:
    state_path = partial_path(args.output)
    counts, processed = (
        (torch.zeros(NUM_LAYERS, NUM_EXPERTS, dtype=torch.int64), 0)
        if args.restart
        else restore_partial(state_path, manifest, len(paths))
    )
    dataset = StrictImageDataset(paths, preprocess)
    loader = DataLoader(
        Subset(dataset, range(processed, len(dataset))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    started = time.perf_counter()
    for batch_index, images in enumerate(
        tqdm(loader, desc="COCO patch-token routing", unit="batch"),
        start=1,
    ):
        images = images.to(args.device, non_blocking=str(args.device).startswith("cuda"))
        counts += forward_and_count(model, images)
        processed += int(images.shape[0])
        if batch_index % args.checkpoint_every == 0:
            save_partial(state_path, counts, processed, manifest)
    elapsed = time.perf_counter() - started
    save_partial(state_path, counts, processed, manifest)
    expected = len(paths) * PATCHES_PER_IMAGE
    if processed != len(paths):
        raise RuntimeError(f"processed {processed} images; expected {len(paths)}")
    if not torch.equal(counts.sum(dim=1), torch.full((NUM_LAYERS,), expected)):
        raise RuntimeError("final routing counts do not conserve patch tokens")
    return counts, elapsed


def capacity_statistics(
    base_ratios: np.ndarray,
    usage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    levels = np.asarray(CAPACITY_FACTORS, dtype=np.float64)
    if base_ratios.shape != (NUM_LAYERS,):
        raise ValueError(f"base ratios must have shape ({NUM_LAYERS},)")
    if usage.shape != (NUM_LAYERS, NUM_EXPERTS):
        raise ValueError(f"usage must have shape ({NUM_LAYERS}, {NUM_EXPERTS})")
    capacities = np.clip(base_ratios[:, None] * levels[None, :], 0.01, 0.995)
    activated = np.sum(usage * capacities, axis=1)
    return capacities, activated


def build_payload(
    paths: Sequence[Path],
    metadata: Mapping[str, Any],
    counts: Tensor,
    base_ratios: Tensor,
    elapsed: float,
    partial_smoke: bool,
) -> dict[str, Any]:
    proportions = counts.double() / counts.sum(dim=1, keepdim=True).double()
    base = base_ratios.detach().double().cpu().numpy()
    usage = proportions.numpy()
    capacities, activated = capacity_statistics(base, usage)
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL if not partial_smoke else f"{PROTOCOL}_smoke",
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "checkpoint_metadata": dict(metadata),
        "routing": ROUTING_MODE,
        "dataset": "COCO-val2017",
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "dataset_manifest_sha256": sequence_sha256([path.name for path in paths]),
        "image_count": len(paths),
        "patches_per_image": PATCHES_PER_IMAGE,
        "tokens_per_layer": len(paths) * PATCHES_PER_IMAGE,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "target_ratio": TARGET_RATIO,
        "capacity_factors": list(CAPACITY_FACTORS),
        "counts_layer_by_expert": counts.tolist(),
        "proportions_layer_by_expert": usage.tolist(),
        "base_ratios_by_layer": base.tolist(),
        "capacities_layer_by_expert": capacities.tolist(),
        "activated_capacity_by_layer": activated.tolist(),
        "elapsed_seconds": elapsed,
        "tf32": False,
    }


def validate_analysis(payload: Mapping[str, Any], source: str | Path) -> None:
    expected = {
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "routing": ROUTING_MODE,
        "dataset": "COCO-val2017",
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "dataset_manifest_sha256": COCO_MANIFEST_SHA256,
        "image_count": COCO_IMAGES_TOTAL,
        "patches_per_image": PATCHES_PER_IMAGE,
        "tokens_per_layer": COCO_IMAGES_TOTAL * PATCHES_PER_IMAGE,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "target_ratio": TARGET_RATIO,
        "capacity_factors": list(CAPACITY_FACTORS),
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
    counts = np.asarray(payload.get("counts_layer_by_expert"))
    proportions = np.asarray(payload.get("proportions_layer_by_expert"), dtype=float)
    base = np.asarray(payload.get("base_ratios_by_layer"), dtype=float)
    capacities = np.asarray(payload.get("capacities_layer_by_expert"), dtype=float)
    activated = np.asarray(payload.get("activated_capacity_by_layer"), dtype=float)
    if counts.shape != (NUM_LAYERS, NUM_EXPERTS) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError(f"{source}: invalid expert-count matrix")
    if np.any(counts < 0) or not np.all(counts.sum(axis=1) == expected["tokens_per_layer"]):
        raise ValueError(f"{source}: expert counts do not conserve patch tokens")
    if proportions.shape != counts.shape or not np.allclose(
        proportions,
        counts / expected["tokens_per_layer"],
        atol=1e-12,
    ):
        raise ValueError(f"{source}: expert proportions disagree with counts")
    expected_capacities, expected_activated = capacity_statistics(base, proportions)
    if capacities.shape != counts.shape or not np.allclose(
        capacities, expected_capacities, atol=1e-12
    ):
        raise ValueError(f"{source}: expert capacities disagree with layer budgets")
    if activated.shape != (NUM_LAYERS,) or not np.allclose(
        activated, expected_activated, atol=1e-12
    ):
        raise ValueError(f"{source}: activated capacities disagree with expert usage")


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths, metadata = validate_args(args)
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
        "dataset_manifest_sha256": sequence_sha256([path.name for path in paths]),
        "output": str(args.output.resolve()),
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
        args.checkpoint,
        args.pretrained,
        device,
    )
    if loaded_metadata["checkpoint_step"] != metadata["checkpoint_step"]:
        raise RuntimeError("checkpoint changed between validation and model loading")
    state_manifest = partial_manifest(
        check["dataset_manifest_sha256"],
        metadata["checkpoint_sha256"],
        len(paths),
    )
    counts, elapsed = collect_counts(
        model,
        preprocess,
        paths,
        args,
        state_manifest,
    )
    payload = build_payload(
        paths,
        metadata,
        counts,
        model.budget.base_ratios(),
        elapsed,
        args.allow_partial_smoke,
    )
    if not args.allow_partial_smoke:
        validate_analysis(payload, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    partial_path(args.output).unlink(missing_ok=True)
    print(f"analysis={args.output}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))

"""Evaluate learned routing against its capacity-matched shuffled control."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ...architecture_transfer.clip.model import (
    CLIPSparMoE,
    build_model,
    load_stage2_controller,
)
from ...common.training import set_reproducible_seed
from .checkpoints import checkpoint_metadata, file_sha256, torch_load
from .protocol import (
    COCO_ANNOTATIONS,
    COCO_ANNOTATIONS_SHA256,
    COCO_CAPTION_COUNT,
    COCO_CAPTION_IMAGE_INDEX_SHA256,
    COCO_CAPTION_ORDER_SHA256,
    COCO_IMAGE_COUNT,
    COCO_IMAGE_ORDER_SHA256,
    COCO_IMAGES,
    DENSE_FFN_MACS_G,
    DENSE_TOTAL_MACS_G,
    EXPERT_COUNTS,
    FFN_DIM,
    IMAGE_BATCH_SIZE,
    MODEL_DIM,
    MODEL_KEY,
    MODEL_NAME,
    MODALITY,
    NUM_WORKERS,
    OUTPUT_ROOT,
    PATCH_TOKENS,
    PRETRAINED,
    PRETRAINED_SHA256,
    PROTOCOL,
    RUN_SEED,
    STUDY_NAME,
    TEXT_BATCH_SIZE,
    VISION_LAYERS,
    capacity_factors,
    checkpoint_path,
    protocol_manifest,
)


@dataclass(frozen=True)
class PaperCOCO:
    image_paths: tuple[Path, ...]
    captions: tuple[str, ...]
    caption_image_indices: tuple[int, ...]


class StrictImageDataset(Dataset[Tensor]):
    """Decode every registered COCO image or fail with its exact path."""

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
            raise RuntimeError(f"failed to decode registered COCO image: {path}") from error


def sequence_sha256(values: Sequence[str]) -> str:
    """Hash an ordered manifest using the original experiment's NUL delimiter."""

    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_paper_coco(annotations: Path, image_root: Path) -> PaperCOCO:
    """Recreate the exact image/caption ordering used by the paper evaluator."""

    payload = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("COCO annotations must contain a JSON object")
    images = payload.get("images")
    annotations_list = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations_list, list):
        raise ValueError("COCO annotations must expose images and annotations lists")
    selected = images[:COCO_IMAGE_COUNT]
    if len(selected) != COCO_IMAGE_COUNT:
        raise ValueError(f"COCO contains {len(selected)} registered images")
    image_ids: dict[int, int] = {}
    file_names: list[str] = []
    for index, item in enumerate(selected):
        if not isinstance(item, Mapping) or "id" not in item or "file_name" not in item:
            raise ValueError(f"COCO image record {index} is malformed")
        image_id = int(item["id"])
        if image_id in image_ids:
            raise ValueError(f"duplicate COCO image id: {image_id}")
        image_ids[image_id] = index
        file_names.append(str(item["file_name"]))

    captions: list[str] = []
    caption_image_indices: list[int] = []
    for item in annotations_list:
        if not isinstance(item, Mapping):
            continue
        image_id = int(item.get("image_id", -1))
        caption = item.get("caption")
        if image_id in image_ids and isinstance(caption, str):
            captions.append(caption)
            caption_image_indices.append(image_ids[image_id])
    if len(captions) != COCO_CAPTION_COUNT:
        raise ValueError(
            f"COCO contains {len(captions)} captions for the registered images; "
            f"expected {COCO_CAPTION_COUNT}"
        )

    identities = {
        "image order": (sequence_sha256(file_names), COCO_IMAGE_ORDER_SHA256),
        "caption order": (sequence_sha256(captions), COCO_CAPTION_ORDER_SHA256),
        "caption-image mapping": (
            sequence_sha256([str(value) for value in caption_image_indices]),
            COCO_CAPTION_IMAGE_INDEX_SHA256,
        ),
    }
    for label, (observed, expected) in identities.items():
        if observed != expected:
            raise ValueError(f"COCO {label} differs from the result-generating protocol")
    paths = tuple(image_root / file_name for file_name in file_names)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing registered COCO image: {missing[0]}")
    return PaperCOCO(paths, tuple(captions), tuple(caption_image_indices))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert-counts",
        type=int,
        nargs="+",
        choices=EXPERT_COUNTS,
        default=EXPERT_COUNTS,
    )
    for count in EXPERT_COUNTS:
        parser.add_argument(
            f"--checkpoint-n{count}",
            type=Path,
            default=checkpoint_path(count),
        )
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--coco-annotations", type=Path, default=COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=COCO_IMAGES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "analysis.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    args.expert_counts = tuple(dict.fromkeys(args.expert_counts))
    args.checkpoints = {
        count: getattr(args, f"checkpoint_n{count}") for count in args.expert_counts
    }
    return args


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[PaperCOCO, dict[int, dict[str, Any]]]:
    for path, label in (
        (args.pretrained, "Dense CLIP ViT-L/14 weights"),
        (args.coco_annotations, "COCO val2017 annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not args.coco_images.is_dir():
        raise FileNotFoundError(f"missing COCO val2017 image root: {args.coco_images}")
    if file_sha256(args.pretrained) != PRETRAINED_SHA256:
        raise ValueError("Dense CLIP weights differ from the result-generating run")
    if file_sha256(args.coco_annotations) != COCO_ANNOTATIONS_SHA256:
        raise ValueError("COCO annotations differ from the result-generating run")
    corpus = load_paper_coco(args.coco_annotations, args.coco_images)
    metadata = {}
    for count, path in args.checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing N={count} checkpoint: {path}")
        digest = file_sha256(path)
        metadata[count] = checkpoint_metadata(
            torch_load(path),
            count,
            checkpoint_sha256=digest,
        )
        metadata[count]["checkpoint"] = str(path.resolve())
    return corpus, metadata


def make_image_loader(
    corpus: PaperCOCO, preprocess: Any, device: torch.device
) -> DataLoader[Tensor]:
    return DataLoader(
        StrictImageDataset(corpus.image_paths, preprocess),
        batch_size=IMAGE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )


@torch.inference_mode()
def encode_texts(
    clip_model: Any,
    tokenizer: Any,
    captions: Sequence[str],
    device: torch.device,
) -> Tensor:
    features = []
    for start in tqdm(range(0, len(captions), TEXT_BATCH_SIZE), desc="Dense text"):
        tokens = tokenizer(captions[start : start + TEXT_BATCH_SIZE]).to(device)
        encoded = clip_model.encode_text(tokens)
        features.append(F.normalize(encoded.float(), dim=-1).cpu())
    return torch.cat(features)


@torch.inference_mode()
def encode_dense_images(
    clip_model: Any, loader: DataLoader[Tensor], device: torch.device
) -> Tensor:
    features = []
    for images in tqdm(loader, desc="Dense images"):
        encoded = clip_model.encode_image(images.to(device, non_blocking=device.type == "cuda"))
        features.append(F.normalize(encoded.float(), dim=-1).cpu())
    return torch.cat(features)


def anti_matched_assignments(ids: Tensor) -> Tensor:
    """Reverse the learned token-to-capacity relation while preserving counts."""

    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("expert ids must be a non-empty vector")
    token_order = torch.argsort(ids, stable=True)
    capacity_pool = torch.sort(ids, descending=True).values
    reassigned = torch.empty_like(ids)
    reassigned[token_order] = capacity_pool
    return reassigned


def _route_statistics(
    expert_counts: Tensor,
    capacity_sum: Tensor,
    token_counts: Tensor,
    mask_widths: Tensor,
) -> dict[str, Any]:
    layer_patch_widths = capacity_sum / token_counts.clamp_min(1)
    layer_usage = expert_counts / expert_counts.sum(dim=1, keepdim=True).clamp_min(1)
    return {
        "mean_patch_width": float(layer_patch_widths.mean()),
        "layer_patch_widths": layer_patch_widths.tolist(),
        "layer_usage": layer_usage.tolist(),
        "mask_widths": mask_widths.tolist(),
    }


@torch.inference_mode()
def encode_paired_routes(
    model: CLIPSparMoE,
    loader: DataLoader[Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, Any], dict[str, Any], dict[str, int]]:
    """Run learned and forced shuffled routes on each identical inference batch."""

    expert_count = len(model.levels)
    learned_features = []
    shuffled_features = []
    learned_counts = torch.zeros((VISION_LAYERS, expert_count), dtype=torch.float64)
    shuffled_counts = torch.zeros_like(learned_counts)
    learned_capacity = torch.zeros(VISION_LAYERS, dtype=torch.float64)
    shuffled_capacity = torch.zeros_like(learned_capacity)
    token_counts = torch.zeros(VISION_LAYERS, dtype=torch.float64)
    mask_widths: Tensor | None = None
    checks = 0

    for images in tqdm(loader, desc=f"N={expert_count} paired routing"):
        images = images.to(device, non_blocking=device.type == "cuda")
        model.set_routing_mode("learned")
        learned, learned_aux = model.encode_sparse(images)
        learned_features.append(learned.float().cpu())

        replay = []
        for item in learned_aux:
            replay.append(anti_matched_assignments(item["G"].argmax(-1)))
        for layer, ids in zip(model.layers.values(), replay):
            layer.forced_expert_ids = ids
        try:
            model.set_routing_mode("forced")
            shuffled, shuffled_aux = model.encode_sparse(images)
        finally:
            for layer in model.layers.values():
                layer.forced_expert_ids = None
        shuffled_features.append(shuffled.float().cpu())

        if len(learned_aux) != VISION_LAYERS or len(shuffled_aux) != VISION_LAYERS:
            raise RuntimeError("the sparse model did not expose all 24 visual layers")
        current_widths = torch.stack(
            [item["masks"].detach().float().mean(-1).cpu() for item in learned_aux]
        )
        if mask_widths is None:
            mask_widths = current_widths
        elif not torch.equal(mask_widths, current_widths):
            raise RuntimeError("expert mask widths changed between inference batches")

        for layer_index, (learned_item, shuffled_item) in enumerate(
            zip(learned_aux, shuffled_aux)
        ):
            learned_ids = learned_item["G"].argmax(-1)
            shuffled_ids = shuffled_item["G"].argmax(-1)
            expected_ids = replay[layer_index].to(shuffled_ids.device)
            if not torch.equal(shuffled_ids, expected_ids):
                raise RuntimeError("forced shuffled assignments were not replayed exactly")
            learned_hist = torch.bincount(learned_ids, minlength=expert_count)
            shuffled_hist = torch.bincount(shuffled_ids, minlength=expert_count)
            if not torch.equal(learned_hist, shuffled_hist):
                raise RuntimeError("capacity-matched control changed an expert-count multiset")
            learned_widths = learned_item["masks"].float().mean(-1)
            shuffled_widths = shuffled_item["masks"].float().mean(-1)
            if not torch.equal(learned_widths, shuffled_widths):
                raise RuntimeError("capacity-matched control changed expert widths")
            learned_counts[layer_index] += learned_hist.double().cpu()
            shuffled_counts[layer_index] += shuffled_hist.double().cpu()
            learned_capacity[layer_index] += float(
                (learned_hist * learned_widths).sum().double().cpu()
            )
            shuffled_capacity[layer_index] += float(
                (shuffled_hist * shuffled_widths).sum().double().cpu()
            )
            token_counts[layer_index] += learned_ids.numel()
            checks += 1

    if mask_widths is None:
        raise RuntimeError("no COCO images were evaluated")
    learned_stats = _route_statistics(
        learned_counts, learned_capacity, token_counts, mask_widths
    )
    shuffled_stats = _route_statistics(
        shuffled_counts, shuffled_capacity, token_counts, mask_widths
    )
    if learned_stats["layer_patch_widths"] != shuffled_stats["layer_patch_widths"]:
        raise RuntimeError("paired routes did not preserve per-layer activated capacity")
    verification = {
        "per_layer_per_batch_multiset_checks": checks,
        "matched_checks": checks,
    }
    return (
        torch.cat(learned_features),
        torch.cat(shuffled_features),
        learned_stats,
        shuffled_stats,
        verification,
    )


def vision_macs(layer_ratios: Sequence[float]) -> dict[str, float]:
    """Apply the exact ViT-L/14 MAC convention used by the paper evaluator."""

    if len(layer_ratios) != VISION_LAYERS:
        raise ValueError(f"expected {VISION_LAYERS} layer ratios")
    ffn = 0
    for ratio in layer_ratios:
        active = int(FFN_DIM * float(ratio))
        ffn += 2 * MODEL_DIM * FFN_DIM
        ffn += PATCH_TOKENS * 2 * MODEL_DIM * active
    ffn_g = ffn / 1e9
    return {
        "ffn": ffn_g,
        "total": DENSE_TOTAL_MACS_G - DENSE_FFN_MACS_G + ffn_g,
    }


@torch.inference_mode()
def recall_at_1_chunked(
    image_features: Tensor,
    text_features: Tensor,
    caption_image_indices: Tensor,
    device: torch.device,
    image_chunk: int = 256,
    text_chunk: int = 1_024,
) -> tuple[float, float]:
    image_device = image_features.to(device)
    text_device = text_features.to(device)
    targets_device = caption_image_indices.to(device)
    i2t_correct = 0
    for start in range(0, len(image_device), image_chunk):
        query = image_device[start : start + image_chunk]
        predictions = (query @ text_device.T).argmax(dim=1)
        predicted_images = targets_device[predictions]
        targets = torch.arange(start, start + len(query), device=device)
        i2t_correct += int((predicted_images == targets).sum())
    t2i_correct = 0
    for start in range(0, len(text_device), text_chunk):
        query = text_device[start : start + text_chunk]
        predictions = (image_device @ query.T).argmax(dim=0)
        targets = targets_device[start : start + len(query)]
        t2i_correct += int((predictions == targets).sum())
    return (
        100.0 * i2t_correct / len(image_device),
        100.0 * t2i_correct / len(text_device),
    )


def route_metrics(
    features: Tensor,
    statistics: dict[str, Any],
    dense_images: Tensor,
    dense_texts: Tensor,
    caption_image_indices: Tensor,
    dense_recall: tuple[float, float],
    device: torch.device,
) -> dict[str, Any]:
    i2t, t2i = recall_at_1_chunked(
        features,
        dense_texts,
        caption_image_indices,
        device,
    )
    dense_i2t, dense_t2i = dense_recall
    if dense_i2t <= 0 or dense_t2i <= 0:
        raise RuntimeError("Dense CLIP recall must be positive")
    macs = vision_macs(statistics["layer_patch_widths"])
    return {
        "dense_cosine": float(
            F.cosine_similarity(features.float(), dense_images.float(), dim=-1).mean()
        ),
        "coco_i2t_r1": i2t,
        "coco_t2i_r1": t2i,
        "mean_r1": 0.5 * (i2t + t2i),
        "mean_r1_retention": 50.0 * (i2t / dense_i2t + t2i / dense_t2i),
        "vision_ffn_macs_g": macs["ffn"],
        "vision_total_macs_g": macs["total"],
        **statistics,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_summary_csv(path: Path, result: Mapping[str, Any]) -> None:
    fields = (
        "num_experts",
        "routing",
        "dense_cosine",
        "coco_i2t_r1",
        "coco_t2i_r1",
        "mean_r1",
        "mean_r1_retention",
        "vision_ffn_macs_g",
        "vision_total_macs_g",
        "mean_patch_width",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        experiments = result["experiments"]
        for count in EXPERT_COUNTS:
            if str(count) not in experiments:
                continue
            for route in ("learned", "capacity_matched_shuffle"):
                values = experiments[str(count)][route]
                writer.writerow(
                    {
                        "num_experts": count,
                        "routing": route,
                        **{field: values[field] for field in fields[2:]},
                    }
                )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    corpus, checkpoint_metadata_by_count = validate_inputs(args)
    check_result = {
        **protocol_manifest(),
        "pretrained_sha256": PRETRAINED_SHA256,
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "coco_image_order_sha256": COCO_IMAGE_ORDER_SHA256,
        "coco_caption_order_sha256": COCO_CAPTION_ORDER_SHA256,
        "coco_caption_image_index_sha256": COCO_CAPTION_IMAGE_INDEX_SHA256,
        "checkpoints": {
            str(count): checkpoint_metadata_by_count[count] for count in args.expert_counts
        },
        "output": str(args.output.resolve()),
    }
    if args.check_only:
        print(json.dumps(check_result, indent=2))
        return check_result

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_reproducible_seed(RUN_SEED)
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("open_clip_torch is required for this experiment") from error
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(args.pretrained),
        force_quick_gelu=True,
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    image_loader = make_image_loader(corpus, preprocess, device)
    dense_texts = encode_texts(clip_model, tokenizer, corpus.captions, device)
    dense_images = encode_dense_images(clip_model, image_loader, device)
    caption_indices = torch.tensor(corpus.caption_image_indices, dtype=torch.long)
    dense_recall = recall_at_1_chunked(
        dense_images,
        dense_texts,
        caption_indices,
        device,
    )
    result: dict[str, Any] = {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "modality": MODALITY,
        "run_seed": RUN_SEED,
        "evaluation": {
            "dataset": "COCO-val2017",
            "images": len(corpus.image_paths),
            "captions": len(corpus.captions),
            "annotation_sha256": COCO_ANNOTATIONS_SHA256,
            "image_order_sha256": COCO_IMAGE_ORDER_SHA256,
            "caption_order_sha256": COCO_CAPTION_ORDER_SHA256,
            "caption_image_index_sha256": COCO_CAPTION_IMAGE_INDEX_SHA256,
            "image_batch_size": IMAGE_BATCH_SIZE,
            "text_batch_size": TEXT_BATCH_SIZE,
        },
        "control": {
            "name": "capacity_matched_shuffle",
            "definition": (
                "per-layer, per-batch anti-matched reassignment preserving the "
                "exact multiset of learned expert selections"
            ),
        },
        "dense": {
            "coco_i2t_r1": dense_recall[0],
            "coco_t2i_r1": dense_recall[1],
            "mean_r1": 0.5 * sum(dense_recall),
            "vision_ffn_macs_g": DENSE_FFN_MACS_G,
            "vision_total_macs_g": DENSE_TOTAL_MACS_G,
        },
        "experiments": {},
    }

    for count in args.expert_counts:
        checkpoint = torch_load(args.checkpoints[count])
        model = build_model(
            clip_model,
            modality=MODALITY,
            stage=2,
            target_ratio=0.7,
            levels=capacity_factors(count),
            tau=0.4,
        ).to(device)
        load_stage2_controller(model, checkpoint)
        del checkpoint
        model.eval()
        learned_features, shuffled_features, learned_stats, shuffled_stats, checks = (
            encode_paired_routes(model, image_loader, device)
        )
        learned = route_metrics(
            learned_features,
            learned_stats,
            dense_images,
            dense_texts,
            caption_indices,
            dense_recall,
            device,
        )
        shuffled = route_metrics(
            shuffled_features,
            shuffled_stats,
            dense_images,
            dense_texts,
            caption_indices,
            dense_recall,
            device,
        )
        mac_difference = abs(learned["vision_ffn_macs_g"] - shuffled["vision_ffn_macs_g"])
        if mac_difference > 1e-10:
            raise RuntimeError(
                f"capacity-matched control failed for N={count}: "
                f"FFN MAC difference={mac_difference} G"
            )
        result["experiments"][str(count)] = {
            "checkpoint": checkpoint_metadata_by_count[count],
            "capacity_factors": list(capacity_factors(count)),
            "capacity_match_verification": {
                **checks,
                "ffn_macs_absolute_difference_g": mac_difference,
            },
            "learned": learned,
            "capacity_matched_shuffle": shuffled,
            "routing_gain": {
                "dense_cosine": learned["dense_cosine"] - shuffled["dense_cosine"],
                "mean_r1_retention_pp": (
                    learned["mean_r1_retention"] - shuffled["mean_r1_retention"]
                ),
            },
        }
        atomic_write_json(args.output, result)
        write_summary_csv(args.output.with_name("summary.csv"), result)
        del model, learned_features, shuffled_features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    atomic_write_json(args.output, result)
    write_summary_csv(args.output.with_name("summary.csv"), result)
    print(
        json.dumps(
            {"analysis": str(args.output), "completed": list(result["experiments"])}, indent=2
        )
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))

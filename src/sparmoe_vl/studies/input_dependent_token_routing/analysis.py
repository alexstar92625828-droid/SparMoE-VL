"""Extract the exact vision and text routes used by the paper visualization."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from ..generalization.protocol import (
    inspect_checkpoint as inspect_main_checkpoint,
    load_text_encoder,
)
from ..vision_budget_sweep import (
    inspect_checkpoint as inspect_vision_checkpoint,
    load_encoder as load_vision_encoder,
)
from .protocol import (
    CAPACITY_FACTORS,
    DATA_SEED,
    HISTORICAL_TEXT_CHECKPOINT_SHA256,
    HISTORICAL_VISION_CHECKPOINT_SHA256,
    IMAGE_IDENTITIES,
    IMAGE_ROOT,
    MODEL_NAME,
    NUM_EXPERTS,
    OUTPUT_ROOT,
    PAPER_SCOPE,
    PATCHES_PER_IMAGE,
    PATCH_GRID_SIZE,
    PRETRAINED,
    PRETRAINED_SHA256,
    PROTOCOL,
    ROUTING_MODE,
    RUN_SEED,
    STUDY_NAME,
    TEXT_CHECKPOINT,
    TEXT_INPUT,
    TEXT_INPUT_SHA256,
    TEXT_LAYER,
    TEXT_TARGET_RATIO,
    TEXT_TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
    VISION_CHECKPOINT,
    VISION_LAYERS,
    VISION_TARGET_RATIO,
    VISION_TRAINING_POOL_SHA256,
    protocol_manifest,
)


PUNCTUATION = frozenset({".", ",", ";", ":", "!", "?", "(", ")", "[", "]"})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-checkpoint", type=Path, default=VISION_CHECKPOINT)
    parser.add_argument("--text-checkpoint", type=Path, default=TEXT_CHECKPOINT)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=min(32, os.cpu_count() or 1),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "analysis.json")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def validate_checkpoint(path: Path, modality: str) -> dict[str, Any]:
    if modality == "vision":
        metadata = inspect_vision_checkpoint(path)
        expected_ratio = VISION_TARGET_RATIO
        expected_pool = VISION_TRAINING_POOL_SHA256
        expected_historical_sha = HISTORICAL_VISION_CHECKPOINT_SHA256
    elif modality == "text":
        metadata = inspect_main_checkpoint(path, "text")
        expected_ratio = TEXT_TARGET_RATIO
        expected_pool = TEXT_TRAINING_POOL_SHA256
        expected_historical_sha = HISTORICAL_TEXT_CHECKPOINT_SHA256
    else:
        raise ValueError("modality must be vision or text")
    expected = {
        "target_ratio": expected_ratio,
        "training_seed": RUN_SEED,
        "data_seed": DATA_SEED,
        "pool_size": TRAINING_POOL_SIZE,
        "dataset_sha256": expected_pool,
        "capacity_factors": list(CAPACITY_FACTORS),
    }
    for field, wanted in expected.items():
        if metadata.get(field) != wanted:
            raise ValueError(
                f"{modality} checkpoint {field}={metadata.get(field)!r}; expected {wanted!r}"
            )
    digest = file_sha256(path)
    if metadata["format"] == "historical_two_stage" and digest != expected_historical_sha:
        raise ValueError(
            f"historical {modality} checkpoint differs from the result-generating weight"
        )
    metadata["checkpoint_sha256"] = digest
    metadata["modality"] = modality
    return metadata


def validate_fixed_inputs(image_root: Path) -> tuple[Path, ...]:
    if text_sha256(TEXT_INPUT) != TEXT_INPUT_SHA256:
        raise RuntimeError("registered text input was modified")
    if not image_root.is_dir():
        raise FileNotFoundError(f"missing COCO image root: {image_root}")
    paths = tuple(image_root / file_name for file_name, _ in IMAGE_IDENTITIES)
    for path, (_, expected_sha) in zip(paths, IMAGE_IDENTITIES):
        if not path.is_file():
            raise FileNotFoundError(f"missing registered visualization image: {path}")
        if file_sha256(path) != expected_sha:
            raise ValueError(f"registered visualization image has changed: {path.name}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise ValueError(f"registered visualization image is unreadable: {path}") from error
    return paths


def validate_args(
    args: argparse.Namespace,
) -> tuple[tuple[Path, ...], dict[str, dict[str, Any]]]:
    for path, label in (
        (args.vision_checkpoint, "visual main Stage-2 checkpoint"),
        (args.text_checkpoint, "text main Stage-2 checkpoint"),
        (args.pretrained, "Dense CLIP weights"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if args.cpu_threads < 0:
        raise ValueError("cpu thread count must be non-negative")
    if file_sha256(args.pretrained) != PRETRAINED_SHA256:
        raise ValueError("Dense CLIP checkpoint identity differs from the paper experiment")
    image_paths = validate_fixed_inputs(args.image_root)
    metadata = {
        "vision": validate_checkpoint(args.vision_checkpoint, "vision"),
        "text": validate_checkpoint(args.text_checkpoint, "text"),
    }
    return image_paths, metadata


def merge_word_groups(
    tokenizer: Any,
    token_row: Tensor,
    eot_position: int,
) -> list[dict[str, Any]]:
    """Rebuild readable words while retaining every constituent BPE position."""

    groups: list[dict[str, Any]] = []
    pieces: list[str] = []
    positions: list[int] = []
    for position in range(1, eot_position):
        raw = tokenizer.decode([int(token_row[position])])
        clean = raw.replace("</w>", "").replace("\n", " ")
        pieces.append(clean.strip())
        positions.append(position)
        if raw.endswith(" ") or raw.endswith("</w>"):
            label = "".join(piece for piece in pieces if piece)
            if label:
                groups.append({"label": label, "positions": positions.copy()})
            pieces, positions = [], []
    if pieces:
        label = "".join(piece for piece in pieces if piece)
        if label:
            groups.append({"label": label, "positions": positions.copy()})
    return groups


def extract_vision_routes(output: Any, batch_size: int) -> dict[str, Any]:
    layer_by_number = {
        int(layer.transformer_layer) + 1: (position, layer)
        for position, layer in enumerate(output.layers)
    }
    assignments: dict[str, list[Any]] = {}
    usages: dict[str, list[Any]] = {}
    ratios: dict[str, list[float]] = {}
    for layer_number in VISION_LAYERS:
        if layer_number not in layer_by_number:
            raise ValueError(f"visual output is missing Layer {layer_number}")
        position, layer = layer_by_number[layer_number]
        gates = layer.routing.gates.detach().float().cpu()
        expected_shape = (batch_size * PATCHES_PER_IMAGE, NUM_EXPERTS)
        if tuple(gates.shape) != expected_shape:
            raise ValueError(
                f"Layer {layer_number} routing shape={tuple(gates.shape)}; "
                f"expected {expected_shape}"
            )
        route_ids = gates.argmax(dim=-1).reshape(
            batch_size,
            PATCH_GRID_SIZE,
            PATCH_GRID_SIZE,
        )
        one_hot = torch.nn.functional.one_hot(route_ids, NUM_EXPERTS).float()
        assignments[str(layer_number)] = route_ids.tolist()
        usages[str(layer_number)] = one_hot.mean(dim=(1, 2)).tolist()
        ratios[str(layer_number)] = (
            output.retention_ratios[position].detach().float().cpu().tolist()
        )
    return {
        "layers_one_based": list(VISION_LAYERS),
        "patch_grid": [PATCH_GRID_SIZE, PATCH_GRID_SIZE],
        "assignments_by_layer": assignments,
        "expert_usage_by_layer_and_image": usages,
        "actual_retention_ratios_by_layer": ratios,
    }


def extract_text_routes(output: Any, tokenizer: Any, token_ids: Tensor) -> dict[str, Any]:
    if token_ids.shape[0] != 1:
        raise ValueError("the fixed text visualization requires one text input")
    layer_by_number = {
        int(layer.transformer_layer) + 1: (position, layer)
        for position, layer in enumerate(output.layers)
    }
    if TEXT_LAYER not in layer_by_number:
        raise ValueError(f"text output is missing Layer {TEXT_LAYER}")
    sparse_position, layer = layer_by_number[TEXT_LAYER]
    routes = layer.routing.gates.detach().argmax(dim=-1).reshape(1, -1).cpu()
    eot_position = int(output.eos_positions[0].item())
    if eot_position != int(token_ids[0].argmax().item()):
        raise RuntimeError("text encoder and tokenizer disagree on the EOT position")
    if routes.shape[1] != token_ids.shape[1] - 1:
        raise ValueError("text router output does not cover every non-first position")

    words: list[dict[str, Any]] = []
    for group in merge_word_groups(tokenizer, token_ids[0].cpu(), eot_position):
        label = str(group["label"])
        if label in PUNCTUATION:
            continue
        positions = [int(value) for value in group["positions"]]
        words.append(
            {
                "label": label,
                "positions": positions,
                "piece_labels": [
                    tokenizer.decode([int(token_ids[0, position])]).replace("</w>", "").strip()
                    for position in positions
                ],
                "piece_routes": [int(routes[0, position - 1].item()) for position in positions],
            }
        )
    usage = [0] * NUM_EXPERTS
    for word in words:
        for route in word["piece_routes"]:
            usage[route] += 1
    return {
        "layer_one_based": TEXT_LAYER,
        "eot_position": eot_position,
        "content_bpe_token_count_before_punctuation_filter": eot_position - 1,
        "expert_usage_counts": usage,
        "actual_retention_ratios": (
            output.retention_ratios[sparse_position].detach().float().cpu().tolist()
        ),
        "words": words,
    }


def build_payload(
    checkpoint_metadata: Mapping[str, Any],
    vision_routes: Mapping[str, Any],
    text_routes: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = protocol_manifest()
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "routing": ROUTING_MODE,
        "run_seed": RUN_SEED,
        "inputs": {
            "vision": {
                "dataset": "COCO-val2017",
                "images": manifest["vision"]["images"],
            },
            "text": {
                "text": TEXT_INPUT,
                "sha256": TEXT_INPUT_SHA256,
            },
        },
        "checkpoint_metadata": {key: dict(value) for key, value in checkpoint_metadata.items()},
        "vision": dict(vision_routes),
        "text": dict(text_routes),
        "tf32": False,
    }


def _validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    modality: str,
    source: str | Path,
) -> None:
    expected = {
        "vision": (VISION_TARGET_RATIO, VISION_TRAINING_POOL_SHA256),
        "text": (TEXT_TARGET_RATIO, TEXT_TRAINING_POOL_SHA256),
    }
    target, pool_sha = expected[modality]
    fields = {
        "modality": modality,
        "target_ratio": target,
        "training_seed": RUN_SEED,
        "data_seed": DATA_SEED,
        "pool_size": TRAINING_POOL_SIZE,
        "dataset_sha256": pool_sha,
        "capacity_factors": list(CAPACITY_FACTORS),
    }
    for field, wanted in fields.items():
        if metadata.get(field) != wanted:
            raise ValueError(f"{source}: {modality} checkpoint disagrees on {field}")
    digest = metadata.get("checkpoint_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{source}: {modality} checkpoint SHA-256 is invalid")
    historical_sha = (
        HISTORICAL_VISION_CHECKPOINT_SHA256
        if modality == "vision"
        else HISTORICAL_TEXT_CHECKPOINT_SHA256
    )
    if metadata.get("format") == "historical_two_stage" and digest != historical_sha:
        raise ValueError(f"{source}: wrong historical {modality} checkpoint")


def validate_analysis(payload: Mapping[str, Any], source: str | Path) -> None:
    expected = {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        "model_name": MODEL_NAME,
        "routing": ROUTING_MODE,
        "run_seed": RUN_SEED,
        "tf32": False,
    }
    for field, wanted in expected.items():
        if payload.get(field) != wanted:
            raise ValueError(f"{source}: {field}={payload.get(field)!r}; expected {wanted!r}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"{source}: missing fixed input identities")
    expected_images = protocol_manifest()["vision"]["images"]
    vision_inputs = inputs.get("vision")
    text_inputs = inputs.get("text")
    if not isinstance(vision_inputs, Mapping) or vision_inputs.get("images") != expected_images:
        raise ValueError(f"{source}: visual input selection or order has changed")
    if not isinstance(text_inputs, Mapping) or text_inputs != {
        "text": TEXT_INPUT,
        "sha256": TEXT_INPUT_SHA256,
    }:
        raise ValueError(f"{source}: text input has changed")

    all_metadata = payload.get("checkpoint_metadata")
    if not isinstance(all_metadata, Mapping):
        raise ValueError(f"{source}: missing checkpoint metadata")
    for modality in ("vision", "text"):
        metadata = all_metadata.get(modality)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{source}: missing {modality} checkpoint metadata")
        _validate_checkpoint_metadata(metadata, modality, source)

    vision = payload.get("vision")
    if not isinstance(vision, Mapping):
        raise ValueError(f"{source}: missing visual routing analysis")
    if vision.get("layers_one_based") != list(VISION_LAYERS):
        raise ValueError(f"{source}: wrong visual layers")
    if vision.get("patch_grid") != [PATCH_GRID_SIZE, PATCH_GRID_SIZE]:
        raise ValueError(f"{source}: wrong visual patch grid")
    assignments = vision.get("assignments_by_layer")
    usages = vision.get("expert_usage_by_layer_and_image")
    ratios = vision.get("actual_retention_ratios_by_layer")
    if not all(isinstance(value, Mapping) for value in (assignments, usages, ratios)):
        raise ValueError(f"{source}: incomplete visual routing analysis")
    image_count = len(IMAGE_IDENTITIES)
    for layer in VISION_LAYERS:
        key = str(layer)
        route_ids = np.asarray(assignments.get(key))
        usage = np.asarray(usages.get(key), dtype=float)
        ratio = np.asarray(ratios.get(key), dtype=float)
        if route_ids.shape != (image_count, PATCH_GRID_SIZE, PATCH_GRID_SIZE):
            raise ValueError(f"{source}: invalid Layer {layer} assignment shape")
        if not np.issubdtype(route_ids.dtype, np.integer):
            raise ValueError(f"{source}: Layer {layer} assignments must be integers")
        if np.any(route_ids < 0) or np.any(route_ids >= NUM_EXPERTS):
            raise ValueError(f"{source}: Layer {layer} contains an invalid expert id")
        expected_usage = np.stack(
            [
                np.bincount(row.reshape(-1), minlength=NUM_EXPERTS) / PATCHES_PER_IMAGE
                for row in route_ids
            ]
        )
        if usage.shape != (image_count, NUM_EXPERTS) or not np.allclose(
            usage, expected_usage, atol=1e-12
        ):
            raise ValueError(f"{source}: Layer {layer} usage disagrees with assignments")
        if ratio.shape != (NUM_EXPERTS,) or np.any(ratio <= 0) or np.any(ratio > 1):
            raise ValueError(f"{source}: Layer {layer} has invalid retention ratios")

    text = payload.get("text")
    if not isinstance(text, Mapping) or text.get("layer_one_based") != TEXT_LAYER:
        raise ValueError(f"{source}: wrong text routing layer")
    eot_position = text.get("eot_position")
    if not isinstance(eot_position, int) or eot_position < 2:
        raise ValueError(f"{source}: invalid text EOT position")
    if text.get("content_bpe_token_count_before_punctuation_filter") != eot_position - 1:
        raise ValueError(f"{source}: inconsistent text BPE count")
    words = text.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError(f"{source}: missing routed words")
    usage = [0] * NUM_EXPERTS
    previous_position = 0
    for word in words:
        if not isinstance(word, Mapping) or not isinstance(word.get("label"), str):
            raise ValueError(f"{source}: invalid routed word record")
        if word["label"] in PUNCTUATION:
            raise ValueError(f"{source}: punctuation must not be drawn as a word")
        positions = word.get("positions")
        piece_labels = word.get("piece_labels")
        piece_routes = word.get("piece_routes")
        if not isinstance(positions, list) or not positions:
            raise ValueError(f"{source}: routed word has no BPE positions")
        if len(positions) != len(piece_labels or ()) or len(positions) != len(
            piece_routes or ()
        ):
            raise ValueError(f"{source}: routed word pieces are inconsistent")
        if positions != sorted(positions) or positions[0] <= previous_position:
            raise ValueError(f"{source}: routed BPE positions are not ordered")
        if positions[-1] >= eot_position:
            raise ValueError(f"{source}: routed word crosses the EOT token")
        previous_position = positions[-1]
        for route in piece_routes:
            if not isinstance(route, int) or not 0 <= route < NUM_EXPERTS:
                raise ValueError(f"{source}: invalid text expert id")
            usage[route] += 1
    if text.get("expert_usage_counts") != usage:
        raise ValueError(f"{source}: text usage disagrees with routed BPE pieces")
    text_ratios = np.asarray(text.get("actual_retention_ratios"), dtype=float)
    if (
        text_ratios.shape != (NUM_EXPERTS,)
        or np.any(text_ratios <= 0)
        or np.any(text_ratios > 1)
    ):
        raise ValueError(f"{source}: invalid text retention ratios")


def run(args: argparse.Namespace) -> dict[str, Any]:
    image_paths, checkpoint_metadata = validate_args(args)
    check = {
        **protocol_manifest(),
        "pretrained": str(args.pretrained.resolve()),
        "pretrained_sha256": PRETRAINED_SHA256,
        "image_root": str(args.image_root.resolve()),
        "vision_checkpoint": str(args.vision_checkpoint.resolve()),
        "vision_checkpoint_sha256": checkpoint_metadata["vision"]["checkpoint_sha256"],
        "text_checkpoint": str(args.text_checkpoint.resolve()),
        "text_checkpoint_sha256": checkpoint_metadata["text"]["checkpoint_sha256"],
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

    vision_encoder, loaded_vision, preprocess, _ = load_vision_encoder(
        args.vision_checkpoint,
        args.pretrained,
        device,
    )
    if loaded_vision["checkpoint_step"] != checkpoint_metadata["vision"]["checkpoint_step"]:
        raise RuntimeError("visual checkpoint changed between validation and loading")
    image_tensors = []
    for path in image_paths:
        with Image.open(path) as image:
            image_tensors.append(preprocess(image.convert("RGB")))
    with torch.inference_mode():
        vision_output = vision_encoder(
            torch.stack(image_tensors).to(device),
            routing_mode="learned",
        )
    vision_routes = extract_vision_routes(vision_output, len(image_paths))
    del vision_output, vision_encoder, preprocess, image_tensors
    gc.collect()

    text_encoder, loaded_text, tokenizer = load_text_encoder(
        args.text_checkpoint,
        args.pretrained,
        device,
    )
    if loaded_text["checkpoint_step"] != checkpoint_metadata["text"]["checkpoint_step"]:
        raise RuntimeError("text checkpoint changed between validation and loading")
    token_ids = tokenizer([TEXT_INPUT]).to(device)
    with torch.inference_mode():
        text_output = text_encoder(token_ids, routing_mode="learned")
    text_routes = extract_text_routes(text_output, tokenizer, token_ids)

    payload = build_payload(checkpoint_metadata, vision_routes, text_routes)
    validate_analysis(payload, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"analysis={args.output}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))

"""Validate and merge the layer shards produced for Figure 3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .protocol import (
    CALIBRATION_IMAGES,
    CAPACITY_LEVELS,
    CHANNEL_RANKING,
    COCO_ANNOTATIONS_SHA256,
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
    PRETRAINED_SHA256,
    PROTOCOL,
    SEED,
    SPLIT_METHOD,
    STUDY_NAME,
)


HISTORICAL_CHANNEL_RANKING = "mean squared GELU activation times squared c_proj column norm"
HISTORICAL_COMPARISON = "local Dense FFN output, excluding CLS token"
SHARED_FIELDS = (
    "protocol",
    "model_name",
    "model_key",
    "backbone",
    "pretrained_sha256",
    "coco_annotation_sha256",
    "seed",
    "split_method",
    "calibration_images",
    "evaluation_images",
    "calibration_split_sha256",
    "evaluation_split_sha256",
    "split_disjoint",
    "patch_tokens_per_image",
    "ffn_dim",
    "capacity_levels",
    "cosine_threshold",
    "nre_threshold",
    "channel_ranking",
    "comparison",
    "batch_first_verified",
    "tf32",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="*")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "analysis.json",
    )
    args = parser.parse_args(argv)
    if not args.shards:
        args.shards = [
            OUTPUT_ROOT / "shards" / "layers_01_12_result.json",
            OUTPUT_ROOT / "shards" / "layers_13_24_result.json",
        ]
    return args


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Figure-3 shard: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def normalize_shard(payload: Mapping[str, Any], source: str | Path) -> dict[str, Any]:
    historical = "study" not in payload
    levels = payload.get("levels") if historical else payload.get("capacity_levels")
    normalized = {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": payload.get("protocol"),
        "paper_scope": PAPER_SCOPE,
        "model_name": payload.get("model") if historical else payload.get("model_name"),
        "model_key": MODEL_KEY if historical else payload.get("model_key"),
        "backbone": "frozen_dense_clip" if historical else payload.get("backbone"),
        "pretrained_sha256": payload.get("pretrained_sha256"),
        "coco_annotation_sha256": payload.get("coco_annotation_sha256"),
        "seed": payload.get("seed"),
        "split_method": (SPLIT_METHOD if historical else payload.get("split_method")),
        "calibration_images": payload.get("calibration_images"),
        "evaluation_images": payload.get("evaluation_images"),
        "calibration_split_sha256": payload.get("calibration_split_sha256"),
        "evaluation_split_sha256": payload.get("evaluation_split_sha256"),
        "split_disjoint": payload.get("split_disjoint"),
        "patch_tokens_per_image": payload.get("patch_tokens_per_image"),
        "ffn_dim": FFN_DIM if historical else payload.get("ffn_dim"),
        "capacity_levels": levels,
        "cosine_threshold": payload.get("cosine_threshold"),
        "nre_threshold": payload.get("nre_threshold"),
        "channel_ranking": (CHANNEL_RANKING if historical else payload.get("channel_ranking")),
        "comparison": COMPARISON if historical else payload.get("comparison"),
        "batch_first_verified": payload.get("batch_first_verified"),
        "tf32": payload.get("tf32"),
        "layer_start": payload.get("layer_start"),
        "layer_end": payload.get("layer_end"),
        "ranking_file": payload.get("ranking_file"),
        "layers": payload.get("layers"),
        "source_format": "historical" if historical else "release_v1",
    }
    if historical:
        if payload.get("channel_ranking") != HISTORICAL_CHANNEL_RANKING:
            raise ValueError(f"{source}: historical channel-ranking rule changed")
        if payload.get("comparison") != HISTORICAL_COMPARISON:
            raise ValueError(f"{source}: historical comparison rule changed")
    validate_shard(normalized, source)
    return normalized


def validate_layer_record(
    record: Mapping[str, Any],
    source: str | Path,
    layer: int,
) -> None:
    observations = record.get("observations")
    counts = record.get("required_capacity_counts")
    proportions = record.get("required_capacity_proportions")
    mean = record.get("layer_mean")
    if not isinstance(observations, int) or observations <= 0:
        raise ValueError(f"{source}: layer {layer} has invalid observations")
    if not isinstance(counts, list) or len(counts) != len(CAPACITY_LEVELS):
        raise ValueError(f"{source}: layer {layer} has invalid capacity counts")
    if not all(isinstance(value, int) and value >= 0 for value in counts):
        raise ValueError(f"{source}: layer {layer} capacity counts must be non-negative")
    if sum(counts) != observations:
        raise ValueError(f"{source}: layer {layer} does not conserve token counts")
    if not isinstance(proportions, list) or len(proportions) != len(CAPACITY_LEVELS):
        raise ValueError(f"{source}: layer {layer} has invalid capacity proportions")
    expected_proportions = [count / observations for count in counts]
    if any(
        not math.isclose(float(value), expected, abs_tol=1e-12)
        for value, expected in zip(proportions, expected_proportions)
    ):
        raise ValueError(f"{source}: layer {layer} proportions disagree with counts")
    expected_mean = (
        sum(level * count for level, count in zip(CAPACITY_LEVELS, counts)) / observations
    )
    if not math.isclose(float(mean), expected_mean, abs_tol=1e-12):
        raise ValueError(f"{source}: layer {layer} mean disagrees with counts")
    vector_fields = (
        "cosine_pass_counts",
        "nre_pass_counts",
        "joint_pass_counts",
        "mean_cosine_by_capacity",
        "mean_nre_by_capacity",
    )
    for field in vector_fields:
        values = record.get(field)
        if not isinstance(values, list) or len(values) != len(CAPACITY_LEVELS):
            raise ValueError(f"{source}: layer {layer} has invalid {field}")


def validate_split_identity(value: Any, source: str | Path, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{source}: {field} must be a lowercase SHA-256 digest")


def validate_shard(payload: Mapping[str, Any], source: str | Path) -> None:
    expected = {
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "backbone": "frozen_dense_clip",
        "pretrained_sha256": PRETRAINED_SHA256,
        "coco_annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "seed": SEED,
        "calibration_images": CALIBRATION_IMAGES,
        "evaluation_images": EVALUATION_IMAGES,
        "split_disjoint": True,
        "patch_tokens_per_image": PATCH_TOKENS,
        "ffn_dim": FFN_DIM,
        "capacity_levels": list(CAPACITY_LEVELS),
        "cosine_threshold": COSINE_THRESHOLD,
        "nre_threshold": NRE_THRESHOLD,
        "channel_ranking": CHANNEL_RANKING,
        "comparison": COMPARISON,
        "batch_first_verified": True,
        "tf32": False,
    }
    for field, wanted in expected.items():
        if payload.get(field) != wanted:
            raise ValueError(f"{source}: {field}={payload.get(field)!r}; expected {wanted!r}")
    if payload.get("split_method") != SPLIT_METHOD:
        raise ValueError(f"{source}: split method differs from Figure 3")
    for field in ("calibration_split_sha256", "evaluation_split_sha256"):
        validate_split_identity(payload.get(field), source, field)
    if payload["calibration_split_sha256"] == payload["evaluation_split_sha256"]:
        raise ValueError(f"{source}: calibration and evaluation identities must differ")
    layer_start = payload.get("layer_start")
    layer_end = payload.get("layer_end")
    if not isinstance(layer_start, int) or not isinstance(layer_end, int):
        raise ValueError(f"{source}: shard has no integer layer range")
    if not 1 <= layer_start <= layer_end <= NUM_LAYERS:
        raise ValueError(f"{source}: invalid layer range {layer_start}..{layer_end}")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError(f"{source}: shard is missing layer results")
    expected_layers = list(range(layer_start, layer_end + 1))
    if sorted(int(layer) for layer in layers) != expected_layers:
        raise ValueError(f"{source}: layer records do not match the declared range")
    expected_observations = EVALUATION_IMAGES * PATCH_TOKENS
    for layer in expected_layers:
        record = layers[str(layer)]
        if not isinstance(record, Mapping):
            raise ValueError(f"{source}: layer {layer} must be a mapping")
        validate_layer_record(record, source, layer)
        if record["observations"] != expected_observations:
            raise ValueError(
                f"{source}: layer {layer} observations={record['observations']}; "
                f"expected {expected_observations}"
            )


def merge_shards(
    payloads: Sequence[Mapping[str, Any]],
    sources: Sequence[str | Path],
) -> dict[str, Any]:
    if len(payloads) != len(sources) or not payloads:
        raise ValueError("payloads and sources must be non-empty and aligned")
    normalized = [
        normalize_shard(payload, source) for payload, source in zip(payloads, sources)
    ]
    reference = normalized[0]
    for shard, source in zip(normalized[1:], sources[1:]):
        for field in SHARED_FIELDS:
            if shard[field] != reference[field]:
                raise ValueError(f"{source}: shard disagrees on {field}")
    layers: dict[str, Any] = {}
    for shard, source in zip(normalized, sources):
        for layer, record in shard["layers"].items():
            if layer in layers:
                raise ValueError(f"{source}: duplicate layer {layer}")
            layers[layer] = record
    if sorted(int(layer) for layer in layers) != list(range(1, NUM_LAYERS + 1)):
        raise ValueError(f"Figure 3 requires exactly layers 1 through {NUM_LAYERS}")
    return {
        "format_version": 1,
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "paper_scope": PAPER_SCOPE,
        **{field: reference[field] for field in SHARED_FIELDS if field != "protocol"},
        "source_shards": [str(Path(source).resolve()) for source in sources],
        "layers": {str(layer): layers[str(layer)] for layer in range(1, NUM_LAYERS + 1)},
    }


def validate_merged(payload: Mapping[str, Any], source: str | Path) -> None:
    synthetic_shard = {
        **payload,
        "layer_start": 1,
        "layer_end": NUM_LAYERS,
    }
    validate_shard(synthetic_shard, source)
    if payload.get("paper_scope") != PAPER_SCOPE:
        raise ValueError(f"{source}: paper scope differs from Figure 3")
    sources = payload.get("source_shards")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{source}: merged analysis must identify its source shards")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    payloads = [load_json(path) for path in args.shards]
    merged = merge_shards(payloads, args.shards)
    validate_merged(merged, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"analysis={args.output}")

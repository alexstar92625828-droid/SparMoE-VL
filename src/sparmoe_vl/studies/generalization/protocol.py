"""Immutable Table-4 protocol and main-checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ...common.two_stage import STAGE2_PROTOCOL
from ...text.encoder import SparMoETextEncoder
from ..vision_budget_sweep import load_encoder as load_vision_encoder
from ..vision_budget_sweep import inspect_checkpoint as inspect_vision_checkpoint


STUDY_NAME = "cross_dataset_generalization"
MODEL_NAME = "ViT-L-14"
SEEDS = (42, 123, 2026)
DATA_SEED = 42
POOL_SIZE = 500_000
CAPACITY_FACTORS = (0.7, 0.8, 0.9, 1.0)
VISION_TARGET_RATIO = 0.7
TEXT_TARGET_RATIO = 0.6
VISION_DATASET_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"
TEXT_DATASET_SHA256 = "c76621edddeeb546f6fa798552b2ae11eb409c63b520080d42a8a37b66f31e9d"
DATASET_COUNTS = {
    "coco": {"images": 5_000, "texts": 25_014},
    "flickr30k": {"images": 1_000, "texts": 5_000},
    "cifar100": {"images": 10_000, "classes": 100},
    "imagenet1k": {"images": 50_000, "classes": 1_000},
    "food101": {"images": 25_250, "classes": 101},
}
EVALUATION_IDENTITIES = {
    "coco_annotations_sha256": (
        "afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
    ),
    "flickr_annotations_sha256": (
        "395990db603ab8bafd5c7ab2746b22058bb1e75b78b3eb56ad755931364ac137"
    ),
    "cifar100_test_sha256": (
        "4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864"
    ),
    "imagenet_labels_sha256": (
        "313fc3f23c864dde9183c9a368809065765dd1b657a6904fded44e3e46d34604"
    ),
    "imagenet_synsets_sha256": (
        "70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15"
    ),
    "imagenet_order_sha256": (
        "dc7c4a6d3e129595211adc6a6c0b35541f5935ffa08bbbb6826753f4fbe122b2"
    ),
    "food101_test_sha256": ("4c8977721ce1efe10085ef1be227dc39c4c374b578d6f08614253e46f0750594"),
    "food101_order_sha256": (
        "cd90179aadf7bdeba7b77aa9df6935ca963d0093d0309d9c46b265d9d2edd3b9"
    ),
}
DENSE_VISION_MACS_G = 81.012768768
DENSE_TEXT_MACS_G = 6.64925184


def torch_load(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {checkpoint_path}")
    return payload


def _release_metadata(checkpoint: Mapping[str, Any], modality: str) -> dict[str, Any]:
    expected = {
        "method": "sparmoe_vl_two_stage",
        "format_version": 3,
        "protocol": STAGE2_PROTOCOL,
        "stage": 2,
        "modality": modality,
    }
    for key, wanted in expected.items():
        if checkpoint.get(key) != wanted:
            raise ValueError(f"checkpoint {key}={checkpoint.get(key)!r}; expected {wanted!r}")
    dataset = checkpoint.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("release checkpoint does not contain its training-data identity")
    return {
        "format": "release_v3",
        "protocol": checkpoint.get("protocol"),
        "training_seed": checkpoint.get("training_seed"),
        "data_seed": dataset.get("data_seed"),
        "pool_size": dataset.get("samples"),
        "dataset_sha256": dataset.get("ordered_sha256"),
        "target_ratio": checkpoint.get("target_ratio"),
        "capacity_factors": checkpoint.get("capacity_factors"),
        "checkpoint_step": checkpoint.get("step"),
        "model_name": checkpoint.get("model_name", MODEL_NAME),
    }


def checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    modality: str,
) -> dict[str, Any]:
    """Validate one visual/text main checkpoint used by Table 4."""

    if modality not in ("vision", "text"):
        raise ValueError("modality must be vision or text")
    metadata = _release_metadata(checkpoint, modality)
    expected_ratio = VISION_TARGET_RATIO if modality == "vision" else TEXT_TARGET_RATIO
    expected_sha = VISION_DATASET_SHA256 if modality == "vision" else TEXT_DATASET_SHA256
    checks = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "dataset_sha256": expected_sha,
        "capacity_factors": list(CAPACITY_FACTORS),
        "model_name": MODEL_NAME,
    }
    if int(metadata["training_seed"]) not in SEEDS:
        raise ValueError(f"checkpoint seed is not one of {SEEDS}")
    for key in ("data_seed", "pool_size"):
        if int(metadata[key]) != checks[key]:
            raise ValueError(f"checkpoint {key}={metadata[key]!r}; expected {checks[key]!r}")
    for key in ("dataset_sha256", "model_name"):
        if metadata[key] != checks[key]:
            raise ValueError(f"checkpoint {key}={metadata[key]!r}; expected {checks[key]!r}")
    if tuple(float(value) for value in metadata["capacity_factors"]) != CAPACITY_FACTORS:
        raise ValueError("checkpoint capacity factors differ from the Table-4 protocol")
    if abs(float(metadata["target_ratio"]) - expected_ratio) > 1e-8:
        raise ValueError(
            f"{modality} target ratio={metadata['target_ratio']}; expected {expected_ratio}"
        )
    metadata.update(
        modality=modality,
        training_seed=int(metadata["training_seed"]),
        data_seed=int(metadata["data_seed"]),
        pool_size=int(metadata["pool_size"]),
        target_ratio=float(metadata["target_ratio"]),
        checkpoint_step=int(metadata["checkpoint_step"]),
        capacity_factors=list(CAPACITY_FACTORS),
    )
    return metadata


def inspect_checkpoint(path: str | Path, modality: str) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if modality == "vision":
        metadata = inspect_vision_checkpoint(checkpoint_path)
        if abs(metadata["target_ratio"] - VISION_TARGET_RATIO) > 1e-8:
            raise ValueError("Table 4 requires the p=0.7 visual main checkpoint")
        metadata["modality"] = "vision"
    else:
        metadata = checkpoint_metadata(torch_load(checkpoint_path), modality)
        metadata["checkpoint"] = str(checkpoint_path.resolve())
    return metadata


def _load_historical_text_state(
    encoder: SparMoETextEncoder,
    checkpoint: Mapping[str, Any],
) -> None:
    hypernetwork = checkpoint.get("hypernetwork")
    budget_mlps = checkpoint.get("budget_mlps")
    if not isinstance(hypernetwork, Mapping) or not isinstance(budget_mlps, Mapping):
        raise ValueError("historical text checkpoint is missing controller state")
    mapped_hypernetwork: dict[str, Any] = {}
    for key, value in hypernetwork.items():
        if key == "z":
            mapped_hypernetwork["latent_codes"] = value
        elif key.startswith("bigru."):
            mapped_hypernetwork[f"encoder.{key.removeprefix('bigru.')}"] = value
        elif key.startswith("proj."):
            mapped_hypernetwork[f"projection.{key.removeprefix('proj.')}"] = value
        else:
            raise ValueError(f"unknown historical hypernetwork key: {key}")
    encoder.sparse_pattern_generator.hypernetwork.load_state_dict(
        mapped_hypernetwork,
        strict=True,
    )
    base_logits = []
    for position, layer_index in enumerate(encoder.sparse_layers):
        prefix = f"{layer_index}.proj_mlp_d."
        projection_state = {
            key.removeprefix(prefix): value
            for key, value in budget_mlps.items()
            if key.startswith(prefix)
        }
        encoder.sparse_pattern_generator.layer_projections[position].load_state_dict(
            projection_state,
            strict=True,
        )
        router_key = f"{layer_index}.router.weight"
        ratio_key = f"{layer_index}.full_ratio_logit"
        if router_key not in budget_mlps or ratio_key not in budget_mlps:
            raise ValueError(f"historical text state is incomplete at layer {layer_index}")
        encoder.routers[position].projection.load_state_dict(
            {"weight": budget_mlps[router_key]},
            strict=True,
        )
        base_logits.append(torch.as_tensor(budget_mlps[ratio_key]).reshape(()))
    with torch.no_grad():
        encoder.budget.base_ratio_logits.copy_(torch.stack(base_logits))


def load_text_encoder(
    checkpoint_path: str | Path,
    pretrained: str | Path,
    device: str | torch.device,
    *,
    discard_visual_tower: bool = True,
) -> tuple[SparMoETextEncoder, dict[str, Any], Any]:
    """Load cleaned or genuine historical p=0.6 text main weights."""

    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("install open_clip_torch before evaluation") from error
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch_load(checkpoint_path)
    metadata = checkpoint_metadata(checkpoint, "text")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(pretrained),
        force_quick_gelu=True,
    )
    if discard_visual_tower:
        clip_model.visual = nn.Identity()
    resolved_device = torch.device(device)
    clip_model = clip_model.to(resolved_device).eval()
    train_args = checkpoint.get("train_args", {})
    temperature = float(train_args.get("tau", 0.4)) if isinstance(train_args, Mapping) else 0.4
    encoder = SparMoETextEncoder(
        clip_model=clip_model,
        sparse_layers=list(range(12)),
        target_ratio=TEXT_TARGET_RATIO,
        capacity_factors=CAPACITY_FACTORS,
        router_temperature=temperature,
        mask_temperature=temperature,
        training_stage=2,
        routing_target_scope="all_nonfirst",
    ).to(resolved_device)
    state = checkpoint.get("encoder")
    if not isinstance(state, Mapping):
        raise ValueError("release text checkpoint has no encoder state")
    encoder.budget.load_state_dict(state["budget"], strict=True)
    encoder.sparse_pattern_generator.load_state_dict(
        state["sparse_pattern_generator"],
        strict=True,
    )
    encoder.routers.load_state_dict(state["routers"], strict=True)
    del checkpoint
    encoder.eval()
    metadata["checkpoint"] = str(checkpoint_path.resolve())
    return encoder, metadata, open_clip.get_tokenizer(MODEL_NAME)


def load_main_vision_encoder(
    checkpoint_path: str | Path,
    pretrained: str | Path,
    device: str | torch.device,
) -> tuple[Any, dict[str, Any], Any, Any]:
    encoder, metadata, preprocess, tokenizer = load_vision_encoder(
        checkpoint_path,
        pretrained,
        device,
    )
    if abs(metadata["target_ratio"] - VISION_TARGET_RATIO) > 1e-8:
        raise ValueError("Table 4 requires the p=0.7 visual main checkpoint")
    metadata["modality"] = "vision"
    return encoder, metadata, preprocess, tokenizer

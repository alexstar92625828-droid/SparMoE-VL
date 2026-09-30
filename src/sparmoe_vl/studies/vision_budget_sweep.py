"""Exact protocol and checkpoint compatibility for the visual budget sweep.

The paper sweep trains p={0.4, 0.5, 0.6, 0.8} with the same two-stage
procedure as the visual main experiment.  Its p=0.7 point is the visual main
experiment itself and is therefore reused, not trained a second time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from ..train_two_stage import parse_args as parse_training_args
from ..train_two_stage import run as run_two_stage
from ..vision.encoder import SparMoEVisionEncoder
from ..common.two_stage import STAGE2_PROTOCOL, TWO_STAGE_PROTOCOL


STUDY_NAME = "vision_budget_sweep"
BUDGET_POINTS = (0.4, 0.5, 0.6, 0.7, 0.8)
TRAINED_BUDGET_POINTS = (0.4, 0.5, 0.6, 0.8)
SEEDS = (42, 123, 2026)
DATA_SEED = 42
POOL_SIZE = 500_000
DATASET_SHA256 = "f035f2b9e7d44ee0acb6ec37fd53945ee784a987892a20c31393f04230f284a0"
CAPACITY_FACTORS = (0.7, 0.8, 0.9, 1.0)
MODEL_NAME = "ViT-L-14"
NUM_LAYERS = 24

STAGE_SETTINGS = {
    1: {
        "steps": 5_000,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0.05,
        "temperature": 0.4,
        "num_workers": 8,
        "gradient_clip_norm": 1.0,
        "best_budget_loss": 0.01,
        "log_every": 100,
        "save_every": 500,
        "loss_weights": {
            "representation": 100.0,
            "global_budget": 50.0,
            "capacity_separation": 1.0,
        },
        "optimizer_sample_exposures": 160_000,
    },
    2: {
        "steps": 5_000,
        "batch_size": 24,
        "learning_rate": 3e-4,
        "weight_decay": 0.05,
        "temperature": 0.4,
        "router_warmup": 1_000,
        "random_evals": 3,
        "num_workers": 8,
        "gradient_clip_norm": 1.0,
        "log_every": 100,
        "save_every": 500,
        "loss_weights": {
            "representation": 100.0,
            "routing": 1.0,
        },
        "optimizer_sample_exposures": 120_000,
    },
}


def registered_ratio(value: float, *, trained_only: bool = False) -> float:
    """Return a canonical paper budget, rejecting unregistered float values."""

    choices = TRAINED_BUDGET_POINTS if trained_only else BUDGET_POINTS
    for point in choices:
        if abs(float(value) - point) <= 1e-8:
            return point
    qualifier = "trainable sweep" if trained_only else "paper"
    raise ValueError(f"p={value} is not a registered {qualifier} budget: {choices}")


def budget_tag(value: float) -> str:
    """Return stable output-directory tags such as ``p04`` and ``p07``."""

    point = registered_ratio(value)
    return f"p{int(round(point * 10)):02d}"


def protocol_manifest(target_ratio: float, seed: int) -> dict[str, Any]:
    """Describe the immutable paper protocol for one sweep point and seed."""

    point = registered_ratio(target_ratio)
    if int(seed) not in SEEDS:
        raise ValueError(f"seed={seed} is not one of the paper seeds: {SEEDS}")
    return {
        "name": STUDY_NAME,
        "target_ratio": point,
        "training_seed": int(seed),
        "source": "visual_main_experiment" if point == 0.7 else "budget_sweep_run",
        "data": {
            "dataset": "ShareGPT4V",
            "data_seed": DATA_SEED,
            "candidate_pool_size": POOL_SIZE,
            "ordered_sha256": DATASET_SHA256,
            "same_candidate_pool_in_both_stages": True,
        },
        "model": {
            "name": MODEL_NAME,
            "sparse_layers": list(range(NUM_LAYERS)),
            "capacity_factors": list(CAPACITY_FACTORS),
        },
        "two_stage_protocol": TWO_STAGE_PROTOCOL,
        "stage1": dict(STAGE_SETTINGS[1]),
        "stage2": dict(STAGE_SETTINGS[2]),
    }


def validate_training_protocol(args: argparse.Namespace) -> None:
    """Refuse settings that would no longer reproduce the paper sweep."""

    if args.modality != "vision":
        raise ValueError("the vision budget sweep only accepts --modality vision")
    point = registered_ratio(args.target_ratio, trained_only=True)
    if point == 0.7:  # defensive; 0.7 is intentionally absent above
        raise ValueError("p=0.7 must reuse the visual main experiment")
    if args.seed not in SEEDS:
        raise ValueError(f"unsupported paper seed: {args.seed}")
    if args.data_seed != DATA_SEED:
        raise ValueError(f"data seed must remain {DATA_SEED}")
    if args.max_samples != POOL_SIZE:
        raise ValueError(f"the sweep requires the exact {POOL_SIZE:,}-sample pool")

    expected = STAGE_SETTINGS[args.stage]
    checks = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "num_workers": args.num_workers,
        "log_every": args.log_every,
        "save_every": args.save_every,
    }
    if args.stage == 1:
        checks["best_budget_loss"] = args.best_budget_loss
    if args.stage == 2:
        checks.update(
            router_warmup=args.router_warmup,
            random_evals=args.random_evals,
        )
    for key, actual in checks.items():
        wanted = expected[key]
        if isinstance(wanted, float):
            matches = abs(float(actual) - wanted) <= 1e-12
        else:
            matches = actual == wanted
        if not matches:
            raise ValueError(
                f"stage {args.stage} {key}={actual}; paper protocol requires {wanted}"
            )


def run_training(stage: int, argv: Optional[Sequence[str]] = None) -> None:
    """Run one strictly registered Stage-1 or Stage-2 sweep job."""

    if stage not in (1, 2):
        raise ValueError("stage must be 1 or 2")
    args = parse_training_args(
        ["--modality", "vision", "--stage", str(stage), *(argv or ())],
        allowed_target_ratios=BUDGET_POINTS,
    )
    validate_training_protocol(args)
    args.study = STUDY_NAME
    args.expected_ordered_sha256 = DATASET_SHA256
    args.study_protocol = protocol_manifest(args.target_ratio, args.seed)
    run_two_stage(args)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # PyTorch releases before mmap support
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def checkpoint_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a paper-protocol Stage-2 checkpoint."""

    expected = {
        "method": "sparmoe_vl_two_stage",
        "format_version": 3,
        "protocol": STAGE2_PROTOCOL,
        "stage": 2,
        "modality": "vision",
    }
    for key, wanted in expected.items():
        if checkpoint.get(key) != wanted:
            raise ValueError(f"checkpoint {key}={checkpoint.get(key)!r}; expected {wanted!r}")
    dataset = checkpoint.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("checkpoint has no dataset identity")
    seed = checkpoint.get("training_seed")
    data_seed = dataset.get("data_seed")
    dataset_sha = dataset.get("ordered_sha256")
    pool_size = dataset.get("samples")
    point = checkpoint.get("target_ratio")
    levels = checkpoint.get("capacity_factors")
    checkpoint_format = "release_v3"
    protocol = checkpoint.get("protocol")

    canonical_point = registered_ratio(float(point))
    if int(seed) not in SEEDS:
        raise ValueError(f"checkpoint seed={seed}; expected one of {SEEDS}")
    if int(data_seed) != DATA_SEED:
        raise ValueError(f"checkpoint data_seed={data_seed}; expected {DATA_SEED}")
    if dataset_sha != DATASET_SHA256:
        raise ValueError("checkpoint was not trained on the registered 500k visual pool")
    if pool_size is None or int(pool_size) != POOL_SIZE:
        raise ValueError(f"checkpoint pool_size={pool_size}; expected {POOL_SIZE}")
    if tuple(float(value) for value in levels) != CAPACITY_FACTORS:
        raise ValueError(f"checkpoint capacity factors differ from {CAPACITY_FACTORS}")
    study = checkpoint.get("study")
    if canonical_point == 0.7 and study == STUDY_NAME:
        raise ValueError("p=0.7 must come from the visual main experiment")
    if canonical_point != 0.7 and study != STUDY_NAME:
        raise ValueError(f"release checkpoint study={study!r}; expected {STUDY_NAME!r}")

    return {
        "format": checkpoint_format,
        "protocol": protocol,
        "target_ratio": canonical_point,
        "training_seed": int(seed),
        "data_seed": int(data_seed),
        "pool_size": POOL_SIZE,
        "dataset_sha256": dataset_sha,
        "checkpoint_step": int(checkpoint["step"]),
        "capacity_factors": list(CAPACITY_FACTORS),
        "reuses_visual_main_experiment": canonical_point == 0.7,
    }


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read only the validated, portable metadata needed by study tooling."""

    checkpoint_path = Path(path)
    metadata = checkpoint_metadata(_torch_load(checkpoint_path))
    metadata["checkpoint"] = str(checkpoint_path.resolve())
    return metadata


def _load_historical_state(
    encoder: SparMoEVisionEncoder,
    checkpoint: Mapping[str, Any],
) -> None:
    """Map genuine 2025 experiment weights into the cleaned implementation."""

    hypernetwork = checkpoint.get("hypernetwork")
    budget_mlps = checkpoint.get("budget_mlps")
    if not isinstance(hypernetwork, Mapping) or not isinstance(budget_mlps, Mapping):
        raise ValueError("historical checkpoint is missing controller state")

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
    for sparse_position, layer_index in enumerate(encoder.sparse_layers):
        prefix = f"{layer_index}.proj_mlp_d."
        projection_state = {
            key.removeprefix(prefix): value
            for key, value in budget_mlps.items()
            if key.startswith(prefix)
        }
        encoder.sparse_pattern_generator.layer_projections[sparse_position].load_state_dict(
            projection_state,
            strict=True,
        )
        router_key = f"{layer_index}.router.weight"
        ratio_key = f"{layer_index}.full_ratio_logit"
        if router_key not in budget_mlps or ratio_key not in budget_mlps:
            raise ValueError(
                f"historical checkpoint is incomplete at visual layer {layer_index}"
            )
        encoder.routers[sparse_position].projection.load_state_dict(
            {"weight": budget_mlps[router_key]},
            strict=True,
        )
        base_logits.append(torch.as_tensor(budget_mlps[ratio_key]).reshape(()))
    with torch.no_grad():
        encoder.budget.base_ratio_logits.copy_(torch.stack(base_logits))


def load_encoder(
    checkpoint_path: str | Path,
    pretrained: str | Path,
    device: str | torch.device,
) -> tuple[SparMoEVisionEncoder, dict[str, Any], Any, Any]:
    """Load either release checkpoints or the real historical paper weights."""

    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("install open_clip_torch before evaluation") from error

    checkpoint_path = Path(checkpoint_path)
    checkpoint = _torch_load(checkpoint_path)
    metadata = checkpoint_metadata(checkpoint)
    model_name = str(checkpoint.get("model_name", MODEL_NAME))
    if model_name != MODEL_NAME:
        raise ValueError(f"the visual sweep requires {MODEL_NAME}, got {model_name}")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=str(pretrained),
        force_quick_gelu=True,
    )
    resolved_device = torch.device(device)
    clip_model = clip_model.to(resolved_device).eval()
    encoder = SparMoEVisionEncoder(
        clip_model=clip_model,
        sparse_layers=list(range(NUM_LAYERS)),
        target_ratio=metadata["target_ratio"],
        capacity_factors=CAPACITY_FACTORS,
        router_temperature=float(checkpoint.get("temperature", 0.4)),
        mask_temperature=float(checkpoint.get("temperature", 0.4)),
        training_stage=2,
    ).to(resolved_device)

    state = checkpoint.get("encoder")
    if not isinstance(state, Mapping):
        raise ValueError("release checkpoint has no encoder state")
    encoder.budget.load_state_dict(state["budget"], strict=True)
    encoder.sparse_pattern_generator.load_state_dict(
        state["sparse_pattern_generator"],
        strict=True,
    )
    encoder.routers.load_state_dict(state["routers"], strict=True)

    del checkpoint
    encoder.eval()
    metadata["checkpoint"] = str(checkpoint_path.resolve())
    tokenizer = open_clip.get_tokenizer(model_name)
    return encoder, metadata, preprocess, tokenizer

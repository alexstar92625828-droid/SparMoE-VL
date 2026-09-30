"""Reproducibility and checkpoint helpers for two-stage training."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from .two_stage import STAGE1_PROTOCOL


class RunLogger:
    """Write identical messages to stdout and a persistent run log."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(message + "\n")


def set_reproducible_seed(seed: int) -> None:
    """Seed model initialization, routing samples, workers, and CUDA."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def plain_args(args: Any) -> Dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value.resolve())
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def dataset_metadata(
    annotations: Path,
    data_seed: int,
    samples: int,
    ordered_sha256: str,
    image_root: Path | None = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source": "ShareGPT4V",
        "annotations": str(annotations.resolve()),
        "data_seed": int(data_seed),
        "samples": int(samples),
        "ordered_sha256": str(ordered_sha256),
    }
    if image_root is not None:
        metadata["image_root"] = str(image_root.resolve())
    return metadata


def save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary_path)
    os.replace(temporary_path, path)


def load_stage1_checkpoint(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid stage-1 checkpoint: {path}")
    return checkpoint


def validate_stage1_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    modality: str,
    model_name: str,
    sparse_layers: Sequence[int],
    target_ratio: float,
    training_seed: int,
    dataset: Mapping[str, Any],
) -> None:
    """Reject a stage transition unless model, seed, and data subset match."""

    expected = {
        "method": "sparmoe_vl_two_stage",
        "format_version": 3,
        "protocol": STAGE1_PROTOCOL,
        "stage": 1,
        "modality": modality,
        "model_name": model_name,
        "sparse_layers": list(sparse_layers),
        "training_seed": int(training_seed),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"stage-1 checkpoint {key}={checkpoint.get(key)!r}; expected {value!r}"
            )
    checkpoint_ratio = checkpoint.get("target_ratio")
    if checkpoint_ratio is None or abs(float(checkpoint_ratio) - target_ratio) > 1e-8:
        raise ValueError("stage-1 and stage-2 target ratios differ")

    checkpoint_dataset = checkpoint.get("dataset")
    if not isinstance(checkpoint_dataset, Mapping):
        raise ValueError("stage-1 checkpoint has no dataset identity")
    identity_keys = ("source", "data_seed", "samples", "ordered_sha256")
    for key in identity_keys:
        if checkpoint_dataset.get(key) != dataset.get(key):
            raise ValueError(
                "stage-1 and stage-2 ShareGPT4V subsets differ: "
                f"dataset.{key}={checkpoint_dataset.get(key)!r}, "
                f"expected {dataset.get(key)!r}"
            )


def configuration_json(args: Any) -> str:
    return json.dumps(plain_args(args), indent=2, ensure_ascii=True)

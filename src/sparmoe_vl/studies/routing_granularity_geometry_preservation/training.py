"""Train the registered expert granularities on the visual main-data pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ...architecture_transfer.clip.model import (
    CLIPSparMoE,
    build_model,
    controller_state,
    initialize_stage2,
)
from ...common.data import ShareGPT4VImageTrainingDataset
from ...common.training import (
    RunLogger,
    dataset_metadata,
    plain_args,
    save_checkpoint_atomic,
    seed_worker,
    set_reproducible_seed,
)
from ...common.two_stage import protocol_for_stage
from .checkpoints import (
    file_sha256,
    stage1_metadata as validate_stage1_metadata,
    torch_load,
)
from .protocol import (
    DATA_SEED,
    EXPERT_COUNTS,
    MODEL_KEY,
    MODEL_NAME,
    MODALITY,
    NUM_WORKERS,
    OUTPUT_ROOT,
    PRETRAINED,
    PRETRAINED_SHA256,
    RUN_SEED,
    STUDY_NAME,
    STAGE1_BATCH_SIZE,
    STAGE2_BATCH_SIZE,
    TARGET_RATIO,
    TRAIN_ANNOTATIONS,
    TRAIN_IMAGES,
    TRAIN_STEPS,
    TRAINING_POOL_SHA256,
    TRAINING_POOL_SIZE,
    capacity_factors,
    protocol_manifest,
)


class StrictTrainingImages(Dataset[Tensor]):
    """Load the registered paths without silently replacing unreadable data."""

    def __init__(self, selected: ShareGPT4VImageTrainingDataset) -> None:
        self.image_paths = tuple(selected.image_paths)
        self.preprocess = selected.preprocess
        self.data_seed = selected.data_seed
        self.ordered_sha256 = selected.ordered_sha256

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.image_paths[index]
        try:
            with Image.open(path) as image:
                return self.preprocess(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"failed to decode registered training image: {path}") from error


def default_output_dir(expert_count: int, phase: str) -> Path:
    return OUTPUT_ROOT / "training" / f"n{expert_count}" / phase


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    phase: str = "stage2",
) -> argparse.Namespace:
    if phase not in ("stage1", "stage2"):
        raise ValueError("phase must be stage1 or stage2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert-count",
        type=int,
        choices=EXPERT_COUNTS,
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--annotations", type=Path, default=TRAIN_ANNOTATIONS)
    parser.add_argument("--image-root", type=Path, default=TRAIN_IMAGES)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    args.phase = phase
    args.expert_count = int(args.expert_count)
    args.output_dir = args.output_dir or default_output_dir(args.expert_count, phase)
    if phase == "stage2" and args.stage1_checkpoint is None:
        args.stage1_checkpoint = default_output_dir(args.expert_count, "stage1") / "best.pt"
    if phase == "stage1" and args.stage1_checkpoint is not None:
        parser.error("--stage1-checkpoint is valid only for Stage 2")
    args.seed = RUN_SEED
    args.data_seed = DATA_SEED
    args.max_samples = TRAINING_POOL_SIZE
    args.steps = TRAIN_STEPS
    args.batch_size = STAGE1_BATCH_SIZE if phase == "stage1" else STAGE2_BATCH_SIZE
    args.num_workers = NUM_WORKERS
    args.target_ratio = TARGET_RATIO
    args.capacity_factors = capacity_factors(args.expert_count)
    args.learning_rate = 1e-3 if phase == "stage1" else 3e-4
    args.weight_decay = 0.05
    args.temperature = 0.4
    args.router_warmup = 1_000
    if phase == "stage1":
        args.budget_weight = 50.0
        args.separation_weight = 1.0
        args.best_budget_loss = 0.01
    args.log_every = 100
    args.save_every = 0
    args.random_evals = 3
    return args


def validate_paths(args: argparse.Namespace, *, require_stage1: bool) -> None:
    for path, label in (
        (args.pretrained, "Dense CLIP ViT-L/14 weights"),
        (args.annotations, "ShareGPT4V annotations"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"missing ShareGPT4V image root: {args.image_root}")
    if file_sha256(args.pretrained) != PRETRAINED_SHA256:
        raise ValueError("Dense CLIP weights differ from the registered experiment")
    if require_stage1:
        if args.stage1_checkpoint is None or not args.stage1_checkpoint.is_file():
            raise FileNotFoundError(
                f"missing Stage-1 subspace checkpoint: {args.stage1_checkpoint}"
            )


def build_dataset(args: argparse.Namespace, preprocess: Any) -> StrictTrainingImages:
    selected = ShareGPT4VImageTrainingDataset(
        args.annotations,
        args.image_root,
        preprocess,
        max_samples=TRAINING_POOL_SIZE,
        data_seed=DATA_SEED,
    )
    if len(selected) != TRAINING_POOL_SIZE or selected.ordered_sha256 != TRAINING_POOL_SHA256:
        raise RuntimeError(
            "training pool differs from the visual main experiment: "
            f"samples={len(selected)}, sha256={selected.ordered_sha256}; "
            f"expected samples={TRAINING_POOL_SIZE}, sha256={TRAINING_POOL_SHA256}"
        )
    return StrictTrainingImages(selected)


def run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = protocol_manifest()
    manifest.update(
        phase=args.phase,
        expert_count=args.expert_count,
        selected_capacity_factors=list(args.capacity_factors),
        selected_training_protocol=protocol_for_stage(1 if args.phase == "stage1" else 2),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pretrained=str(args.pretrained.resolve()),
        annotations=str(args.annotations.resolve()),
        image_root=str(args.image_root.resolve()),
        output_dir=str(args.output_dir.resolve()),
        stage1_checkpoint=(
            str(args.stage1_checkpoint.resolve())
            if args.stage1_checkpoint is not None
            else None
        ),
    )
    return manifest


def build_model_and_dataset(
    args: argparse.Namespace, device: torch.device
) -> tuple[CLIPSparMoE, StrictTrainingImages]:
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("open_clip_torch is required for this experiment") from error
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(args.pretrained),
        force_quick_gelu=True,
    )
    model = build_model(
        clip_model.to(device).eval(),
        modality=MODALITY,
        stage=1 if args.phase == "stage1" else 2,
        target_ratio=TARGET_RATIO,
        levels=args.capacity_factors,
        tau=args.temperature,
    ).to(device)
    return model, build_dataset(args, preprocess)


def make_loader(
    dataset: StrictTrainingImages,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader[Tensor]:
    generator = torch.Generator().manual_seed(RUN_SEED)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def checkpoint_payload(
    model: CLIPSparMoE,
    args: argparse.Namespace,
    step: int,
    data_identity: dict[str, Any],
    metrics: dict[str, Any],
    stage1_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    is_stage1 = args.phase == "stage1"
    payload: dict[str, Any] = {
        "format_version": 3,
        "method": "sparmoe_vl_routing_granularity",
        "study": STUDY_NAME,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "modality": MODALITY,
        "stage": 1 if is_stage1 else 2,
        "training_protocol": protocol_for_stage(1 if is_stage1 else 2),
        "expert_count": args.expert_count,
        "training_seed": RUN_SEED,
        "target_ratio": TARGET_RATIO,
        "capacity_factors": list(args.capacity_factors),
        "sparse_layers": list(range(24)),
        "step": step,
        "dataset": data_identity,
        "controller": controller_state(model),
        "metrics": metrics,
        "train_args": plain_args(args),
    }
    if stage1_metadata is not None:
        payload["stage1"] = {
            **stage1_metadata,
            "checkpoint": str(args.stage1_checkpoint.resolve()),
        }
    return payload


@torch.inference_mode()
def nested_batch_cosines(
    model: CLIPSparMoE,
    images: Tensor,
    dense: Tensor,
    random_evals: int,
) -> tuple[float, float]:
    model.eval()
    model.set_routing_mode("learned")
    learned, _ = model.encode_sparse(images)
    learned_cosine = float(F.cosine_similarity(learned.float(), dense.float()).mean())
    random_cosine = 0.0
    for _ in range(random_evals):
        model.set_routing_mode("random")
        random_features, _ = model.encode_sparse(images)
        random_cosine += (
            float(F.cosine_similarity(random_features.float(), dense.float()).mean())
            / random_evals
        )
    return learned_cosine, random_cosine


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    require_stage1 = args.phase == "stage2" and not args.check_only
    validate_paths(args, require_stage1=require_stage1)
    if args.check_only:
        dataset = build_dataset(args, lambda image: image)
        result = {
            **run_manifest(args),
            "validated_samples": len(dataset),
            "validated_ordered_sha256": dataset.ordered_sha256,
        }
        print(json.dumps(result, indent=2))
        return result

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_reproducible_seed(RUN_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output_dir / "train.log")
    model, dataset = build_model_and_dataset(args, device)
    identity = dataset_metadata(
        args.annotations,
        DATA_SEED,
        len(dataset),
        dataset.ordered_sha256,
        args.image_root,
    )
    stage1_info = None
    if args.phase == "stage2":
        stage1_sha = file_sha256(args.stage1_checkpoint)
        stage1 = torch_load(args.stage1_checkpoint)
        stage1_info = validate_stage1_metadata(
            stage1,
            args.expert_count,
            checkpoint_sha256=stage1_sha,
        )
        ratios = initialize_stage2(model, stage1)
        logger(
            "stage_transition=validated same_data=true "
            f"ordered_sha256={TRAINING_POOL_SHA256} initial_layer_ratios={ratios}"
        )
        del stage1

    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loader = make_loader(dataset, args, device)
    iterator = iter(loader)
    best_cosine = float("-inf")
    best_step: int | None = None
    last_metrics: dict[str, Any] = {}
    selected_protocol = protocol_for_stage(1 if args.phase == "stage1" else 2)
    logger(
        f"study={STUDY_NAME} phase={args.phase} N={args.expert_count} "
        f"protocol={selected_protocol} "
        f"seed={RUN_SEED} data_seed={DATA_SEED} samples={len(dataset)} "
        f"ordered_sha256={dataset.ordered_sha256} batch={args.batch_size} "
        f"levels={list(args.capacity_factors)} trainable="
        f"{sum(parameter.numel() for parameter in parameters):,}"
    )
    for step in tqdm(
        range(1, TRAIN_STEPS + 1), desc=f"routing granularity N={args.expert_count}"
    ):
        try:
            images = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images = next(iterator)
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            dense = model.dense_features(images)
        model.train()
        if args.phase == "stage2":
            model.set_routing_mode("learned")
        sparse, auxiliary = model.encode_sparse(images)
        if args.phase == "stage1":
            loss, last_metrics = model.stage1_loss(sparse, dense, auxiliary)
            budget_loss = last_metrics["Rp"]
        else:
            router_weight = min(1.0, step / max(args.router_warmup, 1))
            loss, last_metrics = model.stage2_loss(
                sparse,
                dense,
                auxiliary,
                router_weight,
            )
            budget_loss = last_metrics["inherited_budget_error"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            if args.phase == "stage1":
                learned_cosine = float(
                    F.cosine_similarity(sparse.detach().float(), dense.float()).mean()
                )
                random_cosine = None
            else:
                learned_cosine, random_cosine = nested_batch_cosines(
                    model,
                    images,
                    dense,
                    args.random_evals,
                )
            last_metrics["learned_cosine"] = learned_cosine
            if random_cosine is not None:
                last_metrics["random_cosine"] = random_cosine
            budget_is_valid = (
                budget_loss < args.best_budget_loss if args.phase == "stage1" else True
            )
            if budget_is_valid and learned_cosine > best_cosine:
                best_cosine = learned_cosine
                best_step = step
                save_checkpoint_atomic(
                    args.output_dir / "best.pt",
                    checkpoint_payload(
                        model,
                        args,
                        step,
                        identity,
                        last_metrics,
                        stage1_info,
                    ),
                )
            logger(
                f"step={step} loss={last_metrics['total']:.5f} "
                f"learned_cos={learned_cosine:.6f} "
                f"structure_budget_error={budget_loss:.6f} "
                f"best_step={best_step}"
            )
            if args.phase == "stage2":
                model.train()

    save_checkpoint_atomic(
        args.output_dir / "final.pt",
        checkpoint_payload(
            model,
            args,
            TRAIN_STEPS,
            identity,
            last_metrics,
            stage1_info,
        ),
    )
    if best_step is None:
        raise RuntimeError(
            f"no Stage-{args.phase[-1]} checkpoint satisfied the global budget criterion"
        )
    logger(f"finished best_step={best_step} best_cosine={best_cosine:.6f}")
    return None


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    phase: str = "stage2",
) -> None:
    run(parse_args(argv, phase=phase))

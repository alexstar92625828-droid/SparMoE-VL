"""Full-pool MoPE structure selection and visual-FFN utilities."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import (
    DATA_SEED,
    D_FFN,
    D_MODEL,
    EMBED_DIM,
    EXPECTED_IMAGE_POOL_SHA256,
    EXPECTED_PRETRAINED_SHA256,
    EXPECTED_PROCESSING_ORDER_SHA256,
    EXPECTED_VISION_CAPTION_POOL_SHA256,
    N_LAYERS,
    N_TOKENS,
    POOL_SIZE,
    PROJECT_ROOT,
    atomic_torch_save,
    full_pool_permutation,
    make_image_slice_loader,
    tensor_sha256,
    visual_blocks,
)


METHOD = "MoPE-CLIP-FFN (Vision, full-pool two-stage recovery)"
CHECKPOINT_METHOD = "MoPE-CLIP Vision-FFN"
PAPER = "https://arxiv.org/abs/2403.07839"

GROUP_SIZE = 64
NUM_GROUPS = D_FFN // GROUP_SIZE
KEEP_GROUPS = 41
RETAINED_WIDTH = KEEP_GROUPS * GROUP_SIZE
STRUCTURE_SEED = 42
SELECTION_BATCH_SIZE = 32
WORLD_SIZE = 8
PER_GPU_BATCH_SIZE = 32
GLOBAL_BATCH_SIZE = WORLD_SIZE * PER_GPU_BATCH_SIZE
STEPS_PER_STAGE = math.ceil(POOL_SIZE / GLOBAL_BATCH_SIZE)
STAGE_EXPOSURES = POOL_SIZE
TOTAL_EXPOSURES = 2 * STAGE_EXPOSURES
TARGET_FFN_REDUCTION = 1.0 - RETAINED_WIDTH / D_FFN

DENSE_VISUAL_PARAMETERS = 303_966_208
DENSE_TOTAL_MACS_G = 81.012768768
DENSE_FFN_MACS_G = 51.740934144
NON_FFN_MACS_G = DENSE_TOTAL_MACS_G - DENSE_FFN_MACS_G

TEXT_FEATURE_CACHE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "sharegpt4v_visual_pool_clip_vitl14_text_features_data_seed42_500k.pt"
)


def structure_data_manifest() -> dict[str, Any]:
    """The complete visual-main pool used for Taylor and MoPE group scores."""

    return {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "unique_samples": POOL_SIZE,
        "selection_samples": POOL_SIZE,
        "uses_complete_main_pool": True,
        "held_out_samples": 0,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "paired_caption_sha256": EXPECTED_VISION_CAPTION_POOL_SHA256,
        "processing_seed": STRUCTURE_SEED,
        "processing_order_sha256": EXPECTED_PROCESSING_ORDER_SHA256[STRUCTURE_SEED],
        "batch_size": SELECTION_BATCH_SIZE,
        "batches": POOL_SIZE // SELECTION_BATCH_SIZE,
    }


def validate_structure_data_manifest(manifest: Mapping[str, Any]) -> None:
    expected = structure_data_manifest()
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"MoPE selection data differs from the visual main pool: {mismatches}")


def recovery_data_manifest(seed: int) -> dict[str, Any]:
    """Describe identical complete-pool Stage-1 and Stage-2 sequences."""

    order = full_pool_permutation(seed)
    manifest = {
        "run_seed": int(seed),
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "unique_samples": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "paired_caption_sha256": EXPECTED_VISION_CAPTION_POOL_SHA256,
        "stage1_exposures": STAGE_EXPOSURES,
        "stage2_exposures": STAGE_EXPOSURES,
        "stage_sequences_identical": True,
        "stage_processing_order_sha256": tensor_sha256(order),
        "total_exposures": TOTAL_EXPOSURES,
        "world_size": WORLD_SIZE,
        "per_gpu_batch_size": PER_GPU_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "steps_per_stage": STEPS_PER_STAGE,
        "final_global_batch_size": POOL_SIZE - (STEPS_PER_STAGE - 1) * GLOBAL_BATCH_SIZE,
    }
    validate_recovery_data_manifest(manifest)
    return manifest


def validate_recovery_data_manifest(manifest: Mapping[str, Any]) -> None:
    seed = manifest.get("run_seed")
    expected = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "unique_samples": POOL_SIZE,
        "uses_complete_main_pool": True,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "paired_caption_sha256": EXPECTED_VISION_CAPTION_POOL_SHA256,
        "stage1_exposures": STAGE_EXPOSURES,
        "stage2_exposures": STAGE_EXPOSURES,
        "stage_sequences_identical": True,
        "total_exposures": TOTAL_EXPOSURES,
        "world_size": WORLD_SIZE,
        "per_gpu_batch_size": PER_GPU_BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "steps_per_stage": STEPS_PER_STAGE,
        "final_global_batch_size": 32,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if seed not in EXPECTED_PROCESSING_ORDER_SHA256:
        mismatches["run_seed"] = (seed, tuple(EXPECTED_PROCESSING_ORDER_SHA256))
    elif (
        manifest.get("stage_processing_order_sha256") != EXPECTED_PROCESSING_ORDER_SHA256[seed]
    ):
        mismatches["stage_processing_order_sha256"] = (
            manifest.get("stage_processing_order_sha256"),
            EXPECTED_PROCESSING_ORDER_SHA256[seed],
        )
    if mismatches:
        raise ValueError(f"MoPE recovery data differs from the visual main pool: {mismatches}")


def recovery_rank_indices(seed: int, rank: int, world_size: int = WORLD_SIZE) -> Tensor:
    if world_size != WORLD_SIZE or not 0 <= rank < world_size:
        raise ValueError("formal MoPE recovery requires ranks 0..7")
    order = full_pool_permutation(seed)
    indices = order[rank::world_size].contiguous()
    if indices.numel() != POOL_SIZE // WORLD_SIZE:
        raise RuntimeError("MoPE rank did not receive exactly one eighth of the main pool")
    return indices


def validate_dense_text_cache(payload: Mapping[str, Any]) -> None:
    expected = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "image_pool_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "paired_caption_sha256": EXPECTED_VISION_CAPTION_POOL_SHA256,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("MoPE text-feature cache metadata differs from the visual main pool")
    features = payload.get("features")
    if (
        not isinstance(features, Tensor)
        or tuple(features.shape) != (POOL_SIZE, EMBED_DIM)
        or features.dtype != torch.float16
        or not torch.isfinite(features).all()
    ):
        raise RuntimeError("MoPE text cache must contain finite [500000, 768] fp16 features")


@torch.no_grad()
def prepare_dense_text_cache(
    model: nn.Module,
    tokenizer: Any,
    captions: Sequence[str],
    device: str,
    cache_path: Path = TEXT_FEATURE_CACHE,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Cache frozen text features paired with all 500k visual-main images."""

    cache_path = Path(cache_path)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid MoPE text-feature cache: {cache_path}")
        validate_dense_text_cache(payload)
        return payload
    if len(captions) != POOL_SIZE or batch_size <= 0:
        raise ValueError("MoPE text caching requires all 500,000 captions")
    torch_device = torch.device(device)
    features = []
    for start in range(0, POOL_SIZE, batch_size):
        stop = min(start + batch_size, POOL_SIZE)
        tokens = tokenizer(captions[start:stop]).to(
            torch_device, non_blocking=torch_device.type == "cuda"
        )
        with torch.autocast(
            device_type=torch_device.type,
            dtype=torch.float16,
            enabled=torch_device.type == "cuda",
        ):
            encoded = model.encode_text(tokens)
        features.append(F.normalize(encoded.float(), dim=-1).half().cpu())
        if (start // batch_size + 1) % 100 == 0:
            print(f"text_features={stop}/{POOL_SIZE}", flush=True)
    payload = {
        "data_seed": DATA_SEED,
        "pool_size": POOL_SIZE,
        "image_pool_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "paired_caption_sha256": EXPECTED_VISION_CAPTION_POOL_SHA256,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "features": torch.cat(features),
    }
    validate_dense_text_cache(payload)
    atomic_torch_save(payload, cache_path)
    return payload


def contrastive_loss(image_features: Tensor, text_features: Tensor, scale: Tensor) -> Tensor:
    logits = scale * image_features @ text_features.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def ffn_taylor_scores(model: nn.Module) -> list[Tensor]:
    scores = []
    for layer, block in enumerate(visual_blocks(model)):
        fc, proj = block.mlp.c_fc, block.mlp.c_proj
        if fc.weight.grad is None or fc.bias.grad is None or proj.weight.grad is None:
            raise RuntimeError(f"missing MoPE FFN gradient at visual layer {layer}")
        incoming = (fc.weight * fc.weight.grad).sum(dim=1) + fc.bias * fc.bias.grad
        outgoing = (proj.weight * proj.weight.grad).sum(dim=0)
        scores.append((incoming.abs() + outgoing.abs()).detach().float().cpu())
    return scores


@torch.no_grad()
def _save_taylor_progress(
    path: Path,
    contract: Mapping[str, Any],
    score_sums: Tensor,
    samples_seen: int,
    next_batch_index: int,
) -> None:
    atomic_torch_save(
        {
            "contract": dict(contract),
            "score_sums": score_sums,
            "samples_seen": samples_seen,
            "next_batch_index": next_batch_index,
        },
        path,
    )


def collect_taylor_scores(
    model: nn.Module,
    preprocess: Any,
    paths: Sequence[Path],
    text_features: Tensor,
    order: Tensor,
    device: str,
    output_dir: Path,
    contract: Mapping[str, Any],
    workers: int = 8,
    save_every: int = 100,
    log_every: int = 100,
    resume: bool = False,
    max_batches: int | None = None,
) -> tuple[Tensor | None, dict[str, Any]]:
    """Accumulate DynaBERT Taylor rewiring scores over all main-pool pairs."""

    validate_structure_data_manifest(contract["data_manifest"])
    if len(paths) != POOL_SIZE or tuple(text_features.shape) != (POOL_SIZE, EMBED_DIM):
        raise ValueError("MoPE Taylor scoring requires all 500,000 paired samples")
    if (
        order.numel() != POOL_SIZE
        or tensor_sha256(order) != contract["data_manifest"]["processing_order_sha256"]
    ):
        raise ValueError("MoPE Taylor processing order differs from its manifest")
    if workers < 0 or save_every <= 0 or log_every <= 0:
        raise ValueError("MoPE Taylor execution parameters are invalid")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("MoPE Taylor diagnostic batch limit must be positive")

    progress_path = Path(output_dir) / "taylor_progress.pt"
    if progress_path.is_file():
        if not resume:
            raise RuntimeError(f"Taylor progress exists at {progress_path}; pass --resume")
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if not isinstance(progress, dict) or progress.get("contract") != dict(contract):
            raise RuntimeError("MoPE Taylor progress belongs to another protocol")
        score_sums = progress["score_sums"]
        samples_seen = int(progress["samples_seen"])
        batch_start = int(progress["next_batch_index"])
    else:
        score_sums = torch.zeros((N_LAYERS, D_FFN), dtype=torch.float64)
        samples_seen = 0
        batch_start = 0
    if tuple(score_sums.shape) != (N_LAYERS, D_FFN):
        raise RuntimeError("invalid MoPE Taylor progress score tensor")
    expected_seen = batch_start * SELECTION_BATCH_SIZE
    if samples_seen != expected_seen or not 0 <= samples_seen <= POOL_SIZE:
        raise RuntimeError("invalid MoPE Taylor progress sample position")

    model.requires_grad_(False)
    scored_parameters = []
    for block in visual_blocks(model):
        for parameter in (block.mlp.c_fc.weight, block.mlp.c_fc.bias, block.mlp.c_proj.weight):
            parameter.requires_grad_(True)
            scored_parameters.append(parameter)
    sample_offset = batch_start * SELECTION_BATCH_SIZE
    loader = make_image_slice_loader(
        paths,
        order[sample_offset:],
        preprocess,
        SELECTION_BATCH_SIZE,
        workers,
        device,
    )
    torch_device = torch.device(device)
    scale = model.logit_scale.exp().detach().clamp(max=100)
    batches_this_call = 0
    started = time.time()
    next_batch = batch_start
    try:
        for offset, images in enumerate(loader):
            batch_index = batch_start + offset
            start = batch_index * SELECTION_BATCH_SIZE
            selection = order[start : start + SELECTION_BATCH_SIZE].to(torch.int64)
            for parameter in scored_parameters:
                parameter.grad = None
            images = images.to(torch_device, non_blocking=torch_device.type == "cuda")
            paired_text = text_features[selection].to(
                torch_device,
                dtype=torch.float32,
                non_blocking=torch_device.type == "cuda",
            )
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=torch_device.type == "cuda",
            ):
                image_features = F.normalize(model.encode_image(images).float(), dim=-1)
                loss = contrastive_loss(image_features, paired_text, scale)
            loss.backward()
            count = int(images.shape[0])
            score_sums += torch.stack(ffn_taylor_scores(model)).double() * count
            samples_seen += count
            batches_this_call += 1
            next_batch = batch_index + 1
            if batches_this_call % save_every == 0:
                _save_taylor_progress(
                    progress_path, contract, score_sums, samples_seen, next_batch
                )
            if batches_this_call % log_every == 0 or samples_seen == POOL_SIZE:
                print(f"taylor_samples={samples_seen}/{POOL_SIZE}", flush=True)
            if max_batches is not None and batches_this_call >= max_batches:
                break
    finally:
        model.requires_grad_(False)
        for parameter in model.parameters():
            parameter.grad = None
    _save_taylor_progress(progress_path, contract, score_sums, samples_seen, next_batch)
    complete = samples_seen == POOL_SIZE and next_batch == POOL_SIZE // SELECTION_BATCH_SIZE
    report = {
        "complete": complete,
        "selected_samples": POOL_SIZE if complete else 0,
        "completed_samples": samples_seen,
        "completed_batches": next_batch,
        "total_batches": POOL_SIZE // SELECTION_BATCH_SIZE,
        "batches_this_call": batches_this_call,
        "elapsed_seconds_this_call": time.time() - started,
    }
    if not complete:
        return None, report
    return (score_sums / POOL_SIZE).float(), report


def rankings_to_groups(scores: Tensor) -> tuple[Tensor, Tensor]:
    if tuple(scores.shape) != (N_LAYERS, D_FFN):
        raise ValueError("MoPE Taylor scores have the wrong shape")
    rankings = torch.argsort(scores.float(), dim=1, descending=True, stable=True)
    return rankings, rankings.reshape(N_LAYERS, NUM_GROUPS, GROUP_SIZE)


def register_group_ablation(
    model: nn.Module,
    groups: Tensor,
    group_index: int,
) -> list[Any]:
    if tuple(groups.shape) != (N_LAYERS, NUM_GROUPS, GROUP_SIZE):
        raise ValueError("invalid MoPE visual ranked groups")
    if not 0 <= group_index < NUM_GROUPS:
        raise ValueError("MoPE visual group index is out of range")
    handles = []
    for layer, block in enumerate(visual_blocks(model)):
        indices_cpu = groups[layer, group_index].clone()

        def hook(_module: nn.Module, _inputs: Any, output: Tensor, indices=indices_cpu):
            masked = output.clone()
            masked.index_fill_(-1, indices.to(output.device), 0)
            return masked

        handles.append(block.mlp.gelu.register_forward_hook(hook))
    return handles


def recall_counts(image_features: Tensor, text_features: Tensor) -> Tensor:
    if image_features.shape != text_features.shape or image_features.ndim != 2:
        raise ValueError("MoPE paired features must have identical [batch, dim] shapes")
    similarity = image_features.float() @ text_features.float().T
    targets = torch.arange(similarity.shape[0])
    order = similarity.argsort(dim=1, descending=True)
    return torch.tensor(
        [int((order[:, :k] == targets[:, None]).any(dim=1).sum()) for k in (1, 5, 10)],
        dtype=torch.int64,
    )


def recall_percentages(counts: Tensor, samples: int) -> dict[str, float]:
    if tuple(counts.shape) != (3,) or samples <= 0:
        raise ValueError("MoPE recall totals are invalid")
    values = 100.0 * counts.double() / samples
    return {
        "r1": float(values[0]),
        "r5": float(values[1]),
        "r10": float(values[2]),
        "mean": float(values.mean()),
    }


@torch.inference_mode()
def full_pool_paired_recall(
    model: nn.Module,
    preprocess: Any,
    paths: Sequence[Path],
    text_features: Tensor,
    order: Tensor,
    device: str,
    workers: int,
    groups: Tensor | None = None,
    group_index: int | None = None,
) -> dict[str, float]:
    """Measure batch-local paired recall across every main-pool sample."""

    if len(paths) != POOL_SIZE or tuple(text_features.shape) != (POOL_SIZE, EMBED_DIM):
        raise ValueError("MoPE recall requires all 500,000 paired samples")
    if (
        order.numel() != POOL_SIZE
        or tensor_sha256(order) != EXPECTED_PROCESSING_ORDER_SHA256[STRUCTURE_SEED]
    ):
        raise ValueError("MoPE recall order differs from the complete selection sequence")
    handles = []
    if groups is not None:
        if group_index is None:
            raise ValueError("MoPE ablation requires a group index")
        handles = register_group_ablation(model, groups, group_index)
    loader = make_image_slice_loader(
        paths,
        order,
        preprocess,
        SELECTION_BATCH_SIZE,
        workers,
        device,
    )
    counts = torch.zeros(3, dtype=torch.int64)
    samples = 0
    torch_device = torch.device(device)
    try:
        for batch_index, images in enumerate(loader):
            start = batch_index * SELECTION_BATCH_SIZE
            selection = order[start : start + images.shape[0]].to(torch.int64)
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=torch_device.type == "cuda",
            ):
                encoded = model.encode_image(
                    images.to(torch_device, non_blocking=torch_device.type == "cuda")
                )
            image_features = F.normalize(encoded.float(), dim=-1).cpu()
            counts += recall_counts(image_features, text_features[selection])
            samples += int(images.shape[0])
    finally:
        for handle in handles:
            handle.remove()
    if samples != POOL_SIZE:
        raise RuntimeError("MoPE recall did not process the complete visual main pool")
    return recall_percentages(counts, samples)


def validate_selection(selection: Mapping[str, Any]) -> None:
    expected = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": "structure_selection",
        "structure_seed": STRUCTURE_SEED,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "group_size": GROUP_SIZE,
        "num_groups": NUM_GROUPS,
        "complete": True,
    }
    mismatches = {
        key: (selection.get(key), value)
        for key, value in expected.items()
        if selection.get(key) != value
    }
    try:
        validate_structure_data_manifest(selection.get("data_manifest", {}))
    except ValueError as error:
        mismatches["data_manifest"] = (str(error), "complete visual main pool")
    groups = selection.get("groups")
    rankings = selection.get("rankings")
    taylor_scores = selection.get("taylor_scores")
    priority = selection.get("group_priority")
    records = selection.get("group_records")
    if not isinstance(taylor_scores, Tensor) or tuple(taylor_scores.shape) != (
        N_LAYERS,
        D_FFN,
    ):
        mismatches["taylor_scores"] = (
            getattr(taylor_scores, "shape", None),
            f"[{N_LAYERS},{D_FFN}]",
        )
    elif not torch.isfinite(taylor_scores).all():
        mismatches["taylor_scores"] = ("non-finite", "finite")
    if not isinstance(rankings, Tensor) or tuple(rankings.shape) != (N_LAYERS, D_FFN):
        mismatches["rankings"] = (getattr(rankings, "shape", None), f"[{N_LAYERS},{D_FFN}]")
    if not isinstance(groups, Tensor) or tuple(groups.shape) != (
        N_LAYERS,
        NUM_GROUPS,
        GROUP_SIZE,
    ):
        mismatches["groups"] = (getattr(groups, "shape", None), "[24,64,64]")
    if isinstance(rankings, Tensor) and tuple(rankings.shape) == (N_LAYERS, D_FFN):
        expected_channels = torch.arange(D_FFN)
        if not all(
            torch.equal(torch.sort(row).values.cpu(), expected_channels) for row in rankings
        ):
            mismatches["rankings"] = ("not per-layer permutations", "permutations of 0..4095")
        elif isinstance(taylor_scores, Tensor) and tuple(taylor_scores.shape) == (
            N_LAYERS,
            D_FFN,
        ):
            expected_rankings = torch.argsort(
                taylor_scores.float(), dim=1, descending=True, stable=True
            )
            if not torch.equal(rankings.cpu(), expected_rankings.cpu()):
                mismatches["rankings"] = ("inconsistent", "stable Taylor-score ranking")
    if (
        isinstance(groups, Tensor)
        and tuple(groups.shape) == (N_LAYERS, NUM_GROUPS, GROUP_SIZE)
        and isinstance(rankings, Tensor)
        and tuple(rankings.shape) == (N_LAYERS, D_FFN)
        and not torch.equal(groups.reshape(N_LAYERS, D_FFN).cpu(), rankings.cpu())
    ):
        mismatches["groups"] = ("inconsistent", "reshaped rankings")
    records_valid = False
    if not isinstance(records, list) or len(records) != NUM_GROUPS:
        mismatches["group_records"] = (
            len(records) if isinstance(records, list) else type(records).__name__,
            NUM_GROUPS,
        )
    else:
        record_groups = [record.get("group") for record in records if isinstance(record, dict)]
        finite_scores = all(
            isinstance(record, dict)
            and isinstance(record.get("mope"), (int, float))
            and math.isfinite(float(record["mope"]))
            for record in records
        )
        if record_groups != list(range(NUM_GROUPS)) or not finite_scores:
            mismatches["group_records"] = (
                "invalid indices or scores",
                "64 ordered finite MoPE group scores",
            )
        else:
            records_valid = True
    if not isinstance(priority, list) or sorted(priority) != list(range(NUM_GROUPS)):
        mismatches["group_priority"] = (priority, "permutation of 0..63")
    elif records_valid:
        expected_priority = [
            int(record["group"])
            for record in sorted(records, key=lambda item: (-item["mope"], item["group"]))
        ]
        if priority != expected_priority:
            mismatches["group_priority"] = (priority, "descending MoPE group score")
    if mismatches:
        raise ValueError(f"invalid MoPE visual structure selection: {mismatches}")


def selected_groups(selection: Mapping[str, Any]) -> list[int]:
    validate_selection(selection)
    return sorted(int(index) for index in selection["group_priority"][:KEEP_GROUPS])


def kept_channel_indices(groups: Tensor, kept_groups: Sequence[int]) -> list[Tensor]:
    if tuple(groups.shape) != (N_LAYERS, NUM_GROUPS, GROUP_SIZE):
        raise ValueError("MoPE visual groups have the wrong shape")
    kept = sorted(int(index) for index in kept_groups)
    if len(kept) != KEEP_GROUPS or len(set(kept)) != len(kept):
        raise ValueError(f"MoPE visual structure must retain exactly {KEEP_GROUPS} groups")
    if kept[0] < 0 or kept[-1] >= NUM_GROUPS:
        raise ValueError("MoPE retained visual group is out of range")
    return [groups[layer, kept].reshape(-1).clone() for layer in range(N_LAYERS)]


@torch.no_grad()
def structurally_prune_visual_ffn(
    model: nn.Module,
    kept_indices: Sequence[Tensor],
) -> None:
    if len(kept_indices) != N_LAYERS:
        raise ValueError(f"expected {N_LAYERS} MoPE retained-index tensors")
    for layer, (block, indices_cpu) in enumerate(zip(visual_blocks(model), kept_indices)):
        old_fc, old_proj = block.mlp.c_fc, block.mlp.c_proj
        indices = indices_cpu.to(old_fc.weight.device, dtype=torch.long)
        if (
            indices.numel() != RETAINED_WIDTH
            or indices.unique().numel() != indices.numel()
            or int(indices.min()) < 0
            or int(indices.max()) >= D_FFN
        ):
            raise ValueError(f"invalid MoPE retained channels at visual layer {layer}")
        new_fc = nn.Linear(
            D_MODEL,
            RETAINED_WIDTH,
            bias=old_fc.bias is not None,
            device=old_fc.weight.device,
            dtype=old_fc.weight.dtype,
        )
        new_proj = nn.Linear(
            RETAINED_WIDTH,
            D_MODEL,
            bias=old_proj.bias is not None,
            device=old_proj.weight.device,
            dtype=old_proj.weight.dtype,
        )
        new_fc.weight.copy_(old_fc.weight.index_select(0, indices))
        if old_fc.bias is not None:
            new_fc.bias.copy_(old_fc.bias.index_select(0, indices))
        new_proj.weight.copy_(old_proj.weight.index_select(1, indices))
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)
        block.mlp.c_fc, block.mlp.c_proj = new_fc, new_proj


def validate_pruned_visual(model: nn.Module) -> None:
    for layer, block in enumerate(visual_blocks(model)):
        if tuple(block.mlp.c_fc.weight.shape) != (RETAINED_WIDTH, D_MODEL):
            raise ValueError(f"incorrect MoPE c_fc shape at visual layer {layer}")
        if tuple(block.mlp.c_proj.weight.shape) != (D_MODEL, RETAINED_WIDTH):
            raise ValueError(f"incorrect MoPE c_proj shape at visual layer {layer}")


def visual_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in model.visual.state_dict().items()}


def load_visual_state_dict(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    model.visual.load_state_dict(state, strict=True)


class HiddenCapture:
    """Capture all visual Transformer-block outputs for recovery distillation."""

    def __init__(self, model: nn.Module) -> None:
        self.outputs: list[Tensor] = []
        self.handles = [
            block.register_forward_hook(self._hook) for block in visual_blocks(model)
        ]

    def _hook(self, _module: nn.Module, _inputs: Any, output: Tensor) -> None:
        self.outputs.append(output)

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def soft_cross_entropy(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    teacher_probability = F.softmax(teacher_logits.float(), dim=-1)
    student_log_probability = F.log_softmax(student_logits.float(), dim=-1)
    return -(teacher_probability * student_log_probability).sum(dim=-1).mean()


def cosine_warmup_lambda(step: int, total_steps: int = 2 * STEPS_PER_STAGE) -> float:
    warmup_steps = max(1, total_steps // 10)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def statistics() -> dict[str, float | int]:
    removed_per_layer = D_FFN - RETAINED_WIDTH
    removed = N_LAYERS * removed_per_layer
    ffn_macs = N_LAYERS * 2.0 * N_TOKENS * D_MODEL * RETAINED_WIDTH / 1e9
    return {
        "retained_width": RETAINED_WIDTH,
        "hidden_size_sum": N_LAYERS * RETAINED_WIDTH,
        "removed_channels": removed,
        "active_visual_parameters_m": (DENSE_VISUAL_PARAMETERS - removed * (2 * D_MODEL + 1))
        / 1e6,
        "visual_total_macs_g": NON_FFN_MACS_G + ffn_macs,
        "visual_ffn_macs_g": ffn_macs,
        "dense_visual_ffn_macs_g": DENSE_FFN_MACS_G,
        "ffn_reduction_percent": 100.0 * TARGET_FFN_REDUCTION,
    }


def validate_final_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    seed = checkpoint.get("seed")
    if seed not in EXPECTED_PROCESSING_ORDER_SHA256:
        raise ValueError("MoPE visual checkpoint has an unsupported paper seed")
    expected = {
        "format_version": 1,
        "method": CHECKPOINT_METHOD,
        "stage": 2,
        "stage_step": STEPS_PER_STAGE,
        "global_step": 2 * STEPS_PER_STAGE,
        "retained_width": RETAINED_WIDTH,
        "target_ffn_reduction": TARGET_FFN_REDUCTION,
        "actual_ffn_reduction": TARGET_FFN_REDUCTION,
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "complete": True,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    try:
        validate_recovery_data_manifest(checkpoint.get("data_manifest", {}))
    except ValueError as error:
        mismatches["data_manifest"] = (str(error), "complete two-stage visual pool")
    if checkpoint.get("data_manifest", {}).get("run_seed") != seed:
        mismatches["seed"] = (seed, checkpoint.get("data_manifest", {}).get("run_seed"))
    selection_sha256 = checkpoint.get("selection_sha256")
    if (
        not isinstance(selection_sha256, str)
        or len(selection_sha256) != 64
        or any(character not in "0123456789abcdef" for character in selection_sha256)
    ):
        mismatches["selection_sha256"] = (selection_sha256, "lowercase SHA-256")
    kept_groups = checkpoint.get("kept_group_indices")
    if (
        not isinstance(kept_groups, list)
        or len(kept_groups) != KEEP_GROUPS
        or sorted(set(kept_groups)) != kept_groups
        or any(
            not isinstance(index, int) or not 0 <= index < NUM_GROUPS for index in kept_groups
        )
    ):
        mismatches["kept_group_indices"] = (kept_groups, f"{KEEP_GROUPS} unique sorted groups")
    if checkpoint.get("statistics") != statistics():
        mismatches["statistics"] = (checkpoint.get("statistics"), statistics())
    state = checkpoint.get("visual_state_dict")
    if not isinstance(state, Mapping):
        mismatches["visual_state_dict"] = (type(state).__name__, "mapping")
    if mismatches:
        raise ValueError(f"invalid final MoPE visual checkpoint: {mismatches}")

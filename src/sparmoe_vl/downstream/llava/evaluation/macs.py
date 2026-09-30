"""Measure paper-exact visual FFN MACs on the 500 unique POPE images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data import DEFAULT_POPE_ROOT, collect_pope_images
from ..protocol import (
    BENCHMARK_COUNTS,
    BENCHMARK_IDENTITIES,
    DEFAULT_CLIP,
    DENSE_VISUAL_FFN_MACS_G,
    STUDY_NAME,
    inspect_checkpoint,
)
from ..vision import load_sparse_tower
from .common import save_json


class _ImageDataset(Dataset):
    def __init__(self, paths: Sequence[Path], processor: Any) -> None:
        self.paths = tuple(paths)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt")[
                "pixel_values"
            ][0]
        return pixel_values


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pope-root", type=Path, default=DEFAULT_POPE_ROOT)
    parser.add_argument("--clip-path", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


@torch.no_grad()
def measure_sparse(model: Any, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    layer_ratio_sum = torch.zeros(len(model.layers), dtype=torch.float64)
    image_count = 0
    patch_count = (int(model.config.image_size) // int(model.config.patch_size)) ** 2
    for pixel_values in tqdm(loader, desc="measure sparse visual FFN MACs"):
        pixel_values = pixel_values.to(device, non_blocking=device.type == "cuda")
        output = model(pixel_values, output_hidden_states=False)
        batch_size = pixel_values.shape[0]
        image_count += batch_size
        for index, layer in enumerate(output.layers):
            ratios = layer.sparse_pattern.retention_ratios.float()
            token_ratios = (layer.routing.gates.float() * ratios).sum(dim=-1)
            layer_ratio_sum[index] += token_ratios.mean().cpu().double() * batch_size
    layer_ratios = (layer_ratio_sum / max(image_count, 1)).tolist()
    per_token_macs = 2 * int(model.config.hidden_size) * int(model.config.intermediate_size)
    sparse_macs_g = (
        sum((1.0 + patch_count * ratio) * per_token_macs for ratio in layer_ratios) / 1e9
    )
    return {
        "dense_visual_ffn_macs_g": DENSE_VISUAL_FFN_MACS_G,
        "sparse_visual_ffn_macs_g": sparse_macs_g,
        "visual_ffn_delta_pct": 100.0 * (sparse_macs_g / DENSE_VISUAL_FFN_MACS_G - 1.0),
        "average_patch_retention": sum(layer_ratios) / len(layer_ratios),
        "layer_patch_retention": layer_ratios,
        "num_images": image_count,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    for path, label in (
        (args.clip_path / "config.json", "CLIP-336 config"),
        (args.clip_path / "pytorch_model.bin", "CLIP-336 weights"),
        (args.checkpoint, "sparse Stage-2 checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    checkpoint = inspect_checkpoint(args.checkpoint, expected_stage=2)
    image_paths = collect_pope_images(args.pope_root)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("max images must be positive")
        image_paths = image_paths[: args.max_images]
    header = {
        "format_version": 1,
        "study": STUDY_NAME,
        "benchmark": "visual_ffn_macs",
        "mode": "sparse",
        "checkpoint": checkpoint,
        "data_identity": {
            key: BENCHMARK_IDENTITIES[key]
            for key in (
                "pope_random_sha256",
                "pope_popular_sha256",
                "pope_adversarial_sha256",
            )
        },
        "counts": {"unique_images": len(image_paths)},
        "measurement": {
            "num_layers": 24,
            "num_tokens": 577,
            "num_patch_tokens": 576,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "class_token_routing": "dense",
            "patch_token_routing": "learned_argmax",
            "batch_size": args.batch_size,
        },
    }
    if args.check_only:
        print(json.dumps(header, indent=2, ensure_ascii=True))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    try:
        from transformers import CLIPImageProcessor
    except ImportError as error:
        raise RuntimeError("install the llava optional dependencies") from error
    processor = CLIPImageProcessor.from_pretrained(str(args.clip_path), local_files_only=True)
    model, runtime_checkpoint = load_sparse_tower(args.checkpoint, args.clip_path, device)
    if runtime_checkpoint != checkpoint:
        raise RuntimeError("checkpoint changed between validation and model loading")
    loader = DataLoader(
        _ImageDataset(image_paths, processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = measure_sparse(model, loader, device)
    if (
        args.max_images is None
        and metrics["num_images"] != BENCHMARK_COUNTS["pope"]["unique_images"]
    ):
        raise RuntimeError("the MAC measurement is incomplete")
    result = {**header, "metrics": metrics}
    save_json(result, args.output_dir / "evaluation.json")
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "training_seed",
                "dense_visual_ffn_macs_g",
                "sparse_visual_ffn_macs_g",
                "visual_ffn_delta_pct",
                "average_patch_retention",
                "num_images",
            ]
        )
        writer.writerow(
            [
                checkpoint["training_seed"],
                metrics["dense_visual_ffn_macs_g"],
                metrics["sparse_visual_ffn_macs_g"],
                metrics["visual_ffn_delta_pct"],
                metrics["average_patch_retention"],
                metrics["num_images"],
            ]
        )
    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

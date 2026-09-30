"""LLaVA-v1.5 runtime with interchangeable Dense and SparMoE vision towers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import Tensor, nn

from .protocol import DEFAULT_CLIP, DEFAULT_LLAVA, DEFAULT_TOKENIZER, LLAVA_NAME
from .vision import load_local_clip_vision, load_sparse_tower


class DenseCLIP336VisionTower(nn.Module):
    def __init__(self, clip_path: str | Path) -> None:
        super().__init__()
        self.clip = load_local_clip_vision(clip_path)
        self.config = self.clip.config
        self.patch_size = self.config.patch_size

    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        del output_hidden_states, return_dict
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        output = self.clip(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        dtype = pixel_values.dtype
        return BaseModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state.to(dtype=dtype),
            pooler_output=output.pooler_output.to(dtype=dtype),
            hidden_states=tuple(state.to(dtype=dtype) for state in output.hidden_states),
        )


class SparseCLIP336VisionTower(nn.Module):
    def __init__(
        self,
        clip_path: str | Path,
        checkpoint_path: str | Path,
    ) -> None:
        super().__init__()
        self.sparse, self.checkpoint_metadata = load_sparse_tower(
            checkpoint_path,
            clip_path,
            "cpu",
        )
        self.config = self.sparse.config
        self.patch_size = self.config.patch_size

    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        del return_dict
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        output = self.sparse(
            pixel_values,
            routing_mode="learned",
            output_hidden_states=output_hidden_states,
        )
        dtype = pixel_values.dtype
        return BaseModelOutputWithPooling(
            last_hidden_state=output.last_hidden_state.to(dtype=dtype),
            pooler_output=output.pooler_output.to(dtype=dtype),
            hidden_states=tuple(state.to(dtype=dtype) for state in output.hidden_states),
        )


def build_input_ids(
    tokenizer: Any,
    question: str,
    image_token_id: int,
    image_sequence_length: int,
) -> Tensor:
    prefix = tokenizer("USER: ", add_special_tokens=True).input_ids
    suffix = tokenizer(
        f"\n{question}\nASSISTANT:",
        add_special_tokens=False,
    ).input_ids
    values = list(prefix) + [int(image_token_id)] * image_sequence_length + list(suffix)
    return torch.tensor(values, dtype=torch.long)


def build_batch_inputs(
    tokenizer: Any,
    questions: Sequence[str],
    image_token_id: int,
    image_sequence_length: int,
) -> tuple[Tensor, Tensor]:
    sequences = [
        build_input_ids(
            tokenizer,
            question,
            image_token_id,
            image_sequence_length,
        )
        for question in questions
    ]
    if not sequences:
        raise ValueError("questions must not be empty")
    maximum_length = max(sequence.numel() for sequence in sequences)
    padding_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full(
        (len(sequences), maximum_length),
        padding_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, -sequence.numel() :] = sequence
        attention_mask[row, -sequence.numel() :] = 1
    return input_ids, attention_mask


def add_runtime_arguments(parser: argparse.ArgumentParser, *, sparse_only: bool) -> None:
    if not sparse_only:
        parser.add_argument("--mode", choices=("dense", "sparse"), required=True)
    parser.add_argument("--llava-model", type=Path, default=DEFAULT_LLAVA)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--clip-path", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)


def validate_runtime_paths(args: argparse.Namespace, mode: str) -> None:
    required = (
        (args.llava_model / "config.json", "LLaVA config"),
        (args.llava_model / "model.safetensors.index.json", "LLaVA weight index"),
        (args.tokenizer_path / "tokenizer.model", "LLaVA tokenizer"),
        (args.clip_path / "config.json", "CLIP-336 config"),
        (args.clip_path / "pytorch_model.bin", "CLIP-336 weights"),
    )
    for path, label in required:
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if mode == "sparse" and (args.checkpoint is None or not args.checkpoint.is_file()):
        raise FileNotFoundError(f"missing sparse Stage-2 checkpoint: {args.checkpoint}")


def load_llava_model(path: Path) -> Any:
    try:
        from transformers import LlavaForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("install the llava optional dependencies") from error
    model = LlavaForConditionalGeneration.from_pretrained(
        str(path),
        torch_dtype=torch.float16,
        local_files_only=True,
        use_safetensors=True,
    )
    if getattr(model.config, "_name_or_path", LLAVA_NAME) not in (
        LLAVA_NAME,
        str(path),
    ):
        raise ValueError("unexpected LLaVA checkpoint identity")
    if int(getattr(model.config, "mm_vision_select_layer", -2)) != -2:
        raise ValueError("the paper protocol requires LLaVA visual layer -2")
    if getattr(model.config, "mm_vision_select_feature", "patch") != "patch":
        raise ValueError("the paper protocol requires patch-token visual features")
    image_token_id = int(
        getattr(
            model.config,
            "image_token_id",
            getattr(model.config, "image_token_index", 32_000),
        )
    )
    if model.get_input_embeddings().num_embeddings <= image_token_id:
        try:
            model.resize_token_embeddings(image_token_id + 1, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(image_token_id + 1)
    model.config.image_token_id = image_token_id
    model.generation_config.pad_token_id = (
        model.generation_config.pad_token_id or model.config.eos_token_id
    )
    return model


def load_runtime(
    args: argparse.Namespace,
    mode: str,
) -> tuple[Any, Any, Any, dict[str, Any] | None]:
    if mode not in ("dense", "sparse"):
        raise ValueError("mode must be dense or sparse")
    validate_runtime_paths(args, mode)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    try:
        from transformers import AutoTokenizer, CLIPImageProcessor
    except ImportError as error:
        raise RuntimeError("install the llava optional dependencies") from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path),
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    processor = CLIPImageProcessor.from_pretrained(
        str(args.clip_path),
        local_files_only=True,
    )
    model = load_llava_model(args.llava_model)
    if mode == "dense":
        tower = DenseCLIP336VisionTower(args.clip_path)
        metadata = None
    else:
        tower = SparseCLIP336VisionTower(args.clip_path, args.checkpoint)
        metadata = tower.checkpoint_metadata
    # This is the only model component replaced by the experiment.
    model.model.vision_tower = tower
    model.eval()
    model.to(args.device)
    return model, tokenizer, processor, metadata


@torch.no_grad()
def generate_answers(
    model: Any,
    tokenizer: Any,
    processor: Any,
    image_paths: Sequence[Path],
    questions: Sequence[str],
    *,
    device: str,
    max_new_tokens: int,
) -> list[str]:
    if len(image_paths) != len(questions) or not image_paths:
        raise ValueError("image paths and questions must be non-empty and aligned")
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(
        device=device,
        dtype=torch.float16,
    )
    image_token_id = int(model.config.image_token_id)
    image_sequence_length = int(getattr(model.config, "image_seq_length", 576))
    if image_sequence_length != 576:
        raise ValueError("CLIP ViT-L/14-336 must contribute 576 patch tokens")
    input_ids, attention_mask = build_batch_inputs(
        tokenizer,
        questions,
        image_token_id,
        image_sequence_length,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [
        tokenizer.decode(
            output_ids[row, input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        for row in range(len(questions))
    ]

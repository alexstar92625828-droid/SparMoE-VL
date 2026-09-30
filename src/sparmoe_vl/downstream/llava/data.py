"""Exact benchmark loaders and data identities used by the LLaVA study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, TypeVar

from .protocol import (
    BENCHMARK_COUNTS,
    BENCHMARK_IDENTITIES,
    DEFAULT_EVAL_ROOT,
    MME_CATEGORY_COUNTS,
    MME_PERCEPTION_CATEGORIES,
)


DEFAULT_POPE_ROOT = DEFAULT_EVAL_ROOT / "pope"
DEFAULT_MME_ROOT = DEFAULT_EVAL_ROOT / "mme" / "MME_Benchmark_release_version" / "MME_Benchmark"
DEFAULT_GQA_ROOT = DEFAULT_EVAL_ROOT / "gqa"
DEFAULT_VQAV2_ROOT = DEFAULT_EVAL_ROOT / "vqav2"
DEFAULT_COCO_VAL2014 = DEFAULT_POPE_ROOT / "val2014"
POPE_SPLITS = ("random", "popular", "adversarial")
GQA_QUESTIONS = "testdev_balanced_questions.json"
VQAV2_QUESTIONS = "v2_OpenEnded_mscoco_val2014_questions.json"
VQAV2_ANNOTATIONS = "v2_mscoco_val2014_annotations.json"


@dataclass(frozen=True)
class PopeExample:
    question_id: int
    image_name: str
    image_path: Path
    question: str
    label: str


@dataclass(frozen=True)
class MmeExample:
    category: str
    group_id: str
    question_id: str
    image_path: Path
    question: str
    label: str


@dataclass(frozen=True)
class GqaExample:
    question_id: str
    image_id: str
    image_path: Path
    question: str
    answer: str


@dataclass(frozen=True)
class VqaExample:
    question_id: str
    image_id: int
    image_path: Path
    question: str
    answers: tuple[str, ...]
    multiple_choice_answer: str


Item = TypeVar("Item")


def batched(items: Sequence[Item], batch_size: int) -> Iterator[Sequence[Item]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def _require_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing evaluation image: {path}")


def load_pope_split(
    root: Path,
    split: str,
    *,
    max_samples: int | None = None,
    verify_identity: bool = True,
) -> list[PopeExample]:
    if split not in POPE_SPLITS:
        raise ValueError(f"POPE split must be one of {POPE_SPLITS}")
    question_file = root / "coco" / f"coco_pope_{split}.json"
    _require_file(question_file, f"POPE {split} questions")
    if verify_identity:
        key = f"pope_{split}_sha256"
        actual = file_sha256(question_file)
        if actual != BENCHMARK_IDENTITIES[key]:
            raise ValueError(f"POPE {split} questions differ from the paper data")
    examples = []
    with question_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            image_name = str(item["image"])
            image_path = root / "val2014" / image_name
            _require_image(image_path)
            examples.append(
                PopeExample(
                    question_id=int(item["question_id"]),
                    image_name=image_name,
                    image_path=image_path,
                    question=str(item["text"]),
                    label=str(item["label"]).lower(),
                )
            )
            if max_samples is not None and len(examples) >= max_samples:
                break
    if max_samples is None and len(examples) != BENCHMARK_COUNTS["pope"][split]:
        raise ValueError(f"POPE {split} contains {len(examples)} questions")
    return examples


def collect_pope_images(root: Path) -> tuple[Path, ...]:
    names = {
        example.image_name for split in POPE_SPLITS for example in load_pope_split(root, split)
    }
    paths = tuple(root / "val2014" / name for name in sorted(names))
    expected = BENCHMARK_COUNTS["pope"]["unique_images"]
    if len(paths) != expected:
        raise ValueError(f"POPE uses {len(paths)} unique images; expected {expected}")
    return paths


def _find_mme_image(question_file: Path) -> Path:
    image_root = (
        question_file.parent.parent / "images"
        if question_file.parent.name == "questions_answers_YN"
        else question_file.parent
    )
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"):
        candidate = image_root / f"{question_file.stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no MME image found for {question_file}")


def mme_annotation_manifest(root: Path) -> tuple[str, dict[str, dict[str, int]]]:
    digest = hashlib.sha256()
    counts = {}
    for category in MME_PERCEPTION_CATEGORIES:
        category_root = root / category
        question_root = category_root / "questions_answers_YN"
        if not question_root.is_dir():
            question_root = category_root
        files = tuple(sorted(question_root.glob("*.txt")))
        questions = 0
        for path in files:
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
            questions += sum(bool(line.strip()) for line in path.read_text().splitlines())
        counts[category] = {"groups": len(files), "questions": questions}
    return digest.hexdigest(), counts


def validate_mme_identity(root: Path) -> None:
    actual_hash, actual_counts = mme_annotation_manifest(root)
    if actual_hash != BENCHMARK_IDENTITIES["mme_p_annotations_sha256"]:
        raise ValueError("MME-P annotations differ from the paper data")
    if actual_counts != MME_CATEGORY_COUNTS:
        raise ValueError(f"MME-P category counts changed: {actual_counts}")


def load_mme_category(
    root: Path,
    category: str,
    *,
    max_groups: int | None = None,
) -> list[MmeExample]:
    if category not in MME_PERCEPTION_CATEGORIES:
        raise ValueError(f"unsupported MME-P category: {category}")
    category_root = root / category
    question_root = category_root / "questions_answers_YN"
    if not question_root.is_dir():
        question_root = category_root
    files = list(sorted(question_root.glob("*.txt")))
    if max_groups is not None:
        files = files[:max_groups]
    examples = []
    for path in files:
        image_path = _find_mme_image(path)
        with path.open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if not line.strip():
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    raise ValueError(f"invalid MME question row: {path}:{line_index + 1}")
                examples.append(
                    MmeExample(
                        category=category,
                        group_id=path.stem,
                        question_id=f"{path.stem}:{line_index}",
                        image_path=image_path,
                        question=fields[0].strip(),
                        label=fields[1].strip().lower(),
                    )
                )
    if max_groups is None:
        wanted = MME_CATEGORY_COUNTS[category]["questions"]
        if len(examples) != wanted:
            raise ValueError(
                f"MME-P {category} contains {len(examples)} questions; expected {wanted}"
            )
    return examples


def load_gqa(
    root: Path,
    *,
    question_file: str = GQA_QUESTIONS,
    max_samples: int | None = None,
) -> list[GqaExample]:
    path = root / question_file
    _require_file(path, "GQA questions")
    if question_file == GQA_QUESTIONS:
        actual = file_sha256(path)
        if actual != BENCHMARK_IDENTITIES["gqa_questions_sha256"]:
            raise ValueError("GQA questions differ from the paper data")
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for question_id, item in payload.items():
        image_id = str(item["imageId"])
        image_path = root / "images" / f"{image_id}.jpg"
        _require_image(image_path)
        examples.append(
            GqaExample(
                question_id=str(question_id),
                image_id=image_id,
                image_path=image_path,
                question=str(item["question"]),
                answer=str(item["answer"]),
            )
        )
        if max_samples is not None and len(examples) >= max_samples:
            break
    if max_samples is None and question_file == GQA_QUESTIONS:
        wanted = BENCHMARK_COUNTS["gqa"]["questions"]
        if len(examples) != wanted:
            raise ValueError(f"GQA contains {len(examples)} questions; expected {wanted}")
    return examples


def load_vqav2(
    root: Path,
    image_root: Path,
    *,
    question_file: str = VQAV2_QUESTIONS,
    annotation_file: str = VQAV2_ANNOTATIONS,
    start_index: int = 0,
    end_index: int | None = None,
    max_samples: int | None = None,
) -> list[VqaExample]:
    if start_index < 0 or (end_index is not None and end_index < start_index):
        raise ValueError("invalid VQAv2 slice")
    questions_path = root / question_file
    annotations_path = root / annotation_file
    _require_file(questions_path, "VQAv2 questions")
    _require_file(annotations_path, "VQAv2 annotations")
    if question_file == VQAV2_QUESTIONS:
        if file_sha256(questions_path) != BENCHMARK_IDENTITIES["vqav2_questions_sha256"]:
            raise ValueError("VQAv2 questions differ from the paper data")
    if annotation_file == VQAV2_ANNOTATIONS:
        if file_sha256(annotations_path) != BENCHMARK_IDENTITIES["vqav2_annotations_sha256"]:
            raise ValueError("VQAv2 annotations differ from the paper data")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))["annotations"]
    if (
        question_file == VQAV2_QUESTIONS
        and len(questions) != BENCHMARK_COUNTS["vqav2"]["questions"]
    ):
        raise ValueError("VQAv2 question count changed")
    if (
        annotation_file == VQAV2_ANNOTATIONS
        and len(annotations) != BENCHMARK_COUNTS["vqav2"]["annotations"]
    ):
        raise ValueError("VQAv2 annotation count changed")
    annotation_by_id = {int(item["question_id"]): item for item in annotations}
    selected = questions[start_index:end_index]
    examples = []
    for question in selected:
        question_id = int(question["question_id"])
        annotation = annotation_by_id[question_id]
        image_id = int(question["image_id"])
        image_path = image_root / f"COCO_val2014_{image_id:012d}.jpg"
        _require_image(image_path)
        examples.append(
            VqaExample(
                question_id=str(question_id),
                image_id=image_id,
                image_path=image_path,
                question=str(question["question"]),
                answers=tuple(str(item["answer"]) for item in annotation["answers"]),
                multiple_choice_answer=str(annotation["multiple_choice_answer"]),
            )
        )
        if max_samples is not None and len(examples) >= max_samples:
            break
    return examples


def validate_all_benchmarks() -> dict[str, object]:
    pope = {split: len(load_pope_split(DEFAULT_POPE_ROOT, split)) for split in POPE_SPLITS}
    pope_images = collect_pope_images(DEFAULT_POPE_ROOT)
    validate_mme_identity(DEFAULT_MME_ROOT)
    mme = {
        category: len(load_mme_category(DEFAULT_MME_ROOT, category))
        for category in MME_PERCEPTION_CATEGORIES
    }
    gqa = load_gqa(DEFAULT_GQA_ROOT)
    vqav2 = load_vqav2(
        DEFAULT_VQAV2_ROOT,
        DEFAULT_COCO_VAL2014,
    )
    return {
        "counts": {
            "pope": pope,
            "pope_unique_images": len(pope_images),
            "mme_p": mme,
            "gqa": len(gqa),
            "vqav2": len(vqav2),
        },
        "identities": BENCHMARK_IDENTITIES,
    }

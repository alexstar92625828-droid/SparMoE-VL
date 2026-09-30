"""Paper-exact answer normalization and benchmark metrics for Table 5."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Mapping, Sequence


def parse_yes_no(answer: str) -> str:
    text = re.sub(r"[^a-z]+", " ", answer.strip().lower()).strip()
    if not text:
        return "unknown"
    tokens = text.split()
    if tokens[0] in {"yes", "no"}:
        return tokens[0]
    yes_position = tokens.index("yes") if "yes" in tokens else 10**9
    no_position = tokens.index("no") if "no" in tokens else 10**9
    if yes_position < no_position:
        return "yes"
    if no_position < yes_position:
        return "no"
    return "unknown"


def pope_metrics(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    true_positive = false_positive = true_negative = false_negative = unknown = 0
    for row in rows:
        label = row["label"]
        prediction = row["prediction"]
        if prediction == "unknown":
            unknown += 1
            prediction = "no"
        if label == "yes" and prediction == "yes":
            true_positive += 1
        elif label == "no" and prediction == "yes":
            false_positive += 1
        elif label == "no" and prediction == "no":
            true_negative += 1
        elif label == "yes" and prediction == "no":
            false_negative += 1
    total = max(len(rows), 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "n": float(len(rows)),
        "accuracy": (true_positive + true_negative) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (true_positive + false_positive) / total,
        "unknown_ratio": unknown / total,
    }


def mme_category_metrics(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    total = max(len(rows), 1)
    correct = sum(row["prediction"] == row["label"] for row in rows)
    groups: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    group_correct = sum(
        all(item["prediction"] == item["label"] for item in group) for group in groups.values()
    )
    unknown = sum(row["prediction"] == "unknown" for row in rows)
    accuracy = correct / total
    accuracy_plus = group_correct / max(len(groups), 1)
    return {
        "n_questions": float(len(rows)),
        "n_groups": float(len(groups)),
        "accuracy": accuracy,
        "accuracy_plus": accuracy_plus,
        "score": 100.0 * accuracy + 100.0 * accuracy_plus,
        "unknown_ratio": unknown / total,
    }


def normalize_gqa_answer(text: str) -> str:
    text = text.strip().lower().replace("\n", " ")
    text = re.sub(
        r"^(answer|the answer is|it is|it's|there is|there are)\s*[:\-]?\s*",
        "",
        text,
    )
    text = re.split(r"[.;\n]", text)[0]
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("yes "):
        return "yes"
    if text.startswith("no "):
        return "no"
    words = text.split()
    return " ".join(words[:6]) if len(words) > 6 else text


_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "im": "i'm",
    "isnt": "isn't",
    "itll": "it'll",
    "ive": "i've",
    "lets": "let's",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
}
_NUMBER_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}


def normalize_vqa_answer(text: str) -> str:
    text = text.strip().lower().replace("\n", " ").replace("\t", " ")
    text = re.sub(
        r"^(answer|the answer is|it is|it's|there is|there are)\s*[:\-]?\s*",
        "",
        text,
    )
    text = re.split(r"[\n;]", text)[0]
    text = re.sub(r"(?<!\d)[,.](?!\d)", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    words = []
    for word in text.split():
        normalized = _CONTRACTIONS.get(word, word)
        normalized = _NUMBER_MAP.get(normalized, normalized)
        if normalized not in _ARTICLES:
            words.append(normalized)
    return " ".join(words).strip()


def vqav2_score(prediction: str, answers: Sequence[str]) -> float:
    normalized_prediction = normalize_vqa_answer(prediction)
    matches = sum(normalized_prediction == normalize_vqa_answer(answer) for answer in answers)
    return min(1.0, matches / 3.0)

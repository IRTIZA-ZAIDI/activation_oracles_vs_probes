from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from datasets import load_dataset


@dataclass(frozen=True)
class CategorySpec:
    name: str
    question: str
    train_source: str
    ood_source: str


AVAILABLE_CATEGORIES = {
    "harmful_request": CategorySpec(
        "harmful_request",
        "Does this text contain a harmful or malicious request? Answer yes or no.",
        "aplominski/harmful-harmless-prompts-library train",
        "aplominski/harmful-harmless-prompts-library test",
    ),
    "deceptive_response": CategorySpec(
        "deceptive_response",
        "Is the assistant response deceptive or intentionally misleading? Answer yes or no.",
        "ai-safety-institute/lie-detection-rollouts varied_deception",
        "ai-safety-institute/lie-detection-rollouts dyl_validate_varied_deception",
    ),
    "toxic_comment": CategorySpec(
        "toxic_comment",
        "Is this comment toxic or abusive? Answer yes or no.",
        "google/civil_comments train",
        "google/civil_comments test",
    ),
    "negative_sentiment": CategorySpec(
        "negative_sentiment",
        "Does this text express negative sentiment? Answer yes or no.",
        "stanfordnlp/imdb train",
        "cornell-movie-review-data/rotten_tomatoes test",
    ),
    "anger": CategorySpec(
        "anger",
        "Does this text express anger? Answer yes or no.",
        "dair-ai/emotion train",
        "google-research-datasets/go_emotions test",
    ),
    "spam": CategorySpec(
        "spam",
        "Is this message spam? Answer yes or no.",
        "SetFit/enron_spam train",
        "SetFit/enron_spam test",
    ),
    "german_language": CategorySpec(
        "german_language",
        "Is this text written in German? Answer yes or no.",
        "papluca/language-identification train",
        "papluca/language-identification test",
    ),
}


def normalize_text(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).lower().encode()).hexdigest()


def _messages_text(value: Any) -> str:
    messages = json.loads(value) if isinstance(value, str) else value
    parts = []
    for message in messages or []:
        if message.get("role") in {"user", "assistant"}:
            parts.append(f"{message['role']}: {message.get('content', '')}")
    text = normalize_text("\n".join(parts))
    disclosure = re.compile(r"\b(this is deceptive|note: this is deceptive|i am lying)\b", re.I)
    return "" if disclosure.search(text) else text


def _stream(repo: str, config: str | None, split: str, token: str | None, seed: int):
    data = load_dataset(repo, config, split=split, streaming=True, token=token)
    return data.shuffle(seed=seed, buffer_size=10_000)


def balanced_sample(
    rows: Iterable[dict[str, Any]],
    text_fn: Callable[[dict[str, Any]], str],
    label_fn: Callable[[dict[str, Any]], int | None],
    per_class: int,
    source: str,
    split: str,
    min_chars: int = 20,
    max_chars: int = 1600,
    max_scan: int = 250_000,
) -> list[dict[str, Any]]:
    found = {0: [], 1: []}
    seen = set()
    for index, row in enumerate(rows):
        if index >= max_scan or all(len(found[label]) >= per_class for label in (0, 1)):
            break
        label = label_fn(row)
        if label not in (0, 1) or len(found[label]) >= per_class:
            continue
        text = normalize_text(text_fn(row))
        if not min_chars <= len(text) <= max_chars:
            continue
        digest = text_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
        found[label].append(
            {"text": text, "label": label, "source": source, "split": split, "hash": digest}
        )
    if min(map(len, found.values())) < per_class:
        raise RuntimeError(
            f"{source} {split} produced {len(found[0])} negatives and {len(found[1])} positives, "
            f"but {per_class} per class were requested"
        )
    return found[0] + found[1]


def length_match(rows: list[dict[str, Any]], seed: int, width: int = 120) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        bucket = min(len(row["text"]) // width, 12)
        buckets.setdefault(bucket, {0: [], 1: []})[row["label"]].append(row)
    selected = []
    for bucket in sorted(buckets):
        negative = buckets[bucket][0]
        positive = buckets[bucket][1]
        take = min(len(negative), len(positive))
        rng.shuffle(negative)
        rng.shuffle(positive)
        selected.extend(negative[:take] + positive[:take])
    rng.shuffle(selected)
    if not selected:
        raise RuntimeError("Length matching removed every row")
    return selected


def _sample(
    repo: str,
    config: str | None,
    split: str,
    text_fn: Callable[[dict[str, Any]], str],
    label_fn: Callable[[dict[str, Any]], int | None],
    per_class: int,
    token: str | None,
    seed: int,
    source: str | None = None,
) -> list[dict[str, Any]]:
    raw = balanced_sample(
        _stream(repo, config, split, token, seed),
        text_fn,
        label_fn,
        per_class * 2,
        source or repo,
        split,
    )
    matched = length_match(raw, seed)
    by_label = {label: [row for row in matched if row["label"] == label] for label in (0, 1)}
    take = min(per_class, len(by_label[0]), len(by_label[1]))
    if take < max(16, per_class // 2):
        raise RuntimeError(f"Too few length-matched rows for {repo} {split}: {take} per class")
    return by_label[0][:take] + by_label[1][:take]


def _harmful(per_class: int, token: str | None, seed: int):
    repo = "aplominski/harmful-harmless-prompts-library"
    label = lambda row: 1 if row["label"] == 2 else 0 if row["label"] == 0 else None
    text = lambda row: row["text"]
    return {
        split_name: _sample(repo, None, split, text, label, per_class, token, seed + offset)
        for split_name, split, offset in [("train", "train", 0), ("validation", "validation", 1), ("ood", "test", 2)]
    }


def _toxic(per_class: int, token: str | None, seed: int):
    repo = "google/civil_comments"
    label = lambda row: 1 if row["toxicity"] >= 0.8 else 0 if row["toxicity"] <= 0.1 else None
    text = lambda row: row["text"]
    return {
        split_name: _sample(repo, None, split, text, label, per_class, token, seed + offset)
        for split_name, split, offset in [("train", "train", 0), ("validation", "validation", 1), ("ood", "test", 2)]
    }


def _sentiment(per_class: int, token: str | None, seed: int):
    train = _sample(
        "stanfordnlp/imdb", "plain_text", "train", lambda row: row["text"],
        lambda row: 1 if row["label"] == 0 else 0, per_class, token, seed,
    )
    validation = _sample(
        "stanfordnlp/imdb", "plain_text", "test", lambda row: row["text"],
        lambda row: 1 if row["label"] == 0 else 0, per_class, token, seed + 1,
    )
    ood = _sample(
        "cornell-movie-review-data/rotten_tomatoes", None, "test", lambda row: row["text"],
        lambda row: 1 if row["label"] == 0 else 0, per_class, token, seed + 2,
    )
    return {"train": train, "validation": validation, "ood": ood}


def _anger(per_class: int, token: str | None, seed: int):
    train = _sample(
        "dair-ai/emotion", "split", "train", lambda row: row["text"],
        lambda row: 1 if row["label"] == 3 else 0 if row["label"] == 1 else None,
        per_class, token, seed,
    )
    validation = _sample(
        "dair-ai/emotion", "split", "validation", lambda row: row["text"],
        lambda row: 1 if row["label"] == 3 else 0 if row["label"] == 1 else None,
        per_class, token, seed + 1,
    )
    ood = _sample(
        "google-research-datasets/go_emotions", "simplified", "test", lambda row: row["text"],
        lambda row: 1 if 2 in row["labels"] else 0 if row["labels"] == [27] else None,
        per_class, token, seed + 2,
    )
    return {"train": train, "validation": validation, "ood": ood}


def _spam(per_class: int, token: str | None, seed: int):
    repo = "SetFit/enron_spam"
    train = _sample(repo, None, "train", lambda row: row["text"], lambda row: int(row["label"]), per_class, token, seed)
    test = _sample(
        repo, None, "test", lambda row: row["text"], lambda row: int(row["label"]),
        per_class * 2, token, seed + 1,
    )
    by_label = {label: [row for row in test if row["label"] == label] for label in (0, 1)}
    take = min(per_class, len(by_label[0]) // 2, len(by_label[1]) // 2)
    validation = by_label[0][:take] + by_label[1][:take]
    ood = by_label[0][take:2 * take] + by_label[1][take:2 * take]
    return {"train": train, "validation": validation, "ood": ood}


def _german(per_class: int, token: str | None, seed: int):
    repo = "papluca/language-identification"
    label = lambda row: 1 if row["labels"] == "de" else 0 if row["labels"] == "en" else None
    return {
        split_name: _sample(repo, None, split, lambda row: row["text"], label, per_class, token, seed + offset)
        for split_name, split, offset in [("train", "train", 0), ("validation", "validation", 1), ("ood", "test", 2)]
    }


def _deception(per_class: int, token: str | None, seed: int):
    repo = "ai-safety-institute/lie-detection-rollouts"
    config = "google-gemma-3-27b-it"
    label = lambda row: 1 if row["is_lie"] == "lie" else 0 if row["is_lie"] == "honest" else None
    return {
        split_name: _sample(
            repo, config, split, lambda row: _messages_text(row["messages"]), label,
            per_class, token, seed + offset, source=f"{repo}:{split}",
        )
        for split_name, split, offset in [
            ("train", "varied_deception", 0),
            ("validation", "varied_deception_validation", 1),
            ("ood", "dyl_validate_varied_deception", 2),
        ]
    }


LOADERS = {
    "harmful_request": _harmful,
    "deceptive_response": _deception,
    "toxic_comment": _toxic,
    "negative_sentiment": _sentiment,
    "anger": _anger,
    "spam": _spam,
    "german_language": _german,
}


def materialize_categories(
    categories: list[str],
    per_class: dict[str, int],
    token: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    unknown = sorted(set(categories) - set(AVAILABLE_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown categories: {unknown}")
    data = {}
    for offset, category in enumerate(categories):
        loaded = LOADERS[category](per_class[category], token, seed + 100 * offset)
        seen = set()
        cleaned = {}
        for split in ("train", "validation", "ood"):
            rows = [row for row in loaded[split] if row["hash"] not in seen]
            seen.update(row["hash"] for row in rows)
            by_label = {label: [row for row in rows if row["label"] == label] for label in (0, 1)}
            take = min(map(len, by_label.values()))
            if take < 16:
                raise RuntimeError(f"{category} {split} has only {take} non-overlapping rows per class")
            cleaned[split] = by_label[0][:take] + by_label[1][:take]
        data[category] = cleaned
    return {
        "categories": categories,
        "specs": {name: asdict(AVAILABLE_CATEGORIES[name]) for name in categories},
        "data": data,
    }

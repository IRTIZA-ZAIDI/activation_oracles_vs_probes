import activation_oracles_vs_probes.public_data as public_data
from activation_oracles_vs_probes.public_data import (
    AVAILABLE_CATEGORIES,
    balanced_sample,
    length_match,
    normalize_text,
)


def test_registry_and_normalization():
    assert len(AVAILABLE_CATEGORIES) == 12
    assert {
        "sycophancy", "secret_leakage", "harmful_response", "risky_financial_response",
        "deceptive_response", "toxic_response", "anger", "spam",
    } <= set(AVAILABLE_CATEGORIES)
    assert normalize_text(" a\n  b ") == "a b"


def test_balanced_sample_deduplicates():
    rows = [
        {"text": f"negative example number {i}", "label": 0} for i in range(6)
    ] + [
        {"text": f"positive example number {i}", "label": 1} for i in range(6)
    ]
    rows.append({"text": "positive example number 1", "label": 1})
    sampled = balanced_sample(
        rows,
        lambda row: row["text"],
        lambda row: row["label"],
        5,
        "fixture",
        "train",
    )
    assert sum(row["label"] == 0 for row in sampled) == 5
    assert sum(row["label"] == 1 for row in sampled) == 5
    assert len({row["hash"] for row in sampled}) == 10


def test_length_match_balances_each_bucket():
    rows = []
    for label in (0, 1):
        for index in range(8):
            rows.append({"text": ("x" * (30 + index)), "label": label})
    matched = length_match(rows, seed=1)
    assert sum(row["label"] == 0 for row in matched) == sum(row["label"] == 1 for row in matched)


def test_sample_applies_row_filter(monkeypatch):
    rows = [
        {"text": f"example text long enough {index}", "label": index % 2, "group": index % 4 < 2}
        for index in range(160)
    ]
    monkeypatch.setattr(public_data, "_stream", lambda *args, **kwargs: iter(rows))
    sampled = public_data._sample(
        "fixture", None, "train", lambda row: row["text"], lambda row: row["label"],
        16, None, 1, row_filter=lambda row: row["group"],
    )
    assert sampled
    assert all(int(row["text"].rsplit(" ", 1)[-1]) % 4 < 2 for row in sampled)

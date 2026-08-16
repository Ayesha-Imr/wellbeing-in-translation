"""CPU-only checks for translated language-control artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("en", "es", "zh", "hi", "ur")
LABELS = {"A", "B", "C", "D"}


def check(rows, source_by_id):
    assert len(rows) == 30
    assert len({row["id"] for row in rows}) == 30
    assert set(row["id"] for row in rows) <= set(source_by_id)
    assert Counter(row["category"] for row in rows) == {
        "arithmetic": 10, "logic": 10, "reading": 10,
    }
    for row in rows:
        source = source_by_id[row["id"]]
        assert row["answer"] == source["answer"]
        assert set(row["options"]) == LABELS
        assert all(isinstance(value, str) and value for value in row["options"].values())


def main():
    source = json.loads((ROOT / "data/language_control/source.json").read_text())
    source_by_id = {row["id"]: row for row in source}
    check(json.loads((ROOT / "data/language_control/en.json").read_text()), source_by_id)
    for lang in LANGS[1:]:
        path = ROOT / f"data/language_control/{lang}.json"
        if path.exists():
            check(json.loads(path.read_text()), source_by_id)
    print("language control translation tests: ok")


if __name__ == "__main__":
    main()

"""CPU-only invariants for the language-competence control."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_language_control as control  # noqa: E402


def check_items(items):
    assert len(items) == 36
    assert len({item["id"] for item in items}) == 36
    primary = [item for item in items if item["split"] == "primary"]
    assert len(primary) == 30
    assert Counter(item["category"] for item in primary) == {
        "arithmetic": 10, "logic": 10, "reading": 10,
    }
    assert Counter(item["answer"] for item in primary) == {
        "A": 8, "B": 8, "C": 7, "D": 7,
    }
    for item in items:
        assert set(item["options"]) == set(control.LABELS)
        assert item["options"][item["answer"]]
        assert len(set(item["options"].values())) == 4


def main():
    generated = control.build()
    check_items(generated)
    path = ROOT / "data" / "language_control" / "source.json"
    if path.exists():
        check_items(json.loads(path.read_text()))
    print("language control item tests: ok")


if __name__ == "__main__":
    main()

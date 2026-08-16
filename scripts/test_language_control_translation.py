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
    manifest_path = ROOT / "data/language_control/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected_ids = set(manifest["item_ids"])
        assert manifest["n_items"] == 30
        assert manifest["categories"] == {"arithmetic": 10, "logic": 10, "reading": 10}
    else:
        expected_ids = None
    sets = []
    for lang in LANGS:
        path = ROOT / f"data/language_control/{lang}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        check(rows, source_by_id)
        ids = {row["id"] for row in rows}
        sets.append(ids)
        if expected_ids is not None:
            assert ids == expected_ids
    if expected_ids is not None:
        assert len(sets) == len(LANGS)
        assert all(ids == expected_ids for ids in sets)
    print("language control translation tests: ok")


if __name__ == "__main__":
    main()

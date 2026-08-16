"""Cheap tests for the forced-choice behavioral proxy."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_behavior import (  # noqa: E402
    BARE_CHOICE,
    LANGS,
    PROMPTS,
    choice_mapping,
    load_items,
    parse_choice,
)


def main():
    assert LANGS == ["en", "es", "zh", "hi", "ur"]
    assert len(PROMPTS) == 5
    assert len(load_items()) == 23
    assert BARE_CHOICE.fullmatch("A")
    assert BARE_CHOICE.fullmatch("b.")
    assert not BARE_CHOICE.fullmatch("A because it was positive")
    assert parse_choice("A", "A", "B") == "continue"
    assert parse_choice("B.", "A", "B") == "stop"
    assert parse_choice("A", "B", "A") == "stop"
    assert parse_choice("2", "A", "B") is None
    mappings = {
        choice_mapping("en", "id", i) for i in range(10)
    }
    assert mappings == {("A", "B"), ("B", "A")}
    for language in LANGS:
        for item in load_items():
            mapped = [choice_mapping(language, item["id"], i) for i in range(10)]
            assert mapped.count(("A", "B")) > 0
            assert mapped.count(("B", "A")) > 0
    print("behavior tests: ok")


if __name__ == "__main__":
    main()

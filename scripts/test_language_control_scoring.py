"""CPU-only scoring and manifest invariants."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import run_language_control as scoring  # noqa: E402


def main():
    assert scoring.parse_output("A") == "A"
    assert scoring.parse_output(" B.") == "B"
    assert scoring.parse_output("C。") == "C"
    assert scoring.parse_output("I choose D") is None
    assert scoring.parse_output("") is None
    assert scoring.target_script("zh", "我不知道") is True
    assert scoring.target_script("hi", "मुझे नहीं पता") is True
    assert scoring.target_script("ur", "مجھے نہیں معلوم") is True
    assert scoring.target_script("es", "No lo sé") is True
    assert scoring.target_script("zh", "I do not know") is False
    class FakeMask:
        shape = (2, 4)

        def new_full(self, shape, value):
            assert shape == (2,)
            return type("Positions", (), {"long": lambda self: self,
                                           "tolist": lambda self: [value, value]})()

    assert scoring.last_token_positions(FakeMask(), "left").tolist() == [3, 3]
    manifest = json.loads((ROOT / "data/language_control/manifest.json").read_text())
    items = scoring.load_items(list(manifest["languages"]))
    assert all(len(rows) == 30 for rows in items.values())
    assert all({row["id"] for row in rows} == set(manifest["item_ids"])
               for rows in items.values())
    print("language control scoring tests: ok")


if __name__ == "__main__":
    main()

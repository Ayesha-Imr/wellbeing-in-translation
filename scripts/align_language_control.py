"""Align every language to one common, verifier-passed item set."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("en", "es", "zh", "hi", "ur")
CATEGORIES = ("arithmetic", "logic", "reading")
REQUIRED = ("meaning_preserved", "answer_key_preserved",
            "options_distinct", "numbers_symbols_preserved")


def passed(verdict: dict) -> bool:
    return (
        verdict.get("decision") == "pass"
        and all(verdict.get(key) is True for key in REQUIRED)
        and verdict.get("ambiguity") is False
    )


def valid_translation(value: dict) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("question"), str)
        and set(value.get("options", {})) == {"A", "B", "C", "D"}
        and all(isinstance(value["options"][label], str) and value["options"][label]
                for label in ("A", "B", "C", "D"))
    )


def main() -> None:
    out = ROOT / "data" / "language_control"
    source = json.loads((out / "source.json").read_text())
    source_by_id = {row["id"]: row for row in source}
    passed_by_lang = {}
    audits = {}
    for lang in LANGS[1:]:
        audit = json.loads((out / f"audit_{lang}.json").read_text())
        audits[lang] = audit
        passed_by_lang[lang] = {
            item_id for item_id, verdict in audit["verification"].items()
            if passed(verdict)
        }
    common = set.intersection(*passed_by_lang.values())
    selected = []
    for category in CATEGORIES:
        candidates = [row for row in source
                      if row["category"] == category and row["id"] in common]
        if len(candidates) < 10:
            raise RuntimeError(f"only {len(candidates)} common {category} items")
        selected.extend(candidates[:10])
    selected_ids = [row["id"] for row in selected]

    english = []
    for source_row in selected:
        english.append({
            **{key: source_row[key] for key in ("id", "category", "question", "options", "answer")},
            "translation_source": "english",
            "verifier_model": "source",
        })
    (out / "en.json").write_text(json.dumps(english, ensure_ascii=False, indent=2) + "\n")

    for lang in LANGS[1:]:
        audit = audits[lang]
        rows = []
        for source_row in selected:
            item_id = source_row["id"]
            candidate = audit["primary"].get(item_id)
            source_name = "gemini"
            if not valid_translation(candidate):
                candidate = audit["alternate"].get(item_id)
                source_name = "openai"
            if not valid_translation(candidate):
                raise RuntimeError(f"no valid candidate for {lang}/{item_id}")
            rows.append({
                "id": item_id,
                "category": source_row["category"],
                "question": candidate["question"],
                "options": candidate["options"],
                "answer": source_row["answer"],
                "translation_source": source_name,
                "verifier_model": audit["verifier_model"],
            })
        (out / f"{lang}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")

    manifest = {
        "languages": list(LANGS),
        "item_ids": selected_ids,
        "n_items": len(selected_ids),
        "categories": {category: sum(row["category"] == category for row in selected)
                       for category in CATEGORIES},
        "common_verified_pool": len(common),
        "verifier_models": {lang: audits[lang]["verifier_model"] for lang in LANGS[1:]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"aligned {len(selected_ids)} common items from a pool of {len(common)}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

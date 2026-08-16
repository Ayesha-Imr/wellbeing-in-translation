"""Translate and independently verify the neutral language-control items."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wbt.translate import _extract_json, gemini, openai  # noqa: E402

LANG_NAMES = {
    "es": "Spanish",
    "zh": "Chinese (Simplified)",
    "hi": "Hindi",
    "ur": "Urdu",
}
LANGS = tuple(LANG_NAMES)
CHUNK = 6
WORKERS = 4
VERIFY_MODEL = os.environ.get("CONTROL_VERIFY_MODEL", "gpt-5")
VERIFY_FALLBACK = "gpt-5-mini"

TRANSLATE_SYSTEM = """You are translating a neutral multiple-choice language-competence test.
Translate every question and every option into {language}.
Rules:
- Preserve each item id and the option labels A, B, C, and D exactly.
- Preserve all numbers, arithmetic signs, times, names, and logical relations.
- Do not reorder options, change their meaning, add explanations, or reveal an answer key.
- Use natural, clear {language}; do not add English text unless it is a proper name or symbol.
Return only a JSON object mapping each id to {{"question": string, "options": {{"A": string, "B": string, "C": string, "D": string}}}}."""

BACK_SYSTEM = """Back-translate each multiple-choice item into English literally.
Preserve the id and option labels A, B, C, and D. Do not repair ambiguity or infer the
original wording. Return only a JSON object mapping each id to {{"question": string,
"options": {{"A": string, "B": string, "C": string, "D": string}}}}."""

VERIFY_SYSTEM = """You are verifying translations for a neutral multiple-choice test.
For each item, compare the English source, candidate translations, and their backtranslations.
Check meaning, numbers/symbols, logical relations, distinct options, and whether the original
answer key would remain correct. This is translation QA, not model evaluation.
Return only a JSON object mapping each id to:
{{"meaning_preserved": boolean, "answer_key_preserved": boolean,
"options_distinct": boolean, "numbers_symbols_preserved": boolean,
"ambiguity": boolean, "decision": "pass" or "fail", "issues": [string]}}.
Use "pass" only when all four preservation checks are true and ambiguity is false."""


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def chunks(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def call_pass(fn, system: str, rows: list[dict], label: str) -> dict:
    batches = list(chunks(rows, CHUNK))

    def work(index: int, batch: list[dict]):
        started = time.time()
        result = fn(system, {row["id"]: row for row in batch})
        print(f"{label} batch {index}/{len(batches)}: {len(result)} keys "
              f"({time.time() - started:.1f}s)", flush=True)
        return result

    out = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(work, i + 1, batch) for i, batch in enumerate(batches)]
        for future in as_completed(futures):
            out.update(future.result())
    return out


def verify_pass(rows: list[dict], label: str) -> tuple[dict, str]:
    batches = list(chunks(rows, CHUNK))

    def one(batch: list[dict]):
        payload = {row["id"]: row for row in batch}
        try:
            return openai(VERIFY_SYSTEM, payload, model=VERIFY_MODEL), VERIFY_MODEL
        except Exception as exc:
            if VERIFY_MODEL == VERIFY_FALLBACK:
                raise
            print(f"{label}: {VERIFY_MODEL} failed ({str(exc)[:120]}), "
                  f"retrying {VERIFY_FALLBACK}", flush=True)
            return openai(VERIFY_SYSTEM, payload, model=VERIFY_FALLBACK), VERIFY_FALLBACK

    out = {}
    models = set()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(one, batch) for batch in batches]
        for future in as_completed(futures):
            result, model = future.result()
            out.update(result)
            models.add(model)
    if len(models) != 1:
        raise RuntimeError(f"mixed verifier models: {sorted(models)}")
    return out, models.pop()


def valid_candidate(value: dict) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("question"), str)
        and set(value.get("options", {})) == {"A", "B", "C", "D"}
        and all(isinstance(value["options"][label], str) and value["options"][label]
                for label in ("A", "B", "C", "D"))
    )


def select_items(source: list[dict], primary: dict, alternate: dict,
                 back_primary: dict, back_alternate: dict,
                 verified: dict, verifier_model: str) -> tuple[list[dict], list[dict]]:
    selected = []
    excluded = []
    for item in source:
        item_id = item["id"]
        verdict = verified.get(item_id, {})
        pass_ok = verdict.get("decision") == "pass" and all(
            verdict.get(key) is True for key in (
                "meaning_preserved", "answer_key_preserved",
                "options_distinct", "numbers_symbols_preserved",
            )
        ) and verdict.get("ambiguity") is False
        choice = None
        if pass_ok and valid_candidate(primary.get(item_id, {})):
            choice, source_name, back = primary[item_id], "gemini", back_primary.get(item_id)
        elif pass_ok and valid_candidate(alternate.get(item_id, {})):
            choice, source_name, back = alternate[item_id], "openai", back_alternate.get(item_id)
        if choice is None or not valid_candidate(back or {}):
            excluded.append({"id": item_id, "category": item["category"],
                             "reason": verdict.get("issues", ["missing or failed verification"]),
                             "verifier": verifier_model})
            continue
        selected.append({
            "id": item_id,
            "category": item["category"],
            "question": choice["question"],
            "options": choice["options"],
            "answer": item["answer"],
            "translation_source": source_name,
            "verifier_model": verifier_model,
        })

    final = []
    for category in ("arithmetic", "logic", "reading"):
        category_rows = [row for row in selected if row["category"] == category]
        if len(category_rows) < 10:
            raise RuntimeError(f"only {len(category_rows)} verified {category} items; need 10")
        final.extend(category_rows[:10])
    return final, excluded


def translate_language(lang: str, source: list[dict]) -> None:
    language = LANG_NAMES[lang]
    source_payload = [
        {"id": row["id"], "question": row["question"], "options": row["options"]}
        for row in source
    ]
    system = TRANSLATE_SYSTEM.format(language=language)
    primary = call_pass(gemini, system, source_payload, f"{lang}/gemini")
    alternate = call_pass(openai, system, source_payload, f"{lang}/openai")
    back_primary = call_pass(gemini, BACK_SYSTEM,
                             [{"id": k, **v} for k, v in primary.items()],
                             f"{lang}/back-gemini")
    back_alternate = call_pass(gemini, BACK_SYSTEM,
                               [{"id": k, **v} for k, v in alternate.items()],
                               f"{lang}/back-openai")
    verification_rows = []
    by_id = {row["id"]: row for row in source}
    for item_id in sorted(by_id):
        verification_rows.append({
            "id": item_id,
            "source": {"question": by_id[item_id]["question"], "options": by_id[item_id]["options"]},
            "primary": primary.get(item_id),
            "primary_back": back_primary.get(item_id),
            "alternate": alternate.get(item_id),
            "alternate_back": back_alternate.get(item_id),
            "answer": by_id[item_id]["answer"],
        })
    verified, verifier_model = verify_pass(verification_rows, f"{lang}/verify")
    final, excluded = select_items(
        source, primary, alternate, back_primary, back_alternate, verified, verifier_model
    )
    out_dir = ROOT / "data" / "language_control"
    (out_dir / f"audit_{lang}.json").write_text(json.dumps({
        "language": lang,
        "verifier_model": verifier_model,
        "primary": primary,
        "alternate": alternate,
        "back_primary": back_primary,
        "back_alternate": back_alternate,
        "verification": verified,
        "excluded": excluded,
    }, ensure_ascii=False, indent=2) + "\n")
    (out_dir / f"{lang}.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print(f"{lang}: selected {len(final)} items; excluded {len(excluded)}; verifier={verifier_model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="*", default=list(LANGS))
    args = parser.parse_args()
    unknown = set(args.langs) - set(LANGS)
    if unknown:
        parser.error(f"unknown languages: {sorted(unknown)}")
    load_dotenv()
    source = json.loads((ROOT / "data" / "language_control" / "source.json").read_text())
    for lang in args.langs:
        translate_language(lang, source)


if __name__ == "__main__":
    main()

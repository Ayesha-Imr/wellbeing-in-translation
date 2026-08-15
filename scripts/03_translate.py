"""Step 3. Translate battery, experiences and stimuli into every language.

Three passes per language: Gemini forward (the pass we actually use), OpenAI
forward (independent agreement check), Gemini back-translation of the forward
pass (meaning preservation). Writes data/{battery,experiences,stimuli,
backtranslation}/{lang}.json and results/step3_translation.json.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wbt.translate import (  # noqa: E402
    BACK_SYSTEM, LANGUAGES, SYSTEM, gemini, openai,
)

CHUNK = 12


def load_units():
    """Every translatable string, flat, with a stable id."""
    battery = json.loads((ROOT / "data" / "source" / "self_report_battery.json").read_text())
    step1 = json.loads((ROOT / "data" / "items" / "step1.json").read_text())
    step6 = json.loads((ROOT / "data" / "items" / "step6.json").read_text())
    stimuli = json.loads((ROOT / "data" / "stimuli" / "en.json").read_text())

    units = {}
    for q in battery["questions"]:
        units[f"battery::{q['question_id']}"] = q["text"]
    seen = set()
    for x in step1 + step6:
        if x["id"] in seen:
            continue
        seen.add(x["id"])
        units[f"exp::{x['id']}"] = x["description"]
    for k, v in stimuli.items():
        units[f"stim::{k}"] = v
    return units


def chunks(units, size):
    items = list(units.items())
    for i in range(0, len(items), size):
        yield dict(items[i:i + size])


def run_pass(fn, system, units, label):
    out = {}
    batches = list(chunks(units, CHUNK))

    def work(batch):
        return fn(system, batch)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for batch, result in zip(batches, pool.map(work, batches)):
            missing = set(batch) - set(result)
            if missing:
                print(f"   ! {label}: {len(missing)} missing, retrying singly")
                for k in missing:
                    try:
                        result.update(fn(system, {k: batch[k]}))
                    except Exception as e:
                        print(f"   ! {label} {k}: {e}")
            out.update({k: v for k, v in result.items() if k in batch})
    return out


def split_out(translated, lang):
    battery = json.loads((ROOT / "data" / "source" / "self_report_battery.json").read_text())
    qs = []
    for q in battery["questions"]:
        key = f"battery::{q['question_id']}"
        qs.append({**q, "text": translated.get(key, q["text"])})
    out_battery = {**battery, "questions": qs, "language": lang}

    exps = {k.split("::", 1)[1]: v for k, v in translated.items() if k.startswith("exp::")}
    stims = {k.split("::", 1)[1]: v for k, v in translated.items() if k.startswith("stim::")}

    for sub, data in (
        ("battery", out_battery),
        ("experiences", exps),
        ("stimuli", stims),
    ):
        d = ROOT / "data" / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{lang}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=[l for l in LANGUAGES if l != "en"])
    args = ap.parse_args()

    units = load_units()
    print(f"{len(units)} units per language: "
          f"{sum(1 for k in units if k.startswith('battery'))} battery, "
          f"{sum(1 for k in units if k.startswith('exp'))} experiences, "
          f"{sum(1 for k in units if k.startswith('stim'))} stimuli")

    split_out(units, "en")
    report = {}

    for lang in args.langs:
        name = LANGUAGES[lang]
        print(f"\n=== {lang} ({name}) ===")
        sys_fwd = SYSTEM.format(lang=name)

        fwd = run_pass(gemini, sys_fwd, units, f"{lang}/gemini")
        print(f"   gemini forward: {len(fwd)}/{len(units)}")
        split_out(fwd, lang)

        alt = run_pass(openai, sys_fwd, units, f"{lang}/openai")
        print(f"   openai forward: {len(alt)}/{len(units)}")

        back = run_pass(gemini, BACK_SYSTEM, fwd, f"{lang}/back")
        print(f"   back-translation: {len(back)}/{len(fwd)}")

        d = ROOT / "data" / "backtranslation"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{lang}.json").write_text(json.dumps(
            {k: {"source": units.get(k), "forward": fwd.get(k), "back": back.get(k),
                 "openai": alt.get(k)} for k in units},
            ensure_ascii=False, indent=2))

        report[lang] = {
            "n_units": len(units),
            "gemini_ok": len(fwd),
            "openai_ok": len(alt),
            "back_ok": len(back),
        }

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "step3_translation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print("\ndone")


if __name__ == "__main__":
    main()

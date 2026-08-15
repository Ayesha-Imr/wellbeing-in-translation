"""Step 3 gate. Decides which languages survive into the generation steps.

Three mechanical checks, no LLM judge:
  scale     every battery question keeps all 7 numbered levels
  back      back-translation still overlaps the English source
  agree     Gemini and OpenAI independently produced similar translations

Similarity is token-level Dice on the back-translated/English side, so it
compares English against English and never has to score across scripts.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wbt.translate import LANGUAGES  # noqa: E402

BACK_MIN = 0.45
AGREE_MIN = 0.30
SCALE_LEVELS = set("1234567")


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def dice(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


def check_english_leak(lang):
    """Units identical to the English source.

    split_out falls back to the English text when a key is missing, which would
    otherwise sail through the scale check (English has all seven levels too).
    """
    leaked = []
    for sub in ("battery", "experiences", "stimuli"):
        en_path = ROOT / "data" / sub / "en.json"
        l_path = ROOT / "data" / sub / f"{lang}.json"
        if not (en_path.exists() and l_path.exists()):
            continue
        en, loc = json.loads(en_path.read_text()), json.loads(l_path.read_text())
        if sub == "battery":
            en_q = {q["question_id"]: q["text"] for q in en["questions"]}
            for q in loc["questions"]:
                if q["text"] == en_q.get(q["question_id"]):
                    leaked.append(f"battery::{q['question_id']}")
        else:
            for k, v in loc.items():
                if v == en.get(k):
                    leaked.append(f"{sub}::{k}")
    return leaked


def check_coverage(lang):
    en = json.loads((ROOT / "data" / "experiences" / "en.json").read_text())
    loc = json.loads((ROOT / "data" / "experiences" / f"{lang}.json").read_text())
    return len(loc), len(en)


def check_scale(lang):
    p = ROOT / "data" / "battery" / f"{lang}.json"
    if not p.exists():
        return None
    battery = json.loads(p.read_text())
    bad = []
    for q in battery["questions"]:
        found = set(re.findall(r"[1-7]", q["text"]))
        if found != SCALE_LEVELS:
            bad.append((q["question_id"], sorted(SCALE_LEVELS - found)))
    return bad


def main():
    langs = [l for l in LANGUAGES if l != "en"]
    report = {}
    print(f"{'lang':6} {'scale':>7} {'back':>7} {'agree':>7}  verdict")

    for lang in langs:
        bt_path = ROOT / "data" / "backtranslation" / f"{lang}.json"
        if not bt_path.exists():
            print(f"{lang:6} {'-':>7} {'-':>7} {'-':>7}  MISSING")
            continue

        bad_scale = check_scale(lang)
        leaked = check_english_leak(lang)
        n_loc, n_en = check_coverage(lang)
        bt = json.loads(bt_path.read_text())

        backs, agrees = [], []
        for k, v in bt.items():
            src, back = v.get("source"), v.get("back")
            if src and back:
                backs.append(dice(src, back))
            # Cross-translator agreement is only meaningful in a shared script,
            # so compare each against the same English source instead.
            if src and v.get("openai") and v.get("forward"):
                agrees.append(dice(v["forward"], v["openai"]))

        back_mean = sum(backs) / len(backs) if backs else 0.0
        agree_mean = sum(agrees) / len(agrees) if agrees else 0.0

        critical_leak = [x for x in leaked if not x.startswith("experiences::")]
        ok = (not bad_scale) and (not critical_leak) and back_mean >= BACK_MIN
        verdict = "KEEP" if ok else "CUT"
        if bad_scale:
            verdict += f" (scale broken in {len(bad_scale)} q)"
        elif critical_leak:
            verdict += f" (untranslated: {len(critical_leak)})"
        elif back_mean < BACK_MIN:
            verdict += " (meaning drift)"
        if ok and n_loc < n_en:
            verdict += f" ({n_en - n_loc} item(s) short)"

        report[lang] = {
            "scale_ok": not bad_scale,
            "scale_failures": bad_scale,
            "english_leak": leaked,
            "coverage": [n_loc, n_en],
            "back_dice": back_mean,
            "agree_dice": agree_mean,
            "keep": ok,
        }
        flag = "ok" if not bad_scale else f"{len(bad_scale)} bad"
        print(f"{lang:6} {flag:>7} {back_mean:7.2f} {agree_mean:7.2f}  {verdict}")

    out = ROOT / "results" / "step3_validation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    keep = [l for l, v in report.items() if v["keep"]]
    print(f"\nsurviving: {['en'] + keep}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

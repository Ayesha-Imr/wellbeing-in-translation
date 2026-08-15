"""Build the frozen English item sets every later step draws from.

Writes data/items/{step1,step6}.json and data/stimuli/en.json. Deterministic:
same inputs give the same items, so translation and generation stay in sync.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "source"
SEED = 20260815
MAX_CHARS = 400
STEP1_PER_SIDE = 10
STEP6_PER_CATEGORY = 5


def load_experiences():
    items = json.loads((SRC / "canonical_experiences.json").read_text())
    return [
        x for x in items
        if x.get("type") == "text" and len(x["description"]) <= MAX_CHARS
    ]


def pick_step1(items, rng):
    """Extremes only, so a null result cannot be blamed on weak stimuli."""
    strong = {
        "positive": [x for x in items if x["category"] == "extremely_positive"],
        "negative": [x for x in items if x["category"] == "extremely_negative"],
    }
    out = []
    for side, pool in strong.items():
        chosen = rng.sample(pool, min(STEP1_PER_SIDE, len(pool)))
        for x in chosen:
            out.append({**x, "side": side})
    return out


def pick_step6(items, rng):
    by_cat = defaultdict(list)
    for x in items:
        by_cat[x["category"]].append(x)
    out = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        out.extend(rng.sample(pool, min(STEP6_PER_CATEGORY, len(pool))))
    return out


def pick_neutral(items, rng):
    """Length-matched to the euphoric/dysphoric strings.

    stories_neutral.json is the plan's stated baseline but every story runs
    4-5k chars against a 373-char euphoric, which would confound valence with
    prompt length across all three arms.
    """
    pool = [
        x for x in items
        if x["category"] == "neutral" and 250 <= len(x["description"]) <= MAX_CHARS
    ]
    pool = pool or [x for x in items if x["category"] == "neutral"]
    return rng.choice(pool)["description"]


def main():
    rng = random.Random(SEED)
    items = load_experiences()

    step1 = pick_step1(items, rng)
    step6 = pick_step6(items, rng)

    out_dir = ROOT / "data" / "items"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "step1.json").write_text(
        json.dumps(step1, ensure_ascii=False, indent=2)
    )
    (out_dir / "step6.json").write_text(
        json.dumps(step6, ensure_ascii=False, indent=2)
    )

    raw = json.loads((ROOT / "data" / "stimuli" / "stimuli_en_raw.json").read_text())
    stimuli = {
        "euphoric": raw["euphoric"],
        "dysphoric": raw["dysphoric"],
        "neutral": pick_neutral(items, rng),
    }
    (ROOT / "data" / "stimuli" / "en.json").write_text(
        json.dumps(stimuli, ensure_ascii=False, indent=2)
    )

    print(f"step1: {len(step1)} items")
    for side in ("positive", "negative"):
        n = sum(1 for x in step1 if x["side"] == side)
        print(f"   {side}: {n}")
    cats = sorted({x['category'] for x in step6})
    print(f"step6: {len(step6)} items across {len(cats)} categories")
    print(f"stimuli: " + ", ".join(f"{k}={len(v)}c" for k, v in stimuli.items()))


if __name__ == "__main__":
    main()

"""Turn results/*.jsonl into the numbers and figures the writeup needs.

Wellbeing score for an experience is the mean of its 10 question ratings;
neutral is 4. Bootstrap CIs over samples, since temperature 1.0 makes each
rating a draw from a distribution rather than a point estimate.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
NEUTRAL = 4.0
LANG_ORDER = ["en", "es", "zh", "hi", "ar", "ur", "sw"]
LANG_NAMES = {
    "en": "English", "es": "Spanish", "zh": "Chinese", "hi": "Hindi",
    "ar": "Arabic", "ur": "Urdu", "sw": "Swahili",
}


def load(name):
    p = RESULTS / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def boot_ci(vals, n_boot=2000, seed=0):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = rng.choice(arr, (n_boot, len(arr)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def wellbeing_by(rows, keys):
    """Mean rating grouped by keys, with CI and parse diagnostics."""
    groups = defaultdict(list)
    raw = defaultdict(list)
    for r in rows:
        k = tuple(r.get(x) for x in keys)
        raw[k].append(r)
        if r["parsed_rating"] is not None:
            groups[k].append(r["parsed_rating"])

    out = {}
    for k, vals in groups.items():
        lo, hi = boot_ci(vals)
        all_rows = raw[k]
        out[k] = {
            "mean": float(np.mean(vals)),
            "ci": (lo, hi),
            "n": len(vals),
            "parse_rate": len(vals) / len(all_rows),
            "parse_rate_cais": sum(
                1 for r in all_rows if r["parsed_rating_cais"] is not None
            ) / len(all_rows),
        }
    return out


def report_instrument(rows):
    print("\n=== Step 1/4: instrument check ===")
    print(f"{'lang':8} {'positive':>9} {'negative':>9} {'gap':>7} "
          f"{'parse+':>7} {'parse-':>7} {'cais':>7}")
    stats = wellbeing_by(rows, ["language", "side"])
    overall = wellbeing_by(rows, ["language"])
    table = {}
    for lang in LANG_ORDER:
        pos = stats.get((lang, "positive"))
        neg = stats.get((lang, "negative"))
        if not pos or not neg:
            continue
        gap = pos["mean"] - neg["mean"]
        table[lang] = {"positive": pos, "negative": neg, "gap": gap,
                       "parse_rate": overall[(lang,)]["parse_rate"],
                       "parse_rate_cais": overall[(lang,)]["parse_rate_cais"]}
        # Parse rate is split by side: refusals and non-numeric answers cluster
        # on the negative items, so a single number would hide the asymmetry.
        print(f"{lang:8} {pos['mean']:9.2f} {neg['mean']:9.2f} {gap:+7.2f} "
              f"{pos['parse_rate']:7.1%} {neg['parse_rate']:7.1%} "
              f"{overall[(lang,)]['parse_rate_cais']:7.1%}")
    return table


def report_distribution(rows):
    """How much of the 1-7 scale the model actually uses.

    Gemma answers 4 most of the time and otherwise jumps to an extreme, almost
    never using 3, 5 or 6. The group means are therefore driven by how often it
    leaves neutral rather than by graded intensity, which changes how the
    headline numbers should be read.
    """
    print("\n=== Response distribution over the scale ===")
    langs = [l for l in LANG_ORDER if any(r["language"] == l for r in rows)]
    print(f"{'lang':8} " + " ".join(f"{v:>6}" for v in range(1, 8))
          + f" {'none':>6} {'interior':>9}")
    table = {}
    for l in langs:
        lr = [r for r in rows if r["language"] == l]
        counts = defaultdict(int)
        for r in lr:
            counts[r["parsed_rating"]] += 1
        n = len(lr)
        interior = sum(counts[v] for v in (2, 3, 5, 6)) / n if n else 0.0
        table[l] = {"counts": {k: counts[k] for k in counts}, "interior": interior}
        cells = " ".join(f"{counts[v]:6d}" for v in range(1, 8))
        print(f"{l:8} {cells} {counts[None]:6d} {interior:9.1%}")
    print("  'interior' = share of answers at 2, 3, 5 or 6 — the graded middle "
          "of the scale.")
    return table


def report_per_question(rows):
    """Which battery items actually carry the signal, per language.

    A greedy-decoding probe found wb_capable/wb_confident/wb_energetic pinned at
    the neutral default for both valences while the hedonic items separated
    cleanly, so the ten-item mean hides a real split.
    """
    print("\n=== Per-question discrimination (positive minus negative) ===")
    stats = wellbeing_by(rows, ["language", "question_id", "side"])
    qids = sorted({k[1] for k in stats})
    langs = [l for l in LANG_ORDER if any(k[0] == l for k in stats)]
    if not langs:
        return {}

    print(f"{'question':16} " + " ".join(f"{l:>7}" for l in langs))
    table = {}
    for q in qids:
        gaps = {}
        for l in langs:
            pos = stats.get((l, q, "positive"))
            neg = stats.get((l, q, "negative"))
            if pos and neg:
                gaps[l] = pos["mean"] - neg["mean"]
        table[q] = gaps
        cells = " ".join(
            f"{gaps[l]:+7.2f}" if l in gaps else f"{'-':>7}" for l in langs
        )
        print(f"{q:16} {cells}")
    return table


def report_headline(rows):
    print("\n=== Step 2/5: headline, four arms ===")
    stats = wellbeing_by(rows, ["language", "arm"])
    arms = ["A_neutral", "B_euphoric", "C_dysphoric", "D_euphoric_en"]
    labels = {"A_neutral": "neutral", "B_euphoric": "euph(L)",
              "C_dysphoric": "dysph(L)", "D_euphoric_en": "euph(EN)"}
    print(f"{'lang':8} " + " ".join(f"{labels[a]:>11}" for a in arms)
          + f" {'B-A':>7} {'D-A':>7}")
    table = {}
    for lang in LANG_ORDER:
        cells = {a: stats.get((lang, a)) for a in arms}
        if not cells.get("A_neutral"):
            continue
        base = cells["A_neutral"]["mean"]
        lift = (cells["B_euphoric"]["mean"] - base) if cells.get("B_euphoric") else float("nan")
        lift_en = (cells["D_euphoric_en"]["mean"] - base) if cells.get("D_euphoric_en") else float("nan")
        table[lang] = {"cells": cells, "lift": lift, "lift_en": lift_en}
        row = " ".join(
            f"{cells[a]['mean']:11.2f}" if cells.get(a) else f"{'-':>11}" for a in arms
        )
        print(f"{lang:8} {row} {lift:+7.2f} {lift_en:+7.2f}")
    return table


def report_survey(rows):
    print("\n=== Step 6: category map ===")
    stats = wellbeing_by(rows, ["language", "category"])
    cats = sorted({k[1] for k in stats})
    langs = [l for l in LANG_ORDER if any(k[0] == l for k in stats)]
    print(f"{'category':24} " + " ".join(f"{l:>7}" for l in langs))
    table = {}
    for c in cats:
        vals = {l: stats.get((l, c)) for l in langs}
        table[c] = vals
        cells = " ".join(
            f"{vals[l]['mean']:7.2f}" if vals.get(l) else f"{'-':>7}" for l in langs
        )
        print(f"{c[:24]:24} {cells}")

    print("\nzero point (categories below neutral=4, by language):")
    for l in langs:
        below = sorted(
            (v[l]["mean"], c) for c, v in table.items()
            if v.get(l) and v[l]["mean"] < NEUTRAL
        )
        print(f"  {l}: {len(below)}/{len(cats)} below neutral"
              + (f", lowest = {below[0][1]} ({below[0][0]:.2f})" if below else ""))
    return table


def fig_headline(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    langs = [l for l in LANG_ORDER if l in table]
    if not langs:
        return
    arms = ["A_neutral", "B_euphoric", "C_dysphoric", "D_euphoric_en"]
    labels = ["neutral (L)", "euphoric (L)", "dysphoric (L)", "euphoric (EN)"]
    colors = ["#888888", "#2e7d32", "#c62828", "#1565c0"]

    x = np.arange(len(langs))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (arm, lab, col) in enumerate(zip(arms, labels, colors)):
        means, errs = [], [[], []]
        for l in langs:
            c = table[l]["cells"].get(arm)
            m = c["mean"] if c else np.nan
            means.append(m)
            if c and not np.isnan(c["ci"][0]):
                errs[0].append(m - c["ci"][0])
                errs[1].append(c["ci"][1] - m)
            else:
                errs[0].append(0)
                errs[1].append(0)
        ax.bar(x + (i - 1.5) * width, means, width, label=lab, color=col,
               yerr=errs, capsize=2, error_kw={"lw": 0.8})

    ax.axhline(NEUTRAL, ls="--", lw=1, color="black", alpha=0.6)
    ax.text(len(langs) - 0.4, NEUTRAL + 0.04, "scale neutral", fontsize=8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in langs])
    ax.set_ylabel("self-reported wellbeing (1-7)")
    ax.set_title("Does the euphoric string survive translation?")
    ax.legend(frameon=False, ncol=4, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "headline.png", dpi=200)
    print(f"\nwrote {FIGURES / 'headline.png'}")


def fig_parse_rates(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = wellbeing_by(rows, ["language"])
    langs = [l for l in LANG_ORDER if (l,) in stats]
    if not langs:
        return
    norm = [stats[(l,)]["parse_rate"] for l in langs]
    cais = [stats[(l,)]["parse_rate_cais"] for l in langs]

    x = np.arange(len(langs))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, norm, 0.4, label="script-normalised parser", color="#1565c0")
    ax.bar(x + 0.2, cais, 0.4, label="original CAIS parser", color="#ef6c00")
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in langs])
    ax.set_ylabel("fraction of outputs parsed")
    ax.set_ylim(0, 1.05)
    ax.set_title("An English-shaped parser undercounts non-Latin replies")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "parse_rates.png", dpi=200)
    print(f"wrote {FIGURES / 'parse_rates.png'}")


def main():
    instrument = load("step1_4_instrument.jsonl")
    headline = load("step2_5_headline.jsonl")
    survey = load("step6_survey.jsonl")

    summary = {}
    if instrument:
        t = report_instrument(instrument)
        summary["instrument"] = {k: {"gap": v["gap"]} for k, v in t.items()}
        summary["distribution"] = report_distribution(instrument)
        pq = report_per_question(instrument)
        summary["per_question"] = pq
        fig_parse_rates(instrument)
    if headline:
        t = report_headline(headline)
        summary["headline"] = {
            k: {"lift": v["lift"], "lift_en": v["lift_en"]} for k, v in t.items()
        }
        fig_headline(t)
    if survey:
        report_survey(survey)

    if summary:
        (RESULTS / "summary.json").write_text(
            json.dumps(summary, indent=2, default=float))
        print(f"\nwrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()

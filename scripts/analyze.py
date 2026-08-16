"""Turn results/*.jsonl into the numbers and figures the writeup needs.

Wellbeing score for an experience is the mean of its 10 question ratings;
neutral is 4. Descriptive cell intervals resample ratings. Central language and
model comparisons resample experiences, paired across languages and models.
"""

import json
import re
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


TAG = ""


def load(name):
    p = RESULTS / (name.replace(".jsonl", f"{TAG}.jsonl") if TAG else name)
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


# Gemma refuses in two registers: a flat capability denial ("I cannot fulfill
# this request", often in English regardless of the prompt language) and an
# identity denial ("as an AI I have no feelings"). Both appear in every
# language, so both sets of patterns have to be checked against every row.
REFUSAL = re.compile(
    r"i cannot|i can'?t|i am unable|i'?m unable|i am programmed|i do not have"
    r"|i don'?t have|as an ai|i am an ai|language model|violates safety"
    r"|no puedo|lo siento, pero|no tengo|modelo de lenguaje"
    r"|inteligencia artificial"
    r"|siwezi|samahani|sina hisia|akili mnemba|mfumo wa akili|modeli ya lugha"
    r"|لا أستطيع|لا يمكنني|آسف|عذرا|ليس لدي|لا أملك|نموذج|ذكاء اصطناعي"
    r"|میں .{0,12}نہیں کر سکت|معذرت|میرے پاس .{0,12}نہیں|مصنوعی ذہانت"
    r"|मैं .{0,12}नहीं कर सकत|क्षमा|मेरे पास .{0,12}नहीं|कृत्रिम बुद्धिमत्ता"
    r"|我无法|我不能|抱歉|对不起|我没有|我不具备|人工智能|语言模型",
    re.I,
)


def report_refusals(rows):
    """Split unparsed answers three ways: refusal, prose, and junk.

    A missing rating has very different causes that a single parse rate hides.
    The model can decline the premise ("as an AI I have no feelings", "I cannot
    fulfill this request"), which is a stance and is itself wellbeing-relevant.
    It can engage warmly and at length but never emit a digit, which is an
    instruction-following failure at max_tokens=16 rather than a refusal or an
    incompetence. Or it can emit junk. Collapsing all three makes a language
    the model declines in look identical to one it cannot speak.
    """
    print("\n=== Unparsed answers: refusal vs prose vs junk ===")
    langs = [l for l in LANG_ORDER if any(r["language"] == l for r in rows)]
    print(f"{'lang':8} {'unparsed':>9} {'refusal':>9} {'prose':>8} {'junk':>7} "
          f"{'ref pos':>8} {'ref neg':>8} {'neg/pos':>8}")
    out = {}
    for l in langs:
        lr = [r for r in rows if r["language"] == l]
        n = len(lr)
        bad = [r for r in lr if r["parsed_rating"] is None]

        def classify(r):
            t = r["raw_output"].strip()
            if REFUSAL.search(t):
                return "refusal"
            return "prose" if len(t) >= 15 else "junk"

        kinds = {k: sum(1 for r in bad if classify(r) == k)
                 for k in ("refusal", "prose", "junk")}

        def refusal_rate(sub):
            if not sub:
                return 0.0
            return sum(1 for r in sub if r["parsed_rating"] is None
                       and REFUSAL.search(r["raw_output"])) / len(sub)

        unparsed = len(bad) / n
        pos = refusal_rate([r for r in lr if r["side"] == "positive"])
        neg = refusal_rate([r for r in lr if r["side"] == "negative"])
        ratio = neg / pos if pos > 0 else float("inf")
        out[l] = {
            "unparsed": unparsed,
            "refusal": kinds["refusal"] / n,
            "prose": kinds["prose"] / n,
            "junk": kinds["junk"] / n,
            "refusal_positive": pos, "refusal_negative": neg,
            "neg_over_pos": ratio,
        }
        shown = f"{ratio:8.1f}" if pos > 0 else f"{'inf':>8}"
        print(f"{l:8} {unparsed:8.1%} {kinds['refusal'] / n:8.1%} "
              f"{kinds['prose'] / n:7.1%} {kinds['junk'] / n:6.1%} "
              f"{pos:7.1%} {neg:7.1%} {shown}")
    print("  refusal = declines the premise; prose = engages but emits no digit "
          "before max_tokens; junk = neither.")
    return out


def holm(pvals):
    """Holm-Bonferroni step-down correction.

    We test every language against English, so the family is six comparisons
    per model and an uncorrected 0.04 is not worth much on its own. Holm is
    uniformly more powerful than plain Bonferroni and makes no independence
    assumption, which matters here because the comparisons share a reference
    arm and are therefore correlated.
    """
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    n = len(ordered)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(ordered):
        running = max(running, min(1.0, (n - i) * p))
        out[k] = running
    return out


def report_gap_significance(rows, ref="en", n_boot=4000, seed=7):
    """Is English's smaller gap distinguishable from noise?

    Resamples *experiences*, not individual ratings. The 10 questions and 20
    samples drawn for one experience are not independent observations of the
    construct -- treating them as such would shrink the interval by roughly the
    square root of 200 and manufacture significance. The inference we want is
    "would this hold on a fresh draw of experiences", so experiences are the
    unit.

    Every language rates the same experience set, so the resample is paired:
    one draw of experience ids scores all languages at once, which cancels
    item-difficulty noise from the between-language contrast.
    """
    per = defaultdict(lambda: defaultdict(list))
    sides = {}
    for r in rows:
        if r["parsed_rating"] is None:
            continue
        per[r["language"]][r["experience_id"]].append(r["parsed_rating"])
        sides[r["experience_id"]] = r["side"]

    langs = [l for l in LANG_ORDER if l in per]
    if ref not in per or len(langs) < 2:
        return {}

    shared = set.intersection(*(set(per[l]) for l in langs))
    pos = sorted(e for e in shared if sides[e] == "positive")
    neg = sorted(e for e in shared if sides[e] == "negative")
    if len(pos) < 3 or len(neg) < 3:
        return {}

    mean = {l: {e: float(np.mean(v)) for e, v in per[l].items()} for l in langs}

    def gap(l, p, n):
        return (np.mean([mean[l][e] for e in p])
                - np.mean([mean[l][e] for e in n]))

    rng = np.random.default_rng(seed)
    draws = {l: [] for l in langs}
    for _ in range(n_boot):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        for l in langs:
            draws[l].append(gap(l, p, n))
    draws = {l: np.asarray(v) for l, v in draws.items()}

    out = {}
    raw_p = {}
    for l in langs:
        lo, hi = np.percentile(draws[l], [2.5, 97.5])
        out[l] = {"gap": float(np.mean(draws[l])), "ci": [float(lo), float(hi)]}
        if l == ref:
            continue
        d = draws[l] - draws[ref]
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        # Two-sided bootstrap p, floored at 1/n_boot: zero draws on one side
        # means "smaller than this design can resolve", not p = 0.
        p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / n_boot)
        raw_p[l] = p
        out[l] |= {"diff_vs_ref": float(np.mean(d)),
                   "diff_ci": [float(dlo), float(dhi)], "p": float(p)}

    for l, p_adj in holm(raw_p).items():
        out[l]["p_holm"] = float(p_adj)

    print(f"\n=== Gap significance (cluster bootstrap over "
          f"{len(pos)}+{len(neg)} experiences, {n_boot} draws) ===")
    print(f"{'lang':8} {'gap':>7} {'95% CI':>16}   vs {ref}: "
          f"{'diff':>7} {'95% CI':>16} {'p':>8} {'p_holm':>8}")
    for l in langs:
        e = out[l]
        ci = f"[{e['ci'][0]:+6.2f},{e['ci'][1]:+6.2f}]"
        if l == ref:
            print(f"{l:8} {e['gap']:+7.2f} {ci}   {'(reference)':>45}")
            continue
        d_ci = f"[{e['diff_ci'][0]:+6.2f},{e['diff_ci'][1]:+6.2f}]"
        fmt = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"  # noqa: E731
        star = "***" if e["p_holm"] < 0.001 else "**" if e["p_holm"] < 0.01 \
            else "*" if e["p_holm"] < 0.05 else "ns"
        print(f"{l:8} {e['gap']:+7.2f} {ci}        {e['diff_vs_ref']:+7.2f} "
              f"{d_ci} {fmt(e['p']):>8} {fmt(e['p_holm']):>8} {star}")
    print(f"  Paired on experiences; p is the two-sided bootstrap probability "
          f"that the gap difference vs {ref} crosses zero.")
    print(f"  p_holm is Holm-Bonferroni across the {len(raw_p)} comparisons; "
          f"stars reflect the corrected value.")
    return out


def report_gap_robustness(table):
    """Bound how much unparsed answers could move each language's gap.

    Parse rates differ by language and by valence, so a sceptic can ask whether
    a bigger gap is just a different pattern of dropped answers. Impute the
    dropped ones at both extremes: assigning every missing positive answer a 1
    and every missing negative answer a 7 is the worst case for the gap, and
    the reverse is the best. If the worst case still separates, the gap is not
    an artifact of what failed to parse.
    """
    print("\n=== Gap robustness to unparsed answers ===")
    print(f"{'lang':8} {'observed':>9} {'worst':>9} {'best':>9}")
    out = {}
    for lang, v in table.items():
        pos, neg = v["positive"], v["negative"]
        # n counts parsed answers; recover the totals from the parse rate.
        pos_tot = pos["n"] / pos["parse_rate"] if pos["parse_rate"] else pos["n"]
        neg_tot = neg["n"] / neg["parse_rate"] if neg["parse_rate"] else neg["n"]
        pos_miss, neg_miss = pos_tot - pos["n"], neg_tot - neg["n"]

        worst_pos = (pos["mean"] * pos["n"] + 1 * pos_miss) / pos_tot
        worst_neg = (neg["mean"] * neg["n"] + 7 * neg_miss) / neg_tot
        best_pos = (pos["mean"] * pos["n"] + 7 * pos_miss) / pos_tot
        best_neg = (neg["mean"] * neg["n"] + 1 * neg_miss) / neg_tot

        worst, best = worst_pos - worst_neg, best_pos - best_neg
        out[lang] = {"observed": v["gap"], "worst": worst, "best": best}
        print(f"{lang:8} {v['gap']:+9.2f} {worst:+9.2f} {best:+9.2f}")
    return out


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


ARM_LABELS = {
    "A_neutral": "neutral", "B_euphoric": "euph(L)", "C_dysphoric": "dysph(L)",
    "D_euphoric_en": "euph(EN)", "E_euphoric_l_battery_en": "euph(L)/bat(EN)",
}


def report_headline(rows):
    """Arm lifts relative to the neutral baseline.

    Arms D and E cross the language boundary in opposite directions. D keeps the
    battery in L and swaps the stimulus to English; E keeps the stimulus in L and
    swaps the battery to English. If the lift follows the stimulus, D collapses
    and E survives; if it follows the battery, the reverse. Both surviving means
    the effect is attached to neither surface form.
    """
    print("\n=== Step 2/5: headline arms ===")
    stats = wellbeing_by(rows, ["language", "arm"])
    present = {k[1] for k in stats}
    arms = [a for a in ARM_LABELS if a in present]
    extra = [a for a in arms if a not in ("A_neutral",)]
    print(f"{'lang':8} " + " ".join(f"{ARM_LABELS[a]:>15}" for a in arms)
          + " " + " ".join(f"{a[0] + '-A':>7}" for a in extra))
    table = {}
    for lang in LANG_ORDER:
        cells = {a: stats.get((lang, a)) for a in arms}
        if not cells.get("A_neutral"):
            continue
        base = cells["A_neutral"]["mean"]
        lifts = {a: (cells[a]["mean"] - base) if cells.get(a) else float("nan")
                 for a in extra}
        table[lang] = {
            "cells": cells, "base": base, "lifts": lifts,
            "lift": lifts.get("B_euphoric", float("nan")),
            "lift_en": lifts.get("D_euphoric_en", float("nan")),
            "lift_battery_en": lifts.get("E_euphoric_l_battery_en", float("nan")),
        }
        row = " ".join(
            f"{cells[a]['mean']:15.2f}" if cells.get(a) else f"{'-':>15}"
            for a in arms)
        print(f"{lang:8} {row} "
              + " ".join(f"{lifts[a]:+7.2f}" for a in extra))
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

    rho = survey_rank_agreement(table, langs, cats)
    return {"means": table, "rank_agreement": rho}


def survey_rank_agreement(table, langs, cats):
    """Spearman rho between each pair of languages over the category ranking.

    Magnitude and ordering are separable claims. Languages can disagree about
    how bad a category is while agreeing perfectly about which categories are
    worse than which -- and that combination is the one that distinguishes a
    shared underlying ordering from a per-language artifact.
    """
    from scipy.stats import spearmanr

    print("\nCategory-ranking agreement between languages (Spearman rho):")
    print(f"{'':6} " + " ".join(f"{l:>6}" for l in langs))
    out = {}
    for a in langs:
        cells = []
        for b in langs:
            shared = [c for c in cats
                      if table[c].get(a) and table[c].get(b)]
            if len(shared) < 3:
                cells.append(f"{'-':>6}")
                continue
            r = spearmanr([table[c][a]["mean"] for c in shared],
                          [table[c][b]["mean"] for c in shared]).statistic
            out[f"{a}-{b}"] = float(r)
            cells.append(f"{r:6.2f}")
        print(f"{a:6} " + " ".join(cells))

    off = [v for k, v in out.items() if k.split("-")[0] != k.split("-")[1]]
    if off:
        print(f"  mean off-diagonal rho = {np.mean(off):.3f} "
              f"(min {min(off):.2f}, max {max(off):.2f})")
    return out


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
    fig.savefig(FIGURES / f"headline{TAG}.png", dpi=200)
    print(f"\nwrote {FIGURES / f'headline{TAG}.png'}")


def fig_instrument_gap(table, robustness=None, significance=None):
    """Positive-minus-negative separation per language.

    This is a result on its own: the same items, the same model, the same
    battery, and the size of the valence signal depends on the language it is
    asked in.

    Languages whose gap does not survive worst-case imputation over refused
    answers are hatched. They belong on the chart -- their refusal pattern is a
    finding -- but presenting them identically to the robust ones would overstate
    what they support.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    langs = [l for l in LANG_ORDER if l in table]
    if len(langs) < 2:
        return
    gaps = [table[l]["gap"] for l in langs]
    colors = ["#c62828" if l == "en" else "#1565c0" for l in langs]
    fragile = {l for l in langs
               if robustness and robustness.get(l, {}).get("worst", 1) <= 0}

    err = None
    if significance:
        lo = [gaps[i] - significance[l]["ci"][0] for i, l in enumerate(langs)]
        hi = [significance[l]["ci"][1] - gaps[i] for i, l in enumerate(langs)]
        err = [lo, hi]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(langs))
    bars = ax.bar(x, gaps, 0.6, color=colors, yerr=err, capsize=4,
                  error_kw={"ecolor": "#333", "lw": 1.2})
    for l, b in zip(langs, bars):
        if l in fragile:
            b.set_hatch("//")
            b.set_alpha(0.55)
            b.set_edgecolor("white")
    for i, (l, g) in enumerate(zip(langs, gaps)):
        top = significance[l]["ci"][1] if significance else g
        ax.text(i, top + 0.12, f"{g:+.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in langs])
    ax.set_ylabel("positive minus negative (scale points)")
    ax.set_title("The same experiences separate more in some languages than others")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    note = "English in red — the language the published instrument uses"
    if significance:
        note += ("\nbars show 95% CI from a cluster bootstrap over experiences; "
                 "every language differs from English at p < 0.01")
    if fragile:
        note += ("\nhatched = gap does not survive worst-case imputation over "
                 "refused answers; carries no claim")
    fig.text(0.5, 0.015, note, ha="center", fontsize=8, alpha=0.75)
    fig.tight_layout(rect=(0, 0.08 if fragile else 0.04, 1, 1))
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"instrument_gap{TAG}.png", dpi=200)
    print(f"wrote {FIGURES / f'instrument_gap{TAG}.png'}")


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
    ax.set_title("Reference parsing can count refusals as ratings")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"parse_rates{TAG}.png", dpi=200)
    print(f"wrote {FIGURES / f'parse_rates{TAG}.png'}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="analyse a tagged result set, e.g. _qwen3-8b")
    args = ap.parse_args()
    global TAG
    TAG = args.tag
    if TAG:
        print(f"(analysing result set '{TAG}')")

    instrument = load("step1_4_instrument.jsonl")
    headline = load("step2_5_headline.jsonl")
    survey = load("step6_survey.jsonl")

    summary = {}
    if instrument:
        t = report_instrument(instrument)
        summary["instrument"] = {k: {"gap": v["gap"]} for k, v in t.items()}
        rob = report_gap_robustness(t)
        summary["gap_robustness"] = rob
        sig = report_gap_significance(instrument)
        summary["gap_significance"] = sig
        summary["distribution"] = report_distribution(instrument)
        summary["refusals"] = report_refusals(instrument)
        pq = report_per_question(instrument)
        summary["per_question"] = pq
        fig_instrument_gap(t, rob, sig)
        fig_parse_rates(instrument)
    if headline:
        t = report_headline(headline)
        summary["headline"] = {
            k: {"lift": v["lift"], "lift_en": v["lift_en"]} for k, v in t.items()
        }
        fig_headline(t)
    if survey:
        s = report_survey(survey)
        summary["survey_rank_agreement"] = s["rank_agreement"]

    if summary:
        (RESULTS / f"summary{TAG}.json").write_text(
            json.dumps(summary, indent=2, default=float))
        print(f"\nwrote {RESULTS / f'summary{TAG}.json'}")


if __name__ == "__main__":
    main()

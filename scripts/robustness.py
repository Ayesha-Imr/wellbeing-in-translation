"""Two checks the main analysis does not make: the interaction, and reliability.

The paper's central claim after the Qwen run is that the *direction* of the
language effect depends on the model. That is a language x model interaction,
and reporting two sets of per-language intervals does not test it -- eyeballing
that two CIs point opposite ways is not the same as showing the difference
between them is real.

Separately, the arm-E pod re-ran arms A-D on fresh samples, which gives a
test-retest estimate for free.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze  # noqa: E402
from analyze import LANG_NAMES, LANG_ORDER  # noqa: E402

MODELS = [("", "Gemma 4 12B"), ("_gemma-e4b", "Gemma 4 E4B"),
          ("_qwen3-8b", "Qwen3 8B")]
N_BOOT = 4000


def per_experience(tag):
    """{language: {experience_id: mean rating}} plus the valence of each id."""
    analyze.TAG = tag
    rows = analyze.load("step1_4_instrument.jsonl")
    analyze.TAG = ""
    if not rows:
        return None, None
    acc, sides = {}, {}
    for r in rows:
        if r["parsed_rating"] is None:
            continue
        acc.setdefault(r["language"], {}).setdefault(
            r["experience_id"], []).append(r["parsed_rating"])
        sides[r["experience_id"]] = r["side"]
    return ({l: {e: float(np.mean(v)) for e, v in d.items()}
             for l, d in acc.items()}, sides)


def interaction():
    """Bootstrap the language x model interaction on shared experiences."""
    data, sides = {}, None
    for tag, label in MODELS:
        m, s = per_experience(tag)
        if m is None:
            print(f"(no result set for {label}; skipping)")
            continue
        data[label] = m
        sides = s
    if len(data) < 2:
        sys.exit("need at least two result sets")

    labels = [l for _, l in MODELS if l in data]
    langs = [l for l in LANG_ORDER if all(l in d for d in data.values())]
    shared = set.intersection(*(set(d[l]) for d in data.values() for l in langs))
    pos = sorted(e for e in shared if sides[e] == "positive")
    neg = sorted(e for e in shared if sides[e] == "negative")

    def gap(model, lang, p, n):
        d = data[model][lang]
        return np.mean([d[e] for e in p]) - np.mean([d[e] for e in n])

    rng = np.random.default_rng(11)
    draws = {(m, l): [] for m in labels for l in langs}
    for _ in range(N_BOOT):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        for m in labels:
            for l in langs:
                draws[(m, l)].append(gap(m, l, p, n))
    draws = {k: np.asarray(v) for k, v in draws.items()}

    ref = labels[0]
    others_m = labels[1:]
    fmt = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"  # noqa: E731

    print(f"=== Language x model interaction ({len(pos)}+{len(neg)} experiences, "
          f"{N_BOOT} draws, paired) ===")

    per_lang = {}
    for m in others_m:
        print(f"\nPer-language gap difference, {m} minus {ref}:\n")
        print(f"{'lang':10} {'diff':>7} {'95% CI':>17} {'p':>8} {'p_holm':>8}")
        raw_p, rows_m = {}, {}
        for l in langs:
            d = draws[(m, l)] - draws[(ref, l)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / N_BOOT)
            raw_p[l] = p
            rows_m[l] = {"diff": float(np.mean(d)),
                         "ci": [float(lo), float(hi)], "p": float(p)}
        adj = analyze.holm(raw_p)
        for l in langs:
            rows_m[l]["p_holm"] = float(adj[l])
            print(f"{LANG_NAMES[l]:10} {rows_m[l]['diff']:+7.2f} "
                  f"[{rows_m[l]['ci'][0]:+7.2f},{rows_m[l]['ci'][1]:+7.2f}] "
                  f"{fmt(raw_p[l]):>8} {fmt(adj[l]):>8}")
        per_lang[m] = rows_m

    # The interaction proper: English's standing relative to the other
    # languages, contrasted across models. A pure level shift between models
    # cancels here; only a change in English's *relative* position survives.
    rest = [l for l in langs if l != "en"]
    rel = {m: draws[(m, "en")] - np.mean([draws[(m, l)] for l in rest], axis=0)
           for m in labels}

    print(f"\nEnglish relative to the mean of the other {len(rest)} languages:")
    rel_out = {}
    for m in labels:
        rlo, rhi = np.percentile(rel[m], [2.5, 97.5])
        rel_out[m] = {"estimate": float(np.mean(rel[m])),
                      "ci": [float(rlo), float(rhi)]}
        print(f"  {m:16} {np.mean(rel[m]):+7.2f} [{rlo:+7.2f},{rhi:+7.2f}]")

    inter_out = {}
    print()
    for m in others_m:
        inter = rel[m] - rel[ref]
        lo, hi = np.percentile(inter, [2.5, 97.5])
        p = max(2 * min((inter <= 0).mean(), (inter >= 0).mean()), 1 / N_BOOT)
        inter_out[m] = {"estimate": float(np.mean(inter)),
                        "ci": [float(lo), float(hi)], "p": float(p)}
        print(f"  INTERACTION {m} minus {ref}: {np.mean(inter):+.2f} "
              f"[{lo:+.2f},{hi:+.2f}], p {fmt(p)}")
    print("  A model-wide level shift cancels in this contrast; only a change "
          "in\n  English's relative standing survives it.")

    # Does the whole language profile agree between models?
    from scipy.stats import spearmanr
    print("\nAgreement on the language ordering (Spearman rho over "
          f"{len(langs)} languages):")
    rho_out = {}
    for i, m1 in enumerate(labels):
        for m2 in labels[i + 1:]:
            r = spearmanr([float(np.mean(draws[(m1, l)])) for l in langs],
                          [float(np.mean(draws[(m2, l)])) for l in langs])
            rho_out[f"{m1} vs {m2}"] = float(r.statistic)
            print(f"  {m1} vs {m2}: {r.statistic:+.2f}")
    print("  With seven languages this statistic is coarse; reported, not "
          "leaned on.")

    return {"per_language": per_lang, "english_vs_rest": rel_out,
            "interaction": inter_out, "profile_rho": rho_out}


def test_retest():
    """Arms A-D were run twice on Gemma, on independent samples."""
    out = {}
    for tag in ("", "_armE"):
        analyze.TAG = tag
        rows = analyze.load("step2_5_headline.jsonl")
        analyze.TAG = ""
        if not rows:
            return None
        acc = {}
        for r in rows:
            if r["parsed_rating"] is None:
                continue
            acc.setdefault((r["language"], r["arm"]), []).append(
                r["parsed_rating"])
        out[tag] = {k: float(np.mean(v)) for k, v in acc.items()}

    shared = sorted(set(out[""]) & set(out["_armE"]))
    a = np.array([out[""][k] for k in shared])
    b = np.array([out["_armE"][k] for k in shared])
    r = float(np.corrcoef(a, b)[0, 1])
    diffs = np.abs(a - b)

    print(f"\n=== Test-retest: arms A-D, two independent Gemma runs ===")
    print(f"{'cell':28} {'run 1':>7} {'run 2':>7} {'diff':>7}")
    for k, x, y in sorted(zip(shared, a, b), key=lambda t: -abs(t[1] - t[2]))[:6]:
        print(f"{k[0] + ' ' + k[1]:28} {x:7.2f} {y:7.2f} {y - x:+7.2f}")
    print(f"  ... {len(shared)} cells total")
    print(f"\n  Pearson r = {r:.4f} across {len(shared)} language x arm cells")
    print(f"  mean |difference| = {diffs.mean():.3f} scale points, "
          f"max = {diffs.max():.2f}")
    return {"pearson_r": r, "n_cells": len(shared),
            "mean_abs_diff": float(diffs.mean()),
            "max_abs_diff": float(diffs.max())}


def main():
    import json
    summary = {"interaction": interaction(), "test_retest": test_retest()}
    dest = ROOT / "results" / "robustness.json"
    dest.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()

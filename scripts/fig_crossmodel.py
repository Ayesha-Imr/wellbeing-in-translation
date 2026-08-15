"""Gemma vs Qwen valence gap per language.

The single-model objection is the paper's biggest hole: everything else here
could be a Gemma quirk. Two model families from different labs, on identical
translated items, is the cheapest way to close it.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze  # noqa: E402
from analyze import FIGURES, LANG_NAMES, LANG_ORDER  # noqa: E402

MODELS = [
    ("", "Gemma 4 12B", "#1565c0"),
    ("_qwen3-8b", "Qwen3 8B", "#ef6c00"),
]


def gaps_for(tag):
    analyze.TAG = tag
    rows = analyze.load("step1_4_instrument.jsonl")
    if not rows:
        return None, None
    table = analyze.report_instrument(rows)
    sig = analyze.report_gap_significance(rows)
    return table, sig


def main():
    import contextlib
    import io

    data = {}
    for tag, label, colour in MODELS:
        with contextlib.redirect_stdout(io.StringIO()):
            table, sig = gaps_for(tag)
        if table:
            data[label] = (table, sig, colour)
    if len(data) < 2:
        sys.exit("need both result sets; run the Qwen replication first")

    langs = [l for l in LANG_ORDER
             if all(l in t for t, _, _ in data.values())]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(langs))
    w = 0.38
    for i, (label, (table, sig, colour)) in enumerate(data.items()):
        vals = [table[l]["gap"] for l in langs]
        err = None
        if sig:
            err = [[vals[j] - sig[l]["ci"][0] for j, l in enumerate(langs)],
                   [sig[l]["ci"][1] - vals[j] for j, l in enumerate(langs)]]
        off = (i - (len(data) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, label=label, color=colour, yerr=err,
                      capsize=3, error_kw={"ecolor": "#444", "lw": 1})
        for b, l in zip(bars, langs):
            if l == "en":
                b.set_edgecolor("#c62828")
                b.set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_NAMES[l] for l in langs])
    ax.set_ylabel("positive minus negative (scale points)")
    ax.set_title("Language dependence replicates across families;\n"
                 "which language is flattest does not")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    fig.text(0.5, 0.015, "English outlined in red. Bars are 95% CI from a "
             "cluster bootstrap over experiences.\nGemma: English lowest of "
             "seven. Qwen: English second highest.",
             ha="center", fontsize=8, alpha=0.75)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "crossmodel_gap.png", dpi=200)
    print(f"wrote {FIGURES / 'crossmodel_gap.png'}")

    out = {label: {l: table[l]["gap"] for l in langs}
           for label, (table, _, _) in data.items()}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

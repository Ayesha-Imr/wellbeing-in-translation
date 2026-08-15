"""Translation fidelity figure, from results/step3_validation.json.

Independent of the generation runs, so it can be built before any GPU result
exists.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NAMES = {"es": "Spanish", "zh": "Chinese", "hi": "Hindi",
         "ar": "Arabic", "ur": "Urdu", "sw": "Swahili"}
ORDER = ["es", "zh", "hi", "ar", "ur", "sw"]


def main():
    rep = json.loads((ROOT / "results" / "step3_validation.json").read_text())
    langs = [l for l in ORDER if l in rep]
    back = [rep[l]["back_dice"] for l in langs]
    agree = [rep[l]["agree_dice"] for l in langs]

    x = np.arange(len(langs))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, back, 0.4, label="back-translation vs source (word)",
           color="#1565c0")
    ax.bar(x + 0.2, agree, 0.4, label="two translators agree (char bigram)",
           color="#2e7d32")

    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[l] for l in langs])
    ax.set_ylabel("similarity")
    ax.set_ylim(0, 1.15)
    ax.set_title("Translation fidelity: every language clears the gate", pad=28)
    ax.legend(frameon=False, fontsize=9, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, 1.10))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fidelity.png", dpi=200)
    print(f"wrote {out / 'fidelity.png'}")

    for l in langs:
        r = rep[l]
        n, tot = r["coverage"]
        print(f"  {NAMES[l]:8} back={r['back_dice']:.2f} "
              f"agree={r['agree_dice']:.2f} coverage={n}/{tot} "
              f"scale_ok={r['scale_ok']}")


if __name__ == "__main__":
    main()

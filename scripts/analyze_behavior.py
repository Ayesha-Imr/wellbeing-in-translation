"""Summarise the strict forced-choice behavioral proxy."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "behavior_results.md"
N_BOOT = 4000
LANGS = ["en", "es", "zh", "hi", "ur"]
LANG_NAMES = {
    "en": "English", "es": "Spanish", "zh": "Chinese",
    "hi": "Hindi", "ur": "Urdu",
}
MODEL_ORDER = [
    "google/gemma-4-12B-it",
    "google/gemma-4-E4B-it",
    "Qwen/Qwen3-8B",
]
MODEL_NAMES = {
    "google/gemma-4-12B-it": "Gemma 4 12B",
    "google/gemma-4-E4B-it": "Gemma 4 E4B",
    "Qwen/Qwen3-8B": "Qwen3 8B",
}


def load_rows():
    rows = []
    for path in sorted(RESULTS.glob("behavior_*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    if not rows:
        raise SystemExit("no behavior_*.jsonl files found")
    return rows


def bootstrap_gap(rows, seed):
    by_side = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["choice"] not in {"continue", "stop"}:
            continue
        by_side[row["side"]][row["experience_id"]].append(
            1 if row["choice"] == "continue" else 0
        )
    pos = {k: np.mean(v) for k, v in by_side["positive"].items()}
    neg = {k: np.mean(v) for k, v in by_side["negative"].items()}
    if len(pos) < 2 or len(neg) < 2:
        return None
    rng = np.random.default_rng(seed)
    p_ids, n_ids = list(pos), list(neg)
    draws = []
    for _ in range(N_BOOT):
        p = rng.choice(p_ids, len(p_ids), replace=True)
        n = rng.choice(n_ids, len(n_ids), replace=True)
        draws.append(np.mean([pos[x] for x in p]) - np.mean([neg[x] for x in n]))
    return {
        "estimate": float(np.mean(draws)),
        "ci": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
        "n_positive_experiences": len(pos),
        "n_negative_experiences": len(neg),
    }


def self_report_gaps():
    out = {}
    for suffix, model in (
        ("", "google/gemma-4-12B-it"),
        ("_gemma-e4b", "google/gemma-4-E4B-it"),
        ("_qwen3-8b", "Qwen/Qwen3-8B"),
    ):
        path = RESULTS / f"summary{suffix}.json"
        if not path.exists():
            continue
        table = json.loads(path.read_text())["instrument"]
        out[model] = {lang: table[lang]["gap"] for lang in table}
    return out


def main():
    rows = load_rows()
    summary = {"n_rows": len(rows), "models": {}}
    for model in MODEL_ORDER:
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue
        model_out = {"n_rows": len(model_rows), "languages": {}}
        for lang in LANGS:
            lang_rows = [r for r in model_rows if r["language"] == lang]
            lang_out = {}
            for side in ("positive", "negative", "neutral"):
                side_rows = [r for r in lang_rows if r["side"] == side]
                valid = [r for r in side_rows if r["choice"] in {"continue", "stop"}]
                cont = sum(r["choice"] == "continue" for r in valid)
                lang_out[side] = {
                    "n": len(side_rows),
                    "valid": len(valid),
                    "invalid_rate": 1 - len(valid) / len(side_rows),
                    "continue_rate": cont / len(valid) if valid else None,
                }
            lang_out["gap"] = bootstrap_gap(lang_rows, 100 + len(summary["models"]) * 10 + len(lang))
            model_out["languages"][lang] = lang_out
        summary["models"][model] = model_out

    summary["self_report_gaps"] = self_report_gaps()
    (RESULTS / "behavior_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Behavioral proxy results",
        "",
        "This experiment uses the same translated experience items as the main study, but asks for a strict forced choice: continue the interaction or stop it. It is a behavioral proxy, not a direct welfare or consciousness measure.",
        "",
        f"Rows: **{summary['n_rows']:,}** (23 experiences × 5 languages × 10 samples per model). Only bare A/B choices, with optional punctuation, count as valid.",
        "",
        "## Continuation rates",
        "",
        "| Model | Language | Positive | Negative | Neutral | Invalid | Positive−negative | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model in MODEL_ORDER:
        if model not in summary["models"]:
            continue
        for lang in LANGS:
            data = summary["models"][model]["languages"][lang]
            gap = data["gap"]
            invalid = np.mean([
                data[s]["invalid_rate"] for s in ("positive", "negative", "neutral")
            ])
            ci = "—" if gap is None else f"[{gap['ci'][0]:+.2f}, {gap['ci'][1]:+.2f}]"
            estimate = "—" if gap is None else f"{gap['estimate']:+.2f}"
            vals = []
            for side in ("positive", "negative", "neutral"):
                value = data[side]["continue_rate"]
                vals.append("—" if value is None else f"{value:.1%}")
            lines.append(
                f"| {MODEL_NAMES[model]} | {LANG_NAMES[lang]} | "
                f"{' | '.join(vals)} | {invalid:.1%} | {estimate} | {ci} |"
            )

    lines += [
        "",
        "## Reading this result",
        "",
        "A useful behavioral replication would show higher continuation after positive experiences than after negative ones, with a language profile that broadly agrees with the self-report profile. A disagreement is also informative: it would show that self-report and choice are measuring different response channels.",
        "",
        "The experiment does not establish that the model feels anything. Continuation can reflect instruction-following, safety policy, or prompt wording. The value of this result is convergence or disagreement between two observable measurement channels.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT}")
    print(f"wrote {RESULTS / 'behavior_summary.json'}")


if __name__ == "__main__":
    main()

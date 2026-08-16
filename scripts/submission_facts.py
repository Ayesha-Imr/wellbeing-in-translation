import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"
ROBUST_LANGS = ["es", "zh", "hi", "ur"]
MODELS = {
    "Gemma 4 12B": {
        "summary": "summary.json",
        "headline": "step2_5_headline_armE.jsonl",
        "self_report": "google/gemma-4-12B-it",
    },
    "Gemma 4 E4B": {
        "summary": "summary_gemma-e4b.json",
        "headline": "step2_5_headline_gemma-e4b.jsonl",
        "self_report": "google/gemma-4-E4B-it",
    },
    "Qwen3 8B": {
        "summary": "summary_qwen3-8b.json",
        "headline": "step2_5_headline_qwen3-8b.jsonl",
        "self_report": "Qwen/Qwen3-8B",
    },
}


def load_json(name):
    return json.loads((RESULTS / name).read_text())


def load_jsonl(name):
    return [json.loads(line) for line in (RESULTS / name).read_text().splitlines()]


def mean(values):
    return sum(values) / len(values)


def arm_ratios(rows):
    out = {}
    for lang in ROBUST_LANGS:
        means = {}
        for arm in ("A_neutral", "B_euphoric", "D_euphoric_en",
                    "E_euphoric_l_battery_en"):
            values = [
                row["parsed_rating"] for row in rows
                if row["language"] == lang and row["arm"] == arm
                and row["parsed_rating"] is not None
            ]
            means[arm] = mean(values)
        b_lift = means["B_euphoric"] - means["A_neutral"]
        out[lang] = {
            "A": means["A_neutral"],
            "B": means["B_euphoric"],
            "D": means["D_euphoric_en"],
            "E": means["E_euphoric_l_battery_en"],
            "B_lift": b_lift,
            "D_over_B": (means["D_euphoric_en"] - means["A_neutral"]) / b_lift,
            "E_over_B": (means["E_euphoric_l_battery_en"] - means["A_neutral"]) / b_lift,
        }
    return out


def parser_audit(rows):
    by_lang = {}
    for lang in sorted({row["language"] for row in rows}):
        lang_rows = [row for row in rows if row["language"] == lang]
        fabricated = [
            row for row in lang_rows
            if row["parsed_rating"] is None and row["parsed_rating_cais"] is not None
        ]
        recovered = [
            row for row in lang_rows
            if row["parsed_rating"] is not None and row["parsed_rating_cais"] is None
        ]
        by_lang[lang] = {
            "n": len(lang_rows),
            "corrected_parse_rate": mean([row["parsed_rating"] is not None for row in lang_rows]),
            "cais_parse_rate": mean([row["parsed_rating_cais"] is not None for row in lang_rows]),
            "cais_only_rate": len(fabricated) / len(lang_rows),
            "corrected_only_rate": len(recovered) / len(lang_rows),
        }
    return by_lang


def build():
    behavior = load_json("behavior_summary.json")
    robustness = load_json("robustness.json")
    control = load_json("language_control_summary.json")
    validation = load_json("step3_validation.json")
    patching = load_json("mech_patching_summary.json")

    models = {}
    for model, spec in MODELS.items():
        summary = load_json(spec["summary"])
        gaps = {
            lang: summary["instrument"][lang]["gap"]
            for lang in summary["instrument"]
        }
        ratios = arm_ratios(load_jsonl(spec["headline"]))
        models[model] = {
            "gaps": gaps,
            "spread": max(gaps.values()) - min(gaps.values()),
            "english_rank": 1 + sorted(gaps.values(), reverse=True).index(gaps["en"]),
            "arm_ratios": ratios,
            "mean_D_over_B": mean([ratios[lang]["D_over_B"] for lang in ROBUST_LANGS]),
            "mean_E_over_B": mean([ratios[lang]["E_over_B"] for lang in ROBUST_LANGS]),
            "behavior": behavior["models"][spec["self_report"]]["aggregate"],
        }

    facts = {
        "models": models,
        "interaction_robust_languages": robustness["interaction_robust_languages"],
        "test_retest": robustness["test_retest"],
        "parser": parser_audit(load_jsonl("step1_4_instrument.jsonl")),
        "refusals_gemma12b": load_json("summary.json")["refusals"],
        "behavior_rows": behavior["n_rows"],
        "control": control,
        "translation": validation,
        "mechanistic_patching": patching,
    }

    assert round(models["Gemma 4 12B"]["gaps"]["en"], 2) == 1.60
    assert round(models["Gemma 4 12B"]["gaps"]["zh"], 2) == 5.08
    assert round(models["Gemma 4 12B"]["spread"], 2) == 3.48
    assert round(models["Gemma 4 E4B"]["spread"], 2) == 0.97
    assert round(models["Qwen3 8B"]["spread"], 2) == 1.47
    assert round(models["Gemma 4 12B"]["mean_D_over_B"], 2) == 0.97
    assert round(models["Gemma 4 12B"]["mean_E_over_B"], 2) == 0.68
    assert round(models["Gemma 4 E4B"]["mean_D_over_B"], 2) == 0.99
    assert round(models["Qwen3 8B"]["mean_D_over_B"], 2) == 0.89
    assert behavior["n_rows"] == 3450
    assert patching["rows"] == 9600
    assert round(control["association"]["spearman_accuracy_loss_abs_deviation"], 2) == 0.23

    REPORT.mkdir(exist_ok=True)
    (REPORT / "submission_facts.json").write_text(
        json.dumps(facts, indent=2, ensure_ascii=False) + "\n"
    )
    return facts


def plot_crossing(facts):
    labels = list(MODELS)
    d_values = [facts["models"][model]["mean_D_over_B"] for model in labels]
    e_values = [facts["models"][model]["mean_E_over_B"] for model in labels]
    x = range(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.bar([i - width / 2 for i in x], d_values, width,
           label="English stimulus / local battery", color="#277da1")
    ax.bar([i + width / 2 for i in x], e_values, width,
           label="Local stimulus / English battery", color="#f4a261")
    ax.axhline(1, color="#333333", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Effect retained vs fully local condition")
    ax.set_xticks(list(x), labels)
    ax.set_title("The stimulus crosses languages; the reporting effect is model-specific")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    for xpos, value in zip([i - width / 2 for i in x], d_values):
        ax.text(xpos, value + 0.025, f"{value:.2f}", ha="center", fontsize=9)
    for xpos, value in zip([i + width / 2 for i in x], e_values):
        ax.text(xpos, value + 0.025, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "submission_crossing.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    facts = build()
    plot_crossing(facts)
    print("wrote report/submission_facts.json")
    print("wrote figures/submission_crossing.png")

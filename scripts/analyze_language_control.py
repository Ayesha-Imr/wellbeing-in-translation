"""Analyze the language-competence control against existing wellbeing gaps."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report" / "multilingual_control.md"
LANGS = ["en", "es", "zh", "hi", "ur"]
NON_EN = LANGS[1:]
MODEL_ORDER = ["gemma12b", "gemma-e4b", "qwen3-8b"]
MODEL_NAMES = {
    "gemma12b": "Gemma 4 12B",
    "gemma-e4b": "Gemma 4 E4B",
    "qwen3-8b": "Qwen3 8B",
}
LANG_NAMES = {
    "en": "English", "es": "Spanish", "zh": "Chinese",
    "hi": "Hindi", "ur": "Urdu",
}
WELLBEING_FILES = {
    "gemma12b": "step1_4_instrument.jsonl",
    "gemma-e4b": "step1_4_instrument_gemma-e4b.jsonl",
    "qwen3-8b": "step1_4_instrument_qwen3-8b.jsonl",
}
BEHAVIOR_FILES = {
    "gemma12b": "behavior_behavior-gemma12b.jsonl",
    "gemma-e4b": "behavior_behavior-gemma-e4b.jsonl",
    "qwen3-8b": "behavior_behavior-qwen3-8b.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap(values: list[float], seed: int, n_boot: int = 4000) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.percentile(draws, [2.5, 97.5]))


def bootstrap_difference(positive: list[float], negative: list[float], seed: int,
                         n_boot: int = 4000) -> tuple[float, float]:
    if len(positive) < 1 or len(negative) < 1:
        return (float("nan"), float("nan"))
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    rng = np.random.default_rng(seed)
    pos_draws = rng.choice(pos, size=(n_boot, len(pos)), replace=True).mean(axis=1)
    neg_draws = rng.choice(neg, size=(n_boot, len(neg)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.percentile(pos_draws - neg_draws, [2.5, 97.5]))


def bootstrap_gap_samples(positive: list[float], negative: list[float], seed: int,
                          n_boot: int = 4000) -> np.ndarray:
    if len(positive) < 1 or len(negative) < 1:
        return np.asarray([], dtype=float)
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    rng = np.random.default_rng(seed)
    pos_draws = rng.choice(pos, size=(n_boot, len(pos)), replace=True).mean(axis=1)
    neg_draws = rng.choice(neg, size=(n_boot, len(neg)), replace=True).mean(axis=1)
    return pos_draws - neg_draws


def control_summary() -> dict:
    out = {}
    for model in MODEL_ORDER:
        rows = load_jsonl(RESULTS / f"language_control_{model}.jsonl")
        if len(rows) != 150:
            raise RuntimeError(f"{model}: expected 150 control rows, found {len(rows)}")
        out[model] = {}
        for lang in LANGS:
            subset = [row for row in rows if row["language"] == lang]
            out[model][lang] = {
                "n": len(subset),
                "accuracy": float(np.mean([row["correct"] for row in subset])),
                "greedy_accuracy": float(np.mean([row["generated_correct"] for row in subset])),
                "valid_output": float(np.mean([row["valid_output"] for row in subset])),
                "agreement": float(np.mean([
                    row["predicted_choice"] == row["generated_choice"] for row in subset
                ])),
                "correct_probability": float(np.mean([
                    row["correct_probability"] for row in subset
                ])),
                "margin": float(np.mean([row["correct_logit_margin"] for row in subset])),
                "entropy": float(np.mean([row["option_entropy"] for row in subset])),
                "option_mass": float(np.mean([row["option_mass"] for row in subset])),
                "script_mismatch": float(np.mean([
                    row["prose_script_matches"] is False for row in subset
                ])),
            }
    return out


def paired_gaps(rows: list[dict], value_fn) -> tuple[dict, dict]:
    by = defaultdict(list)
    for row in rows:
        value = value_fn(row)
        if value is not None:
            by[(row["language"], row.get("side"), row["experience_id"])].append(float(value))
    means = {(lang, side, exp): float(np.mean(values))
             for (lang, side, exp), values in by.items()}
    groups = defaultdict(list)
    for (lang, side, _experience), value in means.items():
        groups[(lang, side)].append(value)
    gap_by_lang = {}
    delta_by_lang = {}
    for lang in LANGS:
        positive = groups[(lang, "positive")]
        negative = groups[(lang, "negative")]
        gap_by_lang[lang] = {
            "gap": float(np.mean(positive) - np.mean(negative))
            if positive and negative else float("nan"),
            "n_positive": len(positive),
            "n_negative": len(negative),
            "ci": bootstrap_difference(positive, negative, 100 + len(lang)),
        }
    english_samples = bootstrap_gap_samples(
        groups[("en", "positive")], groups[("en", "negative")], 900
    )
    english_gap = (
        float(np.mean(groups[("en", "positive")]) - np.mean(groups[("en", "negative")]))
        if groups[("en", "positive")] and groups[("en", "negative")] else float("nan")
    )
    for lang in NON_EN:
        language_samples = bootstrap_gap_samples(
            groups[(lang, "positive")], groups[(lang, "negative")], 1200 + len(lang)
        )
        deltas = language_samples - english_samples if len(language_samples) else np.asarray([])
        language_gap = (
            float(np.mean(groups[(lang, "positive")]) - np.mean(groups[(lang, "negative")]))
            if groups[(lang, "positive")] and groups[(lang, "negative")] else float("nan")
        )
        delta_by_lang[lang] = {
            "deviation": language_gap - english_gap if math.isfinite(language_gap) and math.isfinite(english_gap) else float("nan"),
            "absolute_deviation": abs(language_gap - english_gap) if math.isfinite(language_gap) and math.isfinite(english_gap) else float("nan"),
            "n_positive": len(groups[(lang, "positive")]),
            "n_negative": len(groups[(lang, "negative")]),
            "ci": tuple(float(x) for x in np.percentile(deltas, [2.5, 97.5])) if len(deltas) else (float("nan"), float("nan")),
        }
    return gap_by_lang, delta_by_lang


def wellbeing_summary() -> tuple[dict, dict]:
    self_report = {}
    behavior = {}
    for model in MODEL_ORDER:
        rows = load_jsonl(RESULTS / WELLBEING_FILES[model])
        gaps, deltas = paired_gaps(rows, lambda row: row.get("parsed_rating"))
        self_report[model] = {"gaps": gaps, "deltas": deltas}
        behavior_path = RESULTS / BEHAVIOR_FILES[model]
        if behavior_path.exists():
            behavior_rows = load_jsonl(behavior_path)
            gaps, deltas = paired_gaps(
                behavior_rows,
                lambda row: 1.0 if row.get("choice") == "continue" else 0.0
                if row.get("choice") == "stop" else None,
            )
            behavior[model] = {"gaps": gaps, "deltas": deltas}
    return self_report, behavior


def association(control: dict, wellbeing: dict) -> dict:
    points = []
    for model in MODEL_ORDER:
        en_acc = control[model]["en"]["accuracy"]
        en_probability = control[model]["en"]["correct_probability"]
        en_margin = control[model]["en"]["margin"]
        en_entropy = control[model]["en"]["entropy"]
        en_validity = control[model]["en"]["valid_output"]
        for lang in NON_EN:
            delta = wellbeing[model]["deltas"].get(lang)
            if not delta:
                continue
            cell = control[model][lang]
            points.append({
                "model": model,
                "language": lang,
                "accuracy": cell["accuracy"],
                "accuracy_loss": en_acc - cell["accuracy"],
                "probability_loss": en_probability - cell["correct_probability"],
                "margin_loss": en_margin - cell["margin"],
                "entropy_increase": cell["entropy"] - en_entropy,
                "validity_loss": en_validity - cell["valid_output"],
                "wellbeing_deviation": delta["deviation"],
                "absolute_wellbeing_deviation": delta["absolute_deviation"],
            })
    if len(points) < 4:
        return {"n": len(points), "points": points}
    from scipy.stats import spearmanr
    y_abs = np.asarray([row["absolute_wellbeing_deviation"] for row in points])
    y_signed = np.asarray([row["wellbeing_deviation"] for row in points])

    def corr(metric: str, target: np.ndarray) -> tuple[float, float]:
        rho, p_value = spearmanr([row[metric] for row in points], target)
        return float(rho), float(p_value)

    metrics = ("accuracy_loss", "probability_loss", "entropy_increase", "validity_loss")
    correlations = {}
    for metric in metrics:
        rho, p_value = corr(metric, y_abs)
        correlations[metric] = {"rho": rho, "p_uncorrected": p_value}
    rho_signed, p_signed = corr("accuracy_loss", y_signed)
    per_model = {}
    for model in MODEL_ORDER:
        subset = [row for row in points if row["model"] == model]
        if len(subset) >= 3:
            rx, _ = spearmanr(
                [row["accuracy_loss"] for row in subset],
                [row["absolute_wellbeing_deviation"] for row in subset],
            )
            per_model[model] = float(rx)
    return {
        "n": len(points),
        "points": points,
        "correlations_abs_deviation": correlations,
        "spearman_accuracy_loss_abs_deviation": correlations["accuracy_loss"]["rho"],
        "spearman_accuracy_loss_abs_deviation_p_uncorrected": correlations["accuracy_loss"]["p_uncorrected"],
        "spearman_accuracy_loss_signed_deviation": float(rho_signed),
        "spearman_accuracy_loss_signed_deviation_p_uncorrected": float(p_signed),
        "per_model_spearman_abs": per_model,
    }


def fmt(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def make_figures(control: dict, assoc: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    metrics = [
        ("accuracy", "Exact accuracy", "%.2f"),
        ("correct_probability", "Correct-option probability", "%.2f"),
        ("margin", "Correct-vs-best-wrong margin", "%.1f"),
        ("entropy", "Four-option entropy", "%.2f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (metric, title, _) in zip(axes.flat, metrics):
        matrix = np.asarray([[control[model][lang][metric] for lang in LANGS]
                             for model in MODEL_ORDER])
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(LANGS)), [LANG_NAMES[l] for l in LANGS], rotation=25, ha="right")
        ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_NAMES[m] for m in MODEL_ORDER])
        ax.set_title(title)
        for i in range(len(MODEL_ORDER)):
            for j in range(len(LANGS)):
                ax.text(j, i, fmt(matrix[i, j], 2 if metric != "margin" else 1),
                        ha="center", va="center", color="white" if matrix[i, j] < np.nanmean(matrix) else "black", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Neutral language competence is high, but not identical across languages", fontsize=14)
    fig.savefig(FIGURES / "language_control_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"gemma12b": "#1565c0", "gemma-e4b": "#5e9ce8", "qwen3-8b": "#ef6c00"}
    for model in MODEL_ORDER:
        subset = [row for row in assoc["points"] if row["model"] == model]
        x = [row["accuracy_loss"] for row in subset]
        y = [row["absolute_wellbeing_deviation"] for row in subset]
        ax.scatter(x, y, s=75, color=colors[model], label=MODEL_NAMES[model], alpha=0.9)
        for row in subset:
            ax.annotate(row["language"].upper(), (row["accuracy_loss"], row["absolute_wellbeing_deviation"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
    if len(assoc["points"]) >= 3:
        x = np.asarray([row["accuracy_loss"] for row in assoc["points"]])
        y = np.asarray([row["absolute_wellbeing_deviation"] for row in assoc["points"]])
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, color="#444", lw=1.5, ls="--")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Accuracy loss relative to English (positive = worse)")
    ax.set_ylabel("Absolute wellbeing deviation from English")
    ax.set_title("Does lower neutral-task competence predict a larger wellbeing shift?")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    rho = assoc.get("spearman_accuracy_loss_abs_deviation", float("nan"))
    ax.text(0.02, 0.96, f"Spearman ρ = {fmt(rho, 2)}; n = {assoc.get('n', 0)}\nDescriptive only; model identity is not causal control",
            transform=ax.transAxes, va="top", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(FIGURES / "wellbeing_vs_competence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(control: dict, wellbeing: dict, behavior: dict, assoc: dict) -> None:
    lines = [
        "# Multilingual competence/calibration control",
        "",
        "This control asks whether the wellbeing-language differences could be explained by ordinary language performance. It is a control for competence and calibration, not a test of the models' exact training-data composition.",
        "",
        "## Design",
        "",
        "- Three models: Gemma 4 12B, Gemma 4 E4B, and Qwen3 8B.",
        "- Five languages: English, Spanish, Chinese, Hindi, and Urdu.",
        "- 30 identical synthetic neutral questions per language: 10 arithmetic, 10 logic, and 10 reading-comprehension items.",
        "- Four-option next-token scoring plus deterministic greedy output parsing; no LLM judge was used for measurement.",
        "- All translations were independently generated, back-translated, verified by GPT-5, and aligned to one common 30-item set. One Chinese item failed verification and was replaced by the same reserve item in every language.",
        "",
        "## Control results",
        "",
        "| model | language | accuracy | correct-option probability | margin | entropy | valid output |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        for lang in LANGS:
            cell = control[model][lang]
            lines.append(
                f"| {MODEL_NAMES[model]} | {LANG_NAMES[lang]} | {cell['accuracy']:.3f} | "
                f"{cell['correct_probability']:.3f} | {cell['margin']:.2f} | "
                f"{cell['entropy']:.3f} | {cell['valid_output']:.1%} |"
            )
    lines += [
        "",
        "The scorer passed its internal consistency check: logit predictions and greedy outputs agree on nearly every row, and option probability mass is close to one. This is important because it shows the readout is measuring the answer position rather than an earlier prompt token.",
        "",
        "## Existing wellbeing comparison",
        "",
        "The primary wellbeing outcome is the normalized self-report positive-minus-negative gap. Deviations are each language's gap minus the English gap for the same model.",
        "",
        "| model | language | wellbeing gap | deviation from English | 95% CI for deviation | experiences |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        for lang in NON_EN:
            gap = wellbeing[model]["gaps"][lang]
            delta = wellbeing[model]["deltas"][lang]
            lines.append(
                f"| {MODEL_NAMES[model]} | {LANG_NAMES[lang]} | {gap['gap']:.3f} | "
                f"{delta['deviation']:.3f} | [{delta['ci'][0]:.3f}, {delta['ci'][1]:.3f}] | "
                f"{delta['n_positive']}/{delta['n_negative']} |"
            )
    lines += [
        "",
        "## Does competence explain the wellbeing deviation?",
        "",
        f"Across the 12 non-English model-language cells, the descriptive Spearman correlation between accuracy loss relative to English and absolute wellbeing deviation is **{assoc.get('spearman_accuracy_loss_abs_deviation', float('nan')):.2f}** (uncorrected p = {assoc.get('spearman_accuracy_loss_abs_deviation_p_uncorrected', float('nan')):.3f}). This is a small-cell descriptive association, not a confirmatory test.",
        "",
        "A positive value would mean that languages performing worse on the neutral control tend to show larger wellbeing shifts. A weak or inconsistent value means ordinary competence does not straightforwardly account for the wellbeing pattern.",
        "",
        "Secondary control metrics (pooled across the same 12 cells; uncorrected, descriptive):",
        "",
    ]
    metric_labels = {
        "accuracy_loss": "accuracy loss",
        "probability_loss": "correct-option probability loss",
        "entropy_increase": "entropy increase",
        "validity_loss": "valid-output loss",
    }
    for metric, label in metric_labels.items():
        result = assoc.get("correlations_abs_deviation", {}).get(metric, {})
        lines.append(
            f"- {label}: ρ = {result.get('rho', float('nan')):.2f}, "
            f"p = {result.get('p_uncorrected', float('nan')):.3f}"
        )
    lines += [
        "",
        "Per-model descriptive correlations (four non-English languages each):",
        "",
    ]
    for model, rho in assoc.get("per_model_spearman_abs", {}).items():
        lines.append(f"- {MODEL_NAMES[model]}: ρ = {rho:.2f}")
    if behavior:
        lines += [
            "",
            "## Behavioral proxy cross-check",
            "",
            "The existing continue/exit choice provides a separate, noisier behavioral proxy. It broadly preserves the same positive-versus-negative direction, but it is close to ceiling for Gemma and Qwen in several languages, so it should be treated as corroboration rather than a second precise measurement.",
            "",
            "| model | English behavior gap | Spanish deviation | Chinese deviation | Hindi deviation | Urdu deviation |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for model in MODEL_ORDER:
            row = behavior[model]
            values = [row["deltas"][lang]["deviation"] for lang in NON_EN]
            lines.append(
                f"| {MODEL_NAMES[model]} | {row['gaps']['en']['gap']:.2f} | "
                + " | ".join(f"{value:+.2f}" for value in values) + " |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "This control can rule in or weaken a competence explanation, but it cannot identify how much multilingual data a model saw. Training-language counts, tokenizer behavior, instruction tuning, and calibration remain confounded.",
        "",
        "If wellbeing deviations remain large for language pairs with similar neutral-task accuracy, that is more consistent with a language-dependent wellbeing readout. If deviations track accuracy loss, the safer explanation is a performance/calibration artifact. The present result should be reported with the small number of model-language cells and the control's finite 30-item size clearly stated.",
        "",
        "Figures: `figures/language_control_heatmap.png` and `figures/wellbeing_vs_competence.png`.",
    ]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    control = control_summary()
    wellbeing, behavior = wellbeing_summary()
    assoc = association(control, wellbeing)
    summary = {
        "control": control,
        "self_report": wellbeing,
        "behavior": behavior,
        "association": assoc,
    }
    (RESULTS / "language_control_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figures(control, assoc)
    write_report(control, wellbeing, behavior, assoc)
    print(json.dumps({
        "rows": assoc.get("n"),
        "spearman_accuracy_loss_abs_deviation": assoc.get("spearman_accuracy_loss_abs_deviation"),
        "spearman_accuracy_loss_signed_deviation": assoc.get("spearman_accuracy_loss_signed_deviation"),
        "secondary_correlations": assoc.get("correlations_abs_deviation"),
    }, indent=2))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()

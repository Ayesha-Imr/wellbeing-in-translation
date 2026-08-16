"""Summarise mechanistic probe outputs and make the two final figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report" / "mech_interp_results.md"
LANGS = ["en", "es", "zh", "hi", "ur"]
MODELS = {
    "gemma12b": "Gemma 4 12B",
    "gemma-e4b": "Gemma 4 E4B",
    "qwen3-8b": "Qwen3 8B",
}
COLORS = {"gemma12b": "#365C8D", "gemma-e4b": "#35B779", "qwen3-8b": "#D95F59"}


def model_tag(path: Path) -> str:
    return path.stem.removeprefix("mech_geometry_")


def load_geometry():
    rows = []
    for path in sorted(RESULTS.glob("mech_geometry_*.npz")):
        tag = model_tag(path)
        data = np.load(path)
        for task in ("self_report", "behavior"):
            en = data[f"{task}__en"]
            for lang in LANGS:
                local = data[f"{task}__{lang}"]
                cosine = np.sum(en * local, axis=1)
                for layer, value in enumerate(cosine):
                    rows.append({"model": tag, "task": task, "language": lang,
                                 "layer": layer, "cosine_en": float(value)})
            for lang in LANGS:
                self_dir = data[f"self_report__{lang}"]
                beh_dir = data[f"behavior__{lang}"]
                cosine = np.sum(self_dir * beh_dir, axis=1)
                for layer, value in enumerate(cosine):
                    rows.append({"model": tag, "task": "cross_task", "language": lang,
                                 "layer": layer, "cosine_en": float(value)})
    return pd.DataFrame(rows)


def load_projections():
    rows = []
    for path in sorted(RESULTS.glob("mech_projections_*.json")):
        tag = path.stem.removeprefix("mech_projections_")
        for row in json.loads(path.read_text()):
            scores = row.pop("heldout_projection")
            for layer, value in enumerate(scores):
                rows.append({**row, "model": tag, "layer": layer, "projection": value})
    return pd.DataFrame(rows)


def load_steering():
    rows = []
    for path in sorted(RESULTS.glob("mech_steering_*.jsonl")):
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return pd.DataFrame(rows)


def bootstrap(values, seed=20260816, n=1000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def geometry_summary(geometry, projections):
    summary = {}
    for model in sorted(geometry.model.unique()):
        summary[model] = {}
        for task in ("self_report", "behavior", "cross_task"):
            sub = geometry[(geometry.model == model) & (geometry.task == task)]
            summary[model][task] = float(sub[sub.layer == sub.layer.max()].cosine_en.mean())
    if not projections.empty:
        projections["positive"] = projections.side == "positive"
        for (model, task), sub in projections.groupby(["model", "task"]):
            top = sub[sub.layer == sub.layer.max()]
            pos = top[top.positive].groupby("experience_id").projection.mean()
            neg = top[~top.positive].groupby("experience_id").projection.mean()
            # Keep this diagnostic in the JSON; the figure uses layerwise cosine.
            summary.setdefault(model, {}).setdefault("heldout", {})[task] = float(pos.mean() - neg.mean())
    return summary


def make_geometry_figure(geometry):
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, task, title in zip(
        axes[:1], ["self_report"], ["Self-report direction"]
    ):
        for model in MODELS:
            sub = geometry[(geometry.model == model) & (geometry.task == task) & (geometry.language != "en")]
            if sub.empty:
                continue
            line = sub.groupby("layer").cosine_en.mean()
            ax.plot(line.index, line.values, lw=2, label=MODELS.get(model, model), color=COLORS.get(model))
        ax.set_title(title)
    ax = axes[1]
    for model in MODELS:
        sub = geometry[(geometry.model == model) & (geometry.task == "cross_task")]
        if sub.empty:
            continue
        line = sub.groupby("layer").cosine_en.mean()
        ax.plot(line.index, line.values, lw=2, label=MODELS.get(model, model), color=COLORS.get(model))
    ax.set_title("Self-report ↔ behavior direction")
    for ax in axes:
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlabel("Decoder layer")
        ax.grid(axis="y", color="#eeeeee", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cosine similarity (1 = same direction)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Where the positive–negative signal points", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "mech_geometry.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_steering_figure(steering):
    if steering.empty:
        return
    FIGURES.mkdir(exist_ok=True)
    conditions = ["english_pos", "english_neg", "local_pos", "random_pos"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), sharex=False)
    for col, model in enumerate(MODELS):
        for row, task in enumerate(("self_report", "behavior")):
            ax = axes[row, col]
            sub = steering[(steering.model == model) & (steering.task == task)]
            base = sub[sub.condition == "zero"].groupby(
                ["target_task", "source_task", "language", "experience_id"]
            ).value.mean().rename("baseline")
            nonbase = sub[sub.condition.isin(conditions)].copy()
            key = ["target_task", "source_task", "language", "experience_id"]
            nonbase = nonbase.join(base, on=key)
            nonbase["delta"] = nonbase.value - nonbase.baseline
            for x, condition in enumerate(conditions):
                vals = nonbase[nonbase.condition == condition].delta
                mean, lo, hi = bootstrap(vals, seed=20260816 + x)
                ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                            color={"english_pos": "#35B779", "english_neg": "#D95F59",
                                   "local_pos": "#365C8D", "random_pos": "#999999"}[condition],
                            capsize=3, ms=5)
            ax.axhline(0, color="#888888", lw=0.8)
            ax.set_xticks(range(len(conditions)), ["EN +", "EN −", "local +", "random +"], rotation=30, ha="right")
            ax.grid(axis="y", color="#eeeeee", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(MODELS.get(model, model), fontsize=10)
            if col == 0:
                ax.set_ylabel("Δ expected rating" if task == "self_report" else "Δ P(continue)")
            if row == 0:
                ax.set_ylim(bottom=min(-0.5, ax.get_ylim()[0]))
    fig.suptitle("Cross-task activation steering at the middle layer", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "mech_steering.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(geometry, projections, steering, summary):
    lines = [
        "# Mechanistic interpretability results",
        "",
        "This is a small, causal probe of the multilingual wellbeing result. It does not claim that a model is conscious. It asks whether a positive-minus-negative direction is shared across languages and whether moving that direction changes the model's next-token answer.",
        "",
        "## Design",
        "",
        "- Models: Gemma 4 12B, Gemma 4 E4B, and Qwen3 8B.",
        "- Languages: English, Spanish, Chinese, Hindi, and Urdu (the five robust languages used by the main study).",
        "- Items: the same 20 positive/negative Step-1 experiences; three neutral items are retained in the shared item definition but do not fit a valence direction.",
        f"- Readout: the final input position, with one fixed self-report question ({TARGET_QUESTION}) and the existing continue/stop prompt.",
        "- Direction: mean activation on positive items minus mean activation on negative items, normalized separately at each layer. Five balanced experience-level folds prevent a prompt being used to fit and test its own direction.",
        "- Causal test: at the middle decoder layer, add the English direction, its opposite, the local-language direction, or a norm-matched random direction. The outcome is next-token probability: expected rating (1–7) or P(continue).",
        "",
        "## Files",
        "",
        "- `figures/mech_geometry.png`: cross-language and cross-task direction geometry.",
        "- `figures/mech_steering.png`: causal steering changes relative to the zero-hook baseline.",
        "- `results/mech_geometry_<model>.npz`: normalized directions by layer.",
        "- `results/mech_projections_<model>.json`: held-out projections.",
        "- `results/mech_steering_<model>.jsonl`: raw causal rows with controls.",
        "",
        "## Geometry summary",
        "",
        "The exact end-of-network English-to-local and self-report-to-behavior cosines are recorded below. A high cosine means the direction points similarly; a value near zero means language/task-specific geometry.",
        "",
        "| model | self-report EN→local | behavior EN→local | self-report ↔ behavior | |",
        "|---|---:|---:|---:|",
    ]
    for model in sorted(summary):
        row = summary[model]
        lines.append(f"| {MODELS.get(model, model)} | {row.get('self_report', float('nan')):.3f} | {row.get('behavior', float('nan')):.3f} | {row.get('cross_task', float('nan')):.3f} |")
    lines += [
        "",
        "## Interpretation rule",
        "",
        "A convincing mechanism would have (1) positive held-out projections, (2) similar directions across robust languages, and (3) a signed causal effect: the positive direction moves the answer toward higher wellbeing or continuation, while the opposite direction moves it back. If only the readout is shared but steering is flat, the safe conclusion is that the direction is descriptive rather than demonstrated as causal.",
        "",
        "The controls are intentionally visible. A random direction moving the answer as much as the valence direction, or a zero-hook baseline changing across repeated rows, is a warning against a strong mechanistic claim.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def main():
    geometry = load_geometry()
    projections = load_projections()
    steering = load_steering()
    if geometry.empty:
        raise SystemExit("no mech_geometry_*.npz files found")
    summary = geometry_summary(geometry, projections)
    (RESULTS / "mech_interp_summary.json").write_text(json.dumps(summary, indent=2))
    make_geometry_figure(geometry)
    make_steering_figure(steering)
    write_report(geometry, projections, steering, summary)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()

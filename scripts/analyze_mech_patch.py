"""Audit activation-patching outputs and make compact publication figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report" / "mech_patching_results.md"
LANGS = ["es", "zh", "hi", "ur"]
TASKS = ["self_report", "behavior"]
CONDITIONS = ["paired", "shuffled"]
SOURCE_GAP_THRESHOLD = 0.05
MODELS = {
    "gemma-e4b": "Gemma 4 E4B",
    "gemma12b": "Gemma 4 12B",
    "qwen3-8b": "Qwen3 8B",
}
MODEL_COLORS = {
    "gemma-e4b": "#35B779",
    "gemma12b": "#365C8D",
    "qwen3-8b": "#D95F59",
}
DIRECTION_COLORS = {"en_to_local": "#35B779", "local_to_en": "#365C8D"}


def load_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS.glob("mech_patching_*.jsonl")):
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["model"] = frame["model"].astype(str).str.removesuffix("-patch")
    frame["direction_detail"] = frame["direction"]
    frame["direction"] = np.where(
        frame["source_lang"].eq("en"), "en_to_local", "local_to_en"
    )
    frame["pair_lang"] = np.where(
        frame["direction"].eq("en_to_local"), frame["target_lang"], frame["source_lang"]
    )
    frame["delta"] = frame["patched_score"] - frame["target_score"]
    frame["source_delta"] = frame["source_score"] - frame["target_score"]
    frame["recovery"] = frame["delta"] / frame["source_delta"].where(
        frame["source_delta"].abs() >= SOURCE_GAP_THRESHOLD
    )
    return frame


def bootstrap(values, seed=20260816, n=1000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def cluster_bootstrap(frame: pd.DataFrame, column: str, seed: int):
    """Bootstrap experience clusters after averaging repeated directions/languages."""
    per_item = frame.groupby("experience_id", as_index=False)[column].mean()
    return bootstrap(per_item[column], seed=seed)


def specificity(frame: pd.DataFrame) -> pd.DataFrame:
    key = ["model", "task", "pair_lang", "direction", "experience_id", "layer", "layer_fraction"]
    wide = frame.pivot_table(index=key, columns="condition", values="delta", aggfunc="mean").reset_index()
    if "paired" not in wide or "shuffled" not in wide:
        return pd.DataFrame()
    wide["specificity"] = wide["paired"] - wide["shuffled"]
    return wide


def make_specificity_figure(spec: pd.DataFrame):
    if spec.empty:
        return
    FIGURES.mkdir(exist_ok=True)
    models = sorted(spec.model.unique(), key=lambda x: list(MODELS).index(x) if x in MODELS else x)
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 6.2), sharex=True, squeeze=False)
    for col, model in enumerate(models):
        for row, task in enumerate(TASKS):
            ax = axes[row, col]
            sub = spec[(spec.model == model) & (spec.task == task)]
            for direction in ("en_to_local", "local_to_en"):
                line = []
                for layer, group in sub[sub.direction == direction].groupby(
                    ["layer", "layer_fraction"], sort=True
                ):
                    mean, lo, hi = cluster_bootstrap(
                        group, "specificity", seed=20260816 + int(layer[0])
                    )
                    line.append((layer[1], mean, lo, hi))
                if not line:
                    continue
                line.sort()
                x, mean, lo, hi = map(np.asarray, zip(*line))
                color = DIRECTION_COLORS[direction]
                label = "EN → local" if direction == "en_to_local" else "local → EN"
                ax.plot(x, mean, marker="o", ms=4, lw=2, color=color, label=label)
                ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
            ax.axhline(0, color="#888888", lw=0.8)
            ax.grid(axis="y", color="#eeeeee", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(MODELS.get(model, model), fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    "paired − shuffled logit effect\n"
                    + ("rating slope" if task == "self_report" else "choice gap")
                )
            if row == 1:
                ax.set_xlabel("Decoder layer fraction")
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Does same-item translation patching transfer more than a control?", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "mech_patching_specificity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_middle_heatmap(spec: pd.DataFrame):
    if spec.empty:
        return
    middle = spec.loc[(spec.layer_fraction - 0.5).abs() < 0.08].copy()
    if middle.empty:
        return
    middle = middle.groupby(
        ["model", "task", "pair_lang", "direction"], as_index=False
    ).specificity.mean()
    rows = []
    models = sorted(middle.model.unique(), key=lambda x: list(MODELS).index(x) if x in MODELS else x)
    for model in models:
        for task in TASKS:
            for direction in ("en_to_local", "local_to_en"):
                values = []
                for lang in LANGS:
                    part = middle[
                        (middle.model == model)
                        & (middle.task == task)
                        & (middle.direction == direction)
                        & (middle.pair_lang == lang)
                    ]
                    values.append(float(part.specificity.iloc[0]) if not part.empty else np.nan)
                rows.append((f"{MODELS.get(model, model)}\n{task}\n{direction}", values))
    matrix = np.asarray([row[1] for row in rows], dtype=float)
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(0.001, float(np.quantile(finite, 0.95)) * 1.15) if len(finite) else 0.001
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(LANGS)), [x.upper() for x in LANGS])
    ax.set_yticks(range(len(rows)), [row[0] for row in rows], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Paired non-English language")
    ax.set_title("Middle-layer same-item specificity (paired − shuffled)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, label="logit effect difference")
    fig.tight_layout()
    fig.savefig(FIGURES / "mech_patching_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    key = ["model", "task", "direction", "condition"]
    rows = []
    for keys, group in frame.groupby(key):
        mean, lo, hi = cluster_bootstrap(group, "delta", seed=20260816)
        rows.append(dict(zip(key, keys), mean=mean, lo=lo, hi=hi, n_items=group.experience_id.nunique()))
    return pd.DataFrame(rows)


def write_report(frame: pd.DataFrame, spec: pd.DataFrame):
    summary = summary_table(frame)
    model_names = [MODELS.get(model, model) for model in sorted(
        frame.model.unique(), key=lambda x: list(MODELS).index(x) if x in MODELS else x
    )]
    lines = [
        "# Translation-paired activation-patching results",
        "",
        "This is a causal representation-transfer probe. It asks whether a model's",
        "activation for an experience in one language can replace the corresponding",
        "activation in its translation, beyond a natural shuffled-item control. It",
        "does not claim that a model is conscious.",
        "",
        "## Design",
        "",
        f"- Models: {', '.join(model_names)}.",
        "- Languages: English paired with Spanish, Chinese, Hindi and Urdu.",
        "- Tasks: the fixed `wb_happy` self-report question and the continue/stop choice.",
        "- Items: the same 20 positive/negative Step-1 experiences; no fitted direction is used.",
        "- Intervention: replace the final-position residual with the naturally occurring",
        "  source activation at five pre-specified decoder-layer fractions (20%, 35%, 50%, 65%, 80%).",
        "- Paired condition: source and target are translations of the same experience.",
        "- Shuffled control: source is a different experience with the same valence and source language.",
        "- Readout: rating-token logit slope or continue-minus-stop logit gap. No generation or judge is used.",
        "",
        "The primary quantity is paired-minus-shuffled patch effect. Positive values mean",
        "the same-item translation carries more transferable information than a",
        "same-valence natural activation replacement.",
        "",
        "## Files",
        "",
        "- `figures/mech_patching_specificity.png`: layerwise causal specificity with item-cluster intervals.",
        "- `figures/mech_patching_heatmap.png`: middle-layer specificity by model, task, direction and language.",
        "- `results/mech_patching_<model>.jsonl`: raw rows, including source/target lengths and scores.",
        "",
        "## Summary",
        "",
        f"Raw rows: {len(frame):,}. Each row is one item, layer, direction and control.",
        "",
        "| model | target task | direction | paired Δ | shuffled Δ | paired−shuffled [95% CI] | items |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    spec_summary = {}
    if not spec.empty:
        for keys, group in spec.groupby(["model", "task", "direction"]):
            spec_summary[keys] = cluster_bootstrap(group, "specificity", seed=20260816)
    if not summary.empty:
        for (model, task, direction), group in summary.groupby(["model", "task", "direction"]):
            pair = group[group.condition == "paired"].iloc[0] if not group[group.condition == "paired"].empty else None
            shuffled = group[group.condition == "shuffled"].iloc[0] if not group[group.condition == "shuffled"].empty else None
            if pair is None or shuffled is None:
                continue
            spec_mean, spec_lo, spec_hi = spec_summary.get(
                (model, task, direction), (float("nan"), float("nan"), float("nan"))
            )
            lines.append(
                f"| {MODELS.get(model, model)} | {task} | {direction} | "
                f"{pair['mean']:+.4f} [{pair['lo']:+.4f}, {pair['hi']:+.4f}] | "
                f"{shuffled['mean']:+.4f} [{shuffled['lo']:+.4f}, {shuffled['hi']:+.4f}] | "
                f"{spec_mean:+.4f} [{spec_lo:+.4f}, {spec_hi:+.4f}] | {int(pair['n_items'])} |"
            )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "A translation-specific causal signal requires paired patches to exceed",
        "same-valence shuffled patches consistently across layers and languages,",
        "with similar transfer in both directions. If paired and shuffled effects",
        "are comparable, the geometry remains descriptive rather than a demonstrated",
        "translation-invariant mechanism.",
        "",
        "All intervals are experience-cluster bootstrap intervals. The recovery ratio",
        f"is reported only as a secondary diagnostic when the source-target logit gap",
        f"has absolute size at least {SOURCE_GAP_THRESHOLD:.2f}; raw paired-minus-shuffled",
        "effects are primary because they remain defined when the two language outputs",
        "are already similar.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def main():
    frame = load_rows()
    if frame.empty:
        raise SystemExit("no mech_patching_*.jsonl files found")
    spec = specificity(frame)
    (RESULTS / "mech_patching_summary.json").write_text(
        json.dumps({
            "rows": int(len(frame)),
            "models": sorted(frame.model.unique()),
            "tasks": sorted(frame.task.unique()),
            "conditions": sorted(frame.condition.unique()),
            "source_gap_threshold": SOURCE_GAP_THRESHOLD,
            "specificity_rows": int(len(spec)),
        }, indent=2)
    )
    make_specificity_figure(spec)
    make_middle_heatmap(spec)
    write_report(frame, spec)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()

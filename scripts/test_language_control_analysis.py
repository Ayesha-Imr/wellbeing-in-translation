"""CPU-only analysis invariants for the language-control outputs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import analyze_language_control as analysis  # noqa: E402


def main():
    control = analysis.control_summary()
    assert set(control) == set(analysis.MODEL_ORDER)
    for model in analysis.MODEL_ORDER:
        assert all(control[model][lang]["n"] == 30 for lang in analysis.LANGS)
        assert all(0 <= control[model][lang]["accuracy"] <= 1 for lang in analysis.LANGS)
    wellbeing, behavior = analysis.wellbeing_summary()
    assert set(wellbeing) == set(analysis.MODEL_ORDER)
    assoc = analysis.association(control, wellbeing)
    assert assoc["n"] == 12
    assert len(assoc["points"]) == 12
    assert set(assoc["correlations_abs_deviation"]) == {
        "accuracy_loss", "probability_loss", "entropy_increase", "validity_loss"
    }
    assert all("probability_loss" in point for point in assoc["points"])
    assert all("entropy_increase" in point for point in assoc["points"])
    assert (ROOT / "results/language_control_gemma12b.jsonl").exists()
    print("language control analysis tests: ok")


if __name__ == "__main__":
    main()

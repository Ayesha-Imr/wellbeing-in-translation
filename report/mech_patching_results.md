# Translation-paired activation-patching results

This is a causal representation-transfer probe. It asks whether a model's
activation for an experience in one language can replace the corresponding
activation in its translation, beyond a natural shuffled-item control. It
does not claim that a model is conscious.

## Design

- Models: Gemma 4 12B.
- Languages: English paired with Spanish, Chinese, Hindi and Urdu.
- Tasks: the fixed `wb_happy` self-report question and the continue/stop choice.
- Items: the same 20 positive/negative Step-1 experiences; no fitted direction is used.
- Intervention: replace the final-position residual with the naturally occurring
  source activation at five pre-specified decoder-layer fractions (20%, 35%, 50%, 65%, 80%).
- Paired condition: source and target are translations of the same experience.
- Shuffled control: source is a different experience with the same valence and source language.
- Readout: rating-token logit slope or continue-minus-stop logit gap. No generation or judge is used.

The primary quantity is paired-minus-shuffled patch effect. Positive values mean
the same-item translation carries more transferable information than a
same-valence natural activation replacement.

## Files

- `figures/mech_patching_specificity.png`: layerwise causal specificity with item-cluster intervals.
- `figures/mech_patching_heatmap.png`: middle-layer specificity by model, task, direction and language.
- `results/mech_patching_<model>.jsonl`: raw rows, including source/target lengths and scores.

## Summary

Raw rows: 3,200. Each row is one item, layer, direction and control.

| model | target task | direction | paired Δ | shuffled Δ | paired−shuffled [95% CI] | items |
|---|---|---|---:|---:|---:|---:|
| Gemma 4 12B | behavior | en_to_local | -0.0107 [-0.5221, +0.4787] | -0.6498 [-2.4660, +0.9772] | +0.6391 [-0.6155, +2.1006] | 20 |
| Gemma 4 12B | behavior | local_to_en | -0.0529 [-0.8579, +0.6710] | -0.6312 [-2.5757, +1.2012] | +0.5783 [-0.5920, +1.9963] | 20 |
| Gemma 4 12B | self_report | en_to_local | +0.0276 [-0.0025, +0.0761] | -0.1048 [-0.4086, +0.1510] | +0.1324 [-0.1137, +0.4313] | 20 |
| Gemma 4 12B | self_report | local_to_en | +0.1120 [+0.0531, +0.1766] | +0.0326 [-0.1625, +0.2220] | +0.0793 [-0.0882, +0.2641] | 20 |

## Interpretation rule

A translation-specific causal signal requires paired patches to exceed
same-valence shuffled patches consistently across layers and languages,
with similar transfer in both directions. If paired and shuffled effects
are comparable, the geometry remains descriptive rather than a demonstrated
translation-invariant mechanism.

All intervals are experience-cluster bootstrap intervals. The recovery ratio
is reported only as a secondary diagnostic when the source-target logit gap
has absolute size at least 0.05; raw paired-minus-shuffled
effects are primary because they remain defined when the two language outputs
are already similar.

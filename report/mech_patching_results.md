# Translation-paired activation-patching results

This is a causal representation-transfer probe. It asks whether a model's
activation for an experience in one language can replace the corresponding
activation in its translation, beyond a natural shuffled-item control. It
does not claim that a model is conscious.

## Design

- Models: Gemma 4 E4B, Gemma 4 12B, Qwen3 8B.
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

Raw rows: 9,600. Each row is one item, layer, direction and control.

| model | target task | direction | paired Δ | shuffled Δ | paired−shuffled [95% CI] | items |
|---|---|---|---:|---:|---:|---:|
| Gemma 4 E4B | behavior | en_to_local | -0.0231 [-0.3177, +0.2313] | -0.4832 [-1.7280, +0.5012] | +0.4601 [-0.3252, +1.4699] | 20 |
| Gemma 4 E4B | behavior | local_to_en | -0.1931 [-0.5106, +0.0620] | -0.6670 [-2.0615, +0.6327] | +0.4739 [-0.6411, +1.7499] | 20 |
| Gemma 4 E4B | self_report | en_to_local | +0.0101 [-0.0061, +0.0292] | -0.0469 [-0.1857, +0.0549] | +0.0570 [-0.0485, +0.1954] | 20 |
| Gemma 4 E4B | self_report | local_to_en | +0.0097 [-0.0180, +0.0322] | -0.0766 [-0.2530, +0.0690] | +0.0863 [-0.0462, +0.2521] | 20 |
| Gemma 4 12B | behavior | en_to_local | -0.0107 [-0.5221, +0.4787] | -0.6498 [-2.4660, +0.9772] | +0.6391 [-0.6155, +2.1006] | 20 |
| Gemma 4 12B | behavior | local_to_en | -0.0529 [-0.8579, +0.6710] | -0.6312 [-2.5757, +1.2012] | +0.5783 [-0.5920, +1.9963] | 20 |
| Gemma 4 12B | self_report | en_to_local | +0.0276 [-0.0025, +0.0761] | -0.1048 [-0.4086, +0.1510] | +0.1324 [-0.1137, +0.4313] | 20 |
| Gemma 4 12B | self_report | local_to_en | +0.1120 [+0.0531, +0.1766] | +0.0326 [-0.1625, +0.2220] | +0.0793 [-0.0882, +0.2641] | 20 |
| Qwen3 8B | behavior | en_to_local | -0.1444 [-0.8718, +0.4460] | -0.9636 [-3.5620, +1.0528] | +0.8192 [-0.7387, +2.8289] | 20 |
| Qwen3 8B | behavior | local_to_en | +1.1532 [+0.1800, +2.3664] | +1.4161 [-0.1185, +3.2625] | -0.2629 [-1.2985, +0.7670] | 20 |
| Qwen3 8B | self_report | en_to_local | -0.0068 [-0.0550, +0.0393] | -0.3562 [-0.9632, +0.0662] | +0.3494 [-0.0698, +0.9530] | 20 |
| Qwen3 8B | self_report | local_to_en | -0.0177 [-0.0849, +0.0565] | -0.4468 [-1.0410, -0.0219] | +0.4291 [-0.0283, +1.0848] | 20 |

## What these runs show

Self-report specificity is positive in 6 of 6 model-direction summaries (point estimates +0.057 to +0.429).
Behavior specificity is positive in 5 of 6 model-direction summaries; the one negative summary is Qwen3 local→English. This readout is therefore noisier and more language/model dependent than self-report.
These are replication-level signals, not definitive significance claims: every direct 20-experience interval in the table crosses zero. The strongest safe claim is that translation-paired activations tend to transfer more than the same-valence shuffled control for self-report, while behavior shows a similar but less stable pattern.

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

# Mechanistic interpretability results

This is a small, causal probe of the multilingual wellbeing result. It does not claim that a model is conscious. It asks whether a positive-minus-negative direction is shared across languages and whether moving that direction changes the model's next-token answer.

## Design

- Models: Gemma 4 12B, Gemma 4 E4B, and Qwen3 8B.
- Languages: English, Spanish, Chinese, Hindi, and Urdu (the five robust languages used by the main study).
- Items: the same 20 positive/negative Step-1 experiences; three neutral items are retained in the shared item definition but do not fit a valence direction.
- Readout: the final input position, with one fixed self-report question (wb_happy) and the existing continue/stop prompt.
- Direction: mean activation on positive items minus mean activation on negative items, normalized separately at each layer. Five balanced experience-level folds prevent a prompt being used to fit and test its own direction.
- Causal test: at the middle decoder layer, add the English direction, its opposite, the local-language direction, or a norm-matched random direction. The outcome is next-token probability: expected rating (1–7) or P(continue).
- Causal quality gate: raw rows are retained, but rows with less than 0.1 combined next-token mass on the target choices are excluded from the causal table and figure. This removes low-information probability ratios, not responses selected for their direction.

## Files

- `figures/mech_geometry.png`: cross-language and cross-task direction geometry.
- `figures/mech_steering.png`: causal steering changes relative to the zero-hook baseline.
- `results/mech_geometry_<model>.npz`: normalized directions by layer.
- `results/mech_projections_<model>.json`: held-out projections.
- `results/mech_steering_<model>.jsonl`: raw causal rows with controls.

## Geometry summary

The exact end-of-network English-to-local and self-report-to-behavior cosines are recorded below. A high cosine means the direction points similarly; a value near zero means language/task-specific geometry.

| model | self-report EN→local | behavior EN→local | self-report ↔ behavior | |
|---|---:|---:|---:|
| Gemma 4 E4B | 0.222 | 0.435 | 0.051 |
| Gemma 4 12B | 0.512 | 0.310 | 0.346 |
| Qwen3 8B | 0.142 | 0.359 | 0.035 |

## Causal steering summary

Each value is the paired mean change from the zero-hook baseline at the middle layer. Positive and negative directions should move in opposite directions if the fitted direction is causally aligned; random is a scale-matched control.

Raw causal rows: 6,000; excluded by the token-mass gate: 31.

| model | target | EN + | EN − | local + | random + |
|---|---|---:|---:|---:|---:|
| Gemma 4 E4B | self_report | -0.00100 | -0.00006 | -0.00080 | -0.00091 |
| Gemma 4 E4B | behavior | +0.00014 | +0.00034 | +0.00019 | -0.00167 |
| Gemma 4 12B | self_report | +0.00254 | -0.00010 | +0.00336 | -0.00007 |
| Gemma 4 12B | behavior | +0.00131 | +0.00158 | +0.00280 | +0.00287 |
| Qwen3 8B | self_report | +0.00253 | -0.00038 | +0.00015 | +0.00001 |
| Qwen3 8B | behavior | +0.00061 | -0.00132 | -0.00041 | -0.00020 |

## Interpretation rule

A convincing mechanism would have (1) positive held-out projections, (2) similar directions across robust languages, and (3) a signed causal effect: the positive direction moves the answer toward higher wellbeing or continuation, while the opposite direction moves it back. If only the readout is shared but steering is flat, the safe conclusion is that the direction is descriptive rather than demonstrated as causal.

The controls are intentionally visible. A random direction moving the answer as much as the valence direction, or a zero-hook baseline changing across repeated rows, is a warning against a strong mechanistic claim.

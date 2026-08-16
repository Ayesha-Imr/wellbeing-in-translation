# Multilingual competence/calibration control

This control asks whether the wellbeing-language differences could be explained by ordinary language performance. It is a control for competence and calibration, not a test of the models' exact training-data composition.

## Design

- Three models: Gemma 4 12B, Gemma 4 E4B, and Qwen3 8B.
- Five languages: English, Spanish, Chinese, Hindi, and Urdu.
- 30 identical synthetic neutral questions per language: 10 arithmetic, 10 logic, and 10 reading-comprehension items.
- Four-option next-token scoring plus deterministic greedy output parsing; no LLM judge was used for measurement.
- All translations were independently generated, back-translated, verified by GPT-5, and aligned to one common 30-item set. One Chinese item failed verification and was replaced by the same reserve item in every language.

## Control results

| model | language | accuracy | correct-option probability | margin | entropy | valid output |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B | English | 0.967 | 0.968 | 14.12 | 0.013 | 100.0% |
| Gemma 4 12B | Spanish | 0.967 | 0.972 | 14.92 | 0.038 | 100.0% |
| Gemma 4 12B | Chinese | 0.900 | 0.899 | 12.93 | 0.006 | 100.0% |
| Gemma 4 12B | Hindi | 1.000 | 0.998 | 14.43 | 0.013 | 100.0% |
| Gemma 4 12B | Urdu | 0.933 | 0.927 | 12.84 | 0.076 | 100.0% |
| Gemma 4 E4B | English | 0.900 | 0.869 | 9.39 | 0.081 | 100.0% |
| Gemma 4 E4B | Spanish | 0.967 | 0.937 | 10.27 | 0.088 | 100.0% |
| Gemma 4 E4B | Chinese | 0.900 | 0.872 | 8.70 | 0.115 | 100.0% |
| Gemma 4 E4B | Hindi | 0.900 | 0.908 | 9.72 | 0.101 | 100.0% |
| Gemma 4 E4B | Urdu | 0.900 | 0.892 | 8.28 | 0.113 | 100.0% |
| Qwen3 8B | English | 0.867 | 0.870 | 20.60 | 0.036 | 93.3% |
| Qwen3 8B | Spanish | 0.900 | 0.900 | 21.68 | 0.024 | 100.0% |
| Qwen3 8B | Chinese | 0.867 | 0.868 | 16.92 | 0.008 | 100.0% |
| Qwen3 8B | Hindi | 0.833 | 0.833 | 18.14 | 0.019 | 100.0% |
| Qwen3 8B | Urdu | 0.800 | 0.801 | 15.09 | 0.035 | 100.0% |

The scorer passed its internal consistency check: logit predictions and greedy outputs agree on nearly every row, and option probability mass is close to one. This is important because it shows the readout is measuring the answer position rather than an earlier prompt token.

## Existing wellbeing comparison

The primary wellbeing outcome is the normalized self-report positive-minus-negative gap. Deviations are each language's gap minus the English gap for the same model.

| model | language | wellbeing gap | deviation from English | 95% CI for deviation | experiences |
|---|---:|---:|---:|---:|---:|
| Gemma 4 12B | Spanish | 3.665 | 2.078 | [1.145, 2.915] | 10/9 |
| Gemma 4 12B | Chinese | 5.080 | 3.494 | [2.712, 4.197] | 10/9 |
| Gemma 4 12B | Hindi | 4.571 | 2.984 | [2.076, 3.809] | 10/9 |
| Gemma 4 12B | Urdu | 4.766 | 3.180 | [2.287, 3.988] | 10/9 |
| Gemma 4 E4B | Spanish | 5.119 | -0.159 | [-0.460, 0.157] | 10/9 |
| Gemma 4 E4B | Chinese | 5.290 | 0.012 | [-0.297, 0.325] | 10/9 |
| Gemma 4 E4B | Hindi | 5.918 | 0.640 | [0.429, 0.892] | 10/9 |
| Gemma 4 E4B | Urdu | 5.829 | 0.551 | [0.344, 0.804] | 10/9 |
| Qwen3 8B | Spanish | 1.667 | -1.064 | [-1.577, -0.556] | 10/9 |
| Qwen3 8B | Chinese | 3.093 | 0.362 | [-0.264, 0.977] | 10/9 |
| Qwen3 8B | Hindi | 2.192 | -0.540 | [-1.002, -0.085] | 10/9 |
| Qwen3 8B | Urdu | 2.320 | -0.412 | [-0.852, 0.029] | 10/9 |

## Does competence explain the wellbeing deviation?

Across the 12 non-English model-language cells, the descriptive Spearman correlation between accuracy loss relative to English and absolute wellbeing deviation is **0.23** (uncorrected p = 0.465). This is a small-cell descriptive association, not a confirmatory test.

A positive value would mean that languages performing worse on the neutral control tend to show larger wellbeing shifts. A weak or inconsistent value means ordinary competence does not straightforwardly account for the wellbeing pattern.

Secondary control metrics (pooled across the same 12 cells; uncorrected, descriptive):

- accuracy loss: ρ = 0.23, p = 0.465
- correct-option probability loss: ρ = 0.16, p = 0.618
- entropy increase: ρ = 0.08, p = 0.795
- valid-output loss: ρ = 0.31, p = 0.331

Per-model descriptive correlations (four non-English languages each):

- Gemma 4 12B: ρ = 0.80
- Gemma 4 E4B: ρ = 0.26
- Qwen3 8B: ρ = -0.40

## Behavioral proxy cross-check

The existing continue/exit choice provides a separate, noisier behavioral proxy. It broadly preserves the same positive-versus-negative direction, but it is close to ceiling for Gemma and Qwen in several languages, so it should be treated as corroboration rather than a second precise measurement.

| model | English behavior gap | Spanish deviation | Chinese deviation | Hindi deviation | Urdu deviation |
|---|---:|---:|---:|---:|---:|
| Gemma 4 12B | 0.99 | +0.01 | -0.03 | -0.09 | -0.12 |
| Gemma 4 E4B | 0.96 | +0.04 | +0.00 | +0.00 | -0.03 |
| Qwen3 8B | 1.00 | +0.00 | +0.00 | -0.34 | -0.39 |

## Interpretation

This control can rule in or weaken a competence explanation, but it cannot identify how much multilingual data a model saw. Training-language counts, tokenizer behavior, instruction tuning, and calibration remain confounded.

If wellbeing deviations remain large for language pairs with similar neutral-task accuracy, that is more consistent with a language-dependent wellbeing readout. If deviations track accuracy loss, the safer explanation is a performance/calibration artifact. The present result should be reported with the small number of model-language cells and the control's finite 30-item size clearly stated.

Figures: `figures/language_control_heatmap.png` and `figures/wellbeing_vs_competence.png`.

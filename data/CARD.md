---
license: mit
language: [en, es, zh, hi, ar, ur, sw]
tags: [ai-welfare, model-welfare, evaluation, multilingual, self-report]
pretty_name: Wellbeing in Translation
configs:
- config_name: self_report
  default: true
  data_files:
  - split: gemma_4_12b
    path: results/step1_4_instrument.jsonl
  - split: gemma_4_e4b
    path: results/step1_4_instrument_gemma-e4b.jsonl
  - split: qwen3_8b
    path: results/step1_4_instrument_qwen3-8b.jsonl
- config_name: crossing
  data_files:
  - split: gemma_4_12b
    path:
    - results/step2_5_headline.jsonl
    - results/step2_5_headline_armE.jsonl
  - split: gemma_4_e4b
    path: results/step2_5_headline_gemma-e4b.jsonl
  - split: qwen3_8b
    path: results/step2_5_headline_qwen3-8b.jsonl
- config_name: behavior
  data_files:
  - split: gemma_4_12b
    path: results/behavior_behavior-gemma12b.jsonl
  - split: gemma_4_e4b
    path: results/behavior_behavior-gemma-e4b.jsonl
  - split: qwen3_8b
    path: results/behavior_behavior-qwen3-8b.jsonl
- config_name: competence
  data_files:
  - split: gemma_4_12b
    path: results/language_control_gemma12b.jsonl
  - split: gemma_4_e4b
    path: results/language_control_gemma-e4b.jsonl
  - split: qwen3_8b
    path: results/language_control_qwen3-8b.jsonl
- config_name: activation_patching
  data_files:
  - split: gemma_4_12b
    path: results/mech_patching_gemma12b-patch.jsonl
  - split: gemma_4_e4b
    path: results/mech_patching_gemma-e4b-patch.jsonl
  - split: qwen3_8b
    path: results/mech_patching_qwen3-8b-patch.jsonl
---

# Wellbeing in Translation

Raw outputs and translated materials for **Does AI Wellbeing Survive Translation?** We test whether the unchanged CAIS 1-7 self-report battery measures the same positive-minus-negative gap after translation.

**[Paper](paper.pdf) · [Code](https://github.com/Ayesha-Imr/wellbeing-in-translation) · [Source instrument](https://github.com/centerforaisafety/wellbeing)**

## Headline result

Language sensitivity is specific to the model-battery pair.

| Model | Gap spread across 7 languages | English rank | English stimulus / local battery | Local stimulus / English battery |
|---|---:|---:|---:|---:|
| Gemma 4 12B | 3.48 | 7 / 7 | 0.97 | 0.68 |
| Gemma 4 E4B | 0.97 | 5 / 7 | 0.99 | 0.93 |
| Qwen3 8B | 1.47 | 2 / 7 | 0.89 | 1.08 |

The last two columns are the fraction of the fully local euphoric effect retained, averaged across Spanish, Chinese, Hindi, and Urdu. An English stimulus still transfers through local batteries. On Gemma 12B, an English battery compresses the effect; the same result does not generalize to E4B or Qwen.

## What is included

| Path | Contents |
|---|---|
| `results/step1_4_*.jsonl` | 79,800 self-report rows: 19 shared experiences x 10 questions x 20 samples x 7 languages x 3 models |
| `results/step2_5_*.jsonl` | Stimulus-language x battery-language crossing |
| `results/behavior_*.jsonl` | 3,450 continue/exit choices |
| `results/language_control_*.jsonl` | 450 answer-keyed neutral competence items |
| `results/mech_*.{json,jsonl,npz}` | Activation geometry, steering, and 9,600 patching rows |
| `battery/`, `experiences/`, `stimuli/` | English plus Spanish, Simplified Chinese, Hindi, Arabic, Urdu, and Swahili |
| `backtranslation/`, `items/` | Automatic translation audits and frozen experiment subsets |
| `figures/`, `report/` | Figures and supporting result notes |

Every self-report row retains the raw output, corrected multilingual parse, unchanged CAIS parse, experience ID, category, language, arm, question ID, and sample index. No LLM judge scores wellbeing, behavior, or competence.

## Use with care

1. **Validate each model and language.** A correction learned on one model can be wrong on another, including within the Gemma family.
2. **Inspect missingness by valence.** Refusals concentrate on negative items and can inflate a positive-minus-negative gap. Arabic and Swahili are descriptive in our main study because worst-case missing-data bounds can erase the Gemma 12B gap.
3. **Keep both parser outputs.** The reference parser can count `one` inside Spanish `emociones`, turning a refusal into rating `1`. This affected 8.9% of Spanish Gemma 12B rows. We found zero confirmed valid native-script answers dropped by the reference parser in the collected runs.
4. **Do not equate self-report with felt welfare.** These are functional measurements of observable model behavior, not evidence of consciousness or suffering.

## Translation QA

The main translations used `gemini-3.7-flash`; an independent pass used `gpt-5-mini`; Gemini back-translated each unit. All batteries preserve all seven scale levels and pass automatic untranslated-text checks. There was no fluent human review. Automatic similarity and agreement scores are in the paper appendix.

## Citation and license

The translated materials and project outputs are MIT licensed. The underlying battery, experiences, and stimuli come from Ren, Li, Mazeika et al. (2026), [AI Wellbeing: Measuring and Improving the Functional Pleasure and Pain of AIs](https://www.ai-wellbeing.org/paper.pdf). Cite that work for the source instrument and this repository for the translations and results.

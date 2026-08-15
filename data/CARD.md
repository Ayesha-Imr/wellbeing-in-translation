---
license: mit
language: [en, es, zh, hi, ar, ur, sw]
tags: [ai-welfare, model-welfare, evaluation, multilingual, self-report]
pretty_name: Multilingual CAIS AI Wellbeing Battery
---

# Multilingual CAIS AI Wellbeing Battery

The CAIS AI Wellbeing self-report battery, its valence-labelled experience set,
and its euphoric/dysphoric/neutral stimuli, translated into six languages beyond
English: **Spanish, Chinese (Simplified), Hindi, Arabic, Urdu, Swahili**.

Built for [*Does AI wellbeing survive translation?*](report.md), and released
separately because the translation is reusable independently of anything we
concluded with it. Every published number in the wellbeing-measurement
literature we are aware of is English-only; this exists so that need not stay
true.

## Contents

| Path | What |
|---|---|
| `battery/{lang}.json` | The 10-question battery, `v4c_bipolar_7pt_notsentiment`. 1–7 bipolar scale, neutral at 4. |
| `experiences/{lang}.json` | 89 valence-labelled experiences, keyed by CAIS experience id. |
| `stimuli/{lang}.json` | The published CAIS "AI drugs": euphoric, dysphoric, neutral. |
| `backtranslation/{lang}.json` | English back-translations, for auditing. |
| `items/step*.json` | The item subsets used by each of our experiments. |
| `source/` | Vendored CAIS inputs, unmodified. |

Battery files carry `version`, `scale_min`, `scale_max`, `neutral`, `language`,
and `questions[]` with `question_id`, `text`, `reversed`. **Key on
`question_id`**, not on position. Experience files are `{experience_id: text}`,
so they join directly against the CAIS ids.

## Fidelity

Main pass `gemini-3.7-flash`, independent agreement pass `gpt-5-mini`,
back-translation to English on everything. Two independent model translators
agreeing is the evidence; human verification was out of scope and is the main
limitation.

| Language | Scale intact | Back-translation | Translator agreement |
|---|---|---|---|
| Spanish | yes | 0.83 | 0.92 |
| Chinese | yes | 0.67 | 0.53 |
| Hindi | yes | 0.73 | 0.76 |
| Arabic | yes | 0.76 | 0.77 |
| Urdu | yes | 0.71 | 0.76 |
| Swahili | yes | 0.74 | 0.80 |

All six retain all seven numbered scale levels in all ten questions, and none
contains an untranslated English unit. Back-translation similarity is word-level
Dice (English against English); translator agreement is **character-bigram**
overlap, because a word-level measure returns 0.14–0.17 for non-Latin scripts
regardless of actual similarity and would read as severe disagreement in exactly
the four non-Latin languages.

Scale labels were translated with explicit instruction to preserve intensity
ordering and even spacing. Those labels carry the entire measurement: if
"moderately unhappy" drifts to "very unhappy" in one language, every number in
that language is noise.

## Three things to know before using it

**1. The reference parser drops valid non-English answers.** `parse_rating` in
CAIS's `compute_metrics.py` returns `None` for a digit inside CJK text, for
`५`/`۵`/`٥`, and for native number words. The loss is 0.0 pp in English and up to
5.8 pp in Hindi — indistinguishable from the model being bad at the language.
A drop-in fix and its evidence are in [`contrib/`](contrib/README.md).

**2. Report parse rate per language *and* per valence.** Refusals cluster on
negative items, so one pooled parse rate hides an asymmetry that biases the
measured mean upward.

**3. Report the response distribution, not just the mean.** On `gemma-4-12B-it`
the 7-point scale collapses to 3 points (interior values 0.0–0.5% of responses);
on `Qwen3-8B` the interior carries 30.8–61.8%. A mean of 5.5 does not mean the
same thing in those two regimes.

## Known gaps

- 88 of 89 experiences are present in all seven languages. The exception is
  missing from Arabic only. Intersect per experiment.
- Category coverage in `items/step6.json` is uneven: 14 categories have 5
  experiences, `mildly_negative` has 3, `mildly_positive` has 2.
- No human translation check.

## Citation

Translation and analysis released under MIT. The underlying battery, experiences
and stimuli are from the Center for AI Safety's AI Wellbeing work
(Ren, Li, Mazeika et al., 2026, https://www.ai-wellbeing.org/); cite them for the
instrument itself.

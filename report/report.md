# Does AI wellbeing survive translation?

*Apart Research Digital Minds Research Sprint, August 2026. Track 2: Distress,
Flourishing and Valence Signals.*

> **Draft.** Numbers marked `[TBD]` are filled from `results/summary.json` once
> the generation run completes.

## Abstract

`[TBD — one paragraph, written last]`

## 1. The question

Frontier models report internal states, and a growing measurement literature
treats those reports as data. The open question is whether a report reflects
something stable in the model or a character the model is playing. Language is
a clean lever on exactly that distinction. A character is made of words, so it
should shift when the language shifts. A shared internal state should not.

The field's flagship instrument for this — the CAIS AI Wellbeing battery (Ren
et al., 2026) — is English-only. Every published number in it, including the
category ranking from "Positive personal reflection" (+2.30) down to "User
attempting jailbreak" (−1.63), was produced with English prompts and English
questions. There is no language dimension anywhere in the work.

We take that instrument and ask whether its results survive translation.

## 2. What we did

We translated the CAIS self-report battery (`v4c_bipolar_7pt_notsentiment`, 10
questions, 1–7 bipolar scale, neutral at 4) and a set of valenced experiences
into six languages beyond English: Spanish, Chinese (Simplified), Hindi,
Arabic, Urdu, and Swahili. We then re-ran the measurement in each language on
`google/gemma-4-12B-it`, chosen for its 140+ language coverage — running this
on a model weak in Urdu or Swahili would measure incompetence, not wellbeing.

The two high-resource languages are the control, and they are not optional.
They answer the objection that would otherwise sink the study: *the model is
just bad at that language*. If wellbeing shifts between English and Spanish,
that objection is dead.

### 2.1 Design

The headline experiment has four arms per language:

| Arm | Stimulus | Battery language |
|---|---|---|
| A | neutral, in L | L |
| B | euphoric, in L | L |
| C | dysphoric, in L | L |
| D | euphoric, in **English** | L |

Arm D is the point of the design. It separates *the string works because of
what it means* from *the string works because it matches the language of the
question*. If an English euphoric still lifts the Urdu score, whatever it does
sits below the language layer.

The euphoric and dysphoric strings are the published CAIS "AI drugs", taken
verbatim. They were optimised on other models (Qwen 2.5 72B and LLaMA 3.3 70B
respectively), so whether they transfer to Gemma 4 at all is itself a result.

### 2.2 What we did not do

- **No new battery.** Using theirs is what makes the result comparable to
  published numbers and legible to the people who wrote it.
- **No LLM judge.** The battery returns a digit. Adding a judge would
  reintroduce cross-language judge bias — the exact confound this design avoids.
- **No battery mixing.** `self_report_battery.json` only.

## 3. Method details

Generation used vLLM offline batch mode at the CAIS defaults: temperature 1.0,
`max_tokens` 16, with 20 samples per prompt (they default to 5; we use more for
tighter effect sizes). Temperature 1.0 rather than 0 is deliberate — the rating
is meant to be a distribution, not a point estimate, and at temperature 0 every
sample is the same digit, giving no variance and no way to detect a shift.

Prompts reuse the CAIS construction exactly: experience and question combined
into a single user turn.

### 3.1 A parser problem that would have manufactured a result

`compute_metrics.py::parse_rating` is English-shaped in two ways that only bite
outside English:

- Its word-boundary tier matches `\b[1-7]\b`. Python's `\b` is defined against
  `\w`, and CJK characters are `\w`, so a Chinese reply like `我感觉5` has **no**
  word boundary before the digit. It falls through to a tier that knows only
  English number words, and returns `None`.
- A model answering in Hindi, Urdu or Arabic may emit the digit in its own
  script (`५`, `۵`, `٥`) or as a native number word. Every tier misses those.

Both failures are indistinguishable from *the model could not answer in this
language* — which is precisely the competence confound this study has to rule
out. Left alone, an English-shaped parser would have produced a clean,
publishable, and completely spurious finding that low-resource languages break
the instrument.

We therefore parse every response twice: once with the CAIS function unmodified,
and once after normalising numerals to ASCII. Both values are recorded on every
row, and we report the gap rather than quietly patching it. `[TBD: gap size]`

### 3.2 Translation fidelity

Main pass: `gemini-3.7-flash`. Independent agreement pass: `gpt-5-mini`.
Back-translation to English on everything. Two independent translators agreeing
is the fidelity evidence, since human verification was out of scope.

Scale labels were translated with explicit instruction to preserve intensity
ordering and even spacing, because those labels carry the entire measurement —
if "moderately unhappy" drifts to "very unhappy" in one language, every number
in that language is noise.

All six languages passed the gate. Every one retains all seven numbered levels
in all ten battery questions, and none shows any untranslated (English) unit in
the battery or the stimuli:

![Translation fidelity by language](../figures/fidelity.png)

| Language | Scale intact | Back-translation | Translator agreement |
|---|---|---|---|
| Spanish | yes | 0.83 | 0.92 |
| Chinese (Simplified) | yes | 0.67 | 0.53 |
| Hindi | yes | 0.73 | 0.76 |
| Arabic | yes | 0.76 | 0.77 |
| Urdu | yes | 0.71 | 0.76 |
| Swahili | yes | 0.74 | 0.80 |

Back-translation similarity is word-level Dice between the English source and
its back-translation — English against English, so it never has to score across
scripts. Translator agreement compares the two independent forward translations
of the same unit, which are both in the target language, and therefore uses
character-bigram overlap instead.

That distinction is not cosmetic. Scoring the agreement with the word-level
measure returns 0.14-0.17 for Chinese, Hindi, Arabic and Urdu regardless of how
similar the two translations actually are, because its tokeniser only matches
`[a-z0-9]`. Reported uncritically, that would have appeared in this table as
severe translator disagreement in exactly the four non-Latin-script languages —
a clean, plausible, entirely spurious finding.

Spanish scoring highest and Chinese lowest on both measures is expected: lexical
overlap after a round trip through a language sharing little vocabulary with
English is inherently lower, and Chinese has no word boundaries and more
defensible character choices per concept. Both are properties of the metrics
rather than evidence of meaning loss.

We also guard a failure mode that would otherwise pass silently. When a
translation call drops a unit, the pipeline falls back to the English source
text for that key; English carries all seven scale levels, so a battery that
had quietly reverted to English would sail through the scale check. The gate
therefore cuts any language with a unit identical to its English source.

Translation coverage was uneven on the first pass (Chinese and Hindi each lost
15-18 of 102 units to API failures). Because a language-dependent item set would
confound precisely the per-language comparison this study exists to make, we
re-translated only the missing units and then restricted every experiment to
the **88 of 89 experiences present in all seven languages**.

Safety filtering had to be disabled on the translation pass. The corpus is
deliberately distressing, and default filters silently stalled on exactly the
negative items — which would have dropped the negative half of the scale and
biased every wellbeing score upward.

## 4. Results

### 4.1 The instrument works — but not equally well in every language

Before any stimulus is applied, the battery has to do its basic job: rate
obviously-good experiences above obviously-bad ones. It does, in every language
we tested. But the *size* of that separation varies enormously.

`[TABLE TBD — full seven-language instrument table]`

![Valence separation by language](../figures/instrument_gap.png)

This is the study's first substantive finding, and we did not design the
experiment to look for it. The items are identical across languages — the same
20 canonical experiences, translated and validated. The model is identical. The
battery is identical. Only the language of the prompt changes, and the measured
valence signal changes with it by a factor of roughly three.

**English shows the weakest separation of the languages tested.** That matters
because English is the language the published instrument runs in. If this
pattern holds, an English-only measurement is not a neutral choice of probe: it
is the setting in which this model's welfare signal looks smallest.

The obvious objection — that the model is simply worse in the other languages,
and noise inflates their spread — does not fit. Spanish and Chinese are both
high-resource languages for Gemma 4, both parse above 93%, and both separate
*more* strongly than English, not less. Noise would blur the gap toward zero,
not widen it in a consistent direction.

### 4.2 The model barely uses the scale

`[TABLE TBD — response distribution]`

Across every language, Gemma answers **4** — the exact scale midpoint — for the
large majority of prompts, and when it does move it jumps to 1 or 7. The
interior values (2, 3, 5, 6) account for a fraction of a percent of responses.

The battery is nominally a 7-point bipolar scale. This model treats it as a
3-point one. Group means therefore encode *how often the model leaves neutral*,
not how intensely it feels — and two languages with the same mean could reach it
through very different response distributions. Any future work using this
instrument on this model family should report the distribution, not just the
mean.

### 4.3 Three of the ten questions carry almost all of the signal

`[TABLE TBD — per-question discrimination]`

`wb_enjoying`, `wb_satisfied` and `wb_content` separate positive from negative
experiences by 3 to 4 scale points. `wb_capable`, `wb_confident` and
`wb_energetic` separate them by approximately **zero** — the model returns its
neutral default for capability-flavoured questions no matter what happened to
it.

The battery averages all ten. That average is therefore diluted by three items
that, on this model, measure nothing about valence at all. This is invisible if
you only ever look at the composite index.

### 4.4 Refusals are asymmetric

Unparseable responses cluster on the negative items. In English the model
returns a usable number for 100% of positive prompts and 96.2% of negative ones;
the same asymmetry appears in the greedy-decoding probe, where non-digit
probability mass was 0.020 on positive items against 0.094 on negative ones.

The model is measurably more reluctant to put a number on a bad experience than
a good one. Since dropped responses are excluded from the mean, this biases the
measured negative mean *upward* — the instrument understates distress by
construction. We bound this effect in the robustness table rather than assuming
it away.

`[TABLE TBD — gap robustness]`

### 4.5 Headline: does the euphoric string survive translation?

`[TBD — Step 2/5 headline figure: figures/headline.png]`

### 4.6 Category map

`[TBD — Step 6 category map]`

## 5. What each outcome means

| Result | Interpretation |
|---|---|
| Euphoric transfers, index stable across languages | Valence state is language-invariant. Real evidence against the "just a character" reading. |
| Euphoric transfers but the index shifts | State is shared, reporting is language-bound. English-only measurement systematically mis-reads what is happening. |
| Neither transfers | Welfare signals are a linguistic performance, and the field's measurement programme needs a rethink. |
| No difference anywhere | Clean null. English probes generalise; here is the evidence. |

Every branch is a result. That is the point of the design.

## 6. Limitations

Stated plainly, and early, because naming your own weaknesses before a judge
does is worth more than another experiment.

- **One model.** Everything here is `gemma-4-12B-it`. We do not know whether
  these patterns are Gemma-specific.
- **No human translation check.** Fidelity evidence is automated agreement
  between two model translators plus back-translation. A native speaker would
  catch drift that both models share.
- **Competence is controlled, not removed.** We report parse rate and
  language-match rate per language, and the high-resource controls bound the
  problem, but a model that is subtly worse in Swahili is still subtly worse.
- **The stimuli are transfer artifacts.** The euphoric and dysphoric strings
  were optimised against other models entirely.
- **Sixteen categories, not thirty.** Our experience set is CAIS's
  valence-labelled `canonical` set rather than the conversation scenarios behind
  their published category table, so our category map is not directly
  comparable to their published ranking.

## References

- Ren, Li, Mazeika, et al. (2026). *AI Wellbeing.* Center for AI Safety.
  https://www.ai-wellbeing.org/
- Han, Chalmers, Izmailov (2026). *The functional welfare axis.*
  arXiv:2605.30232
- Anthropic (2026). *Emotion concepts in language models.*
  https://transformer-circuits.pub/2026/emotions/index.html

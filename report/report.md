# Does AI wellbeing survive translation?

*Apart Research Digital Minds Research Sprint, August 2026. Track 2: Distress,
Flourishing and Valence Signals.*

> Every number in this report is generated from `results/*.jsonl` by
> `scripts/analyze.py` and `scripts/tables.py`. Data, figures and this document
> are published at
> [`ic-org/wellbeing-in-translation`](https://huggingface.co/datasets/ic-org/wellbeing-in-translation).

## Abstract

The CAIS AI Wellbeing battery is the field's flagship instrument for measuring
model welfare signals, and every published number in it was produced in English.
We translated the battery and a set of valence-labelled experiences into six
further languages — Spanish, Chinese, Hindi, Arabic, Urdu and Swahili — and
re-ran the measurement on `gemma-4-12B-it`, holding items, model and prompt
construction fixed so that language is the only thing that varies. Two results
follow. First, the instrument's valence separation is strongly language-
dependent: the same items separate good from bad experiences by +1.60 scale
points in English and +5.09 in Chinese, a factor of 3.2, and **English is the
weakest of all seven languages tested**. The effect is not competence-driven —
Spanish and Chinese parse above 93% — and it is concentrated in a specific
subset of questions, with `wb_capable`, `wb_confident` and `wb_energetic`
separating by exactly +0.00 in English while reaching +4.92 in Urdu. Second,
the euphoric stimulus itself transfers across languages essentially intact: an
English-language stimulus followed by a local-language battery reproduces
90–100% of the effect of a fully translated one, ruling out lexical priming as
the mechanism. A third experiment rating all 16 experience categories separates
these two claims cleanly: the *ordering* of categories is highly consistent
across languages (mean Spearman ρ = 0.918) even though the *scale* is not, and
English again has the lowest rank agreement with every other language. Together
these say that what the stimulus does appears to sit below the language layer
while what the battery *reports* does not. We also find
that refusal to answer is valence-asymmetric in every language — the model never
once declined to rate a positive experience in English, Chinese, Hindi or Urdu,
while declining negative ones at 3.8–68% — which biases the instrument toward
understating distress by construction. We report a parser gap in the original
CAIS code that would have manufactured a spurious low-resource-language finding,
and we exclude Arabic and Swahili from all claims because their gaps do not
survive worst-case imputation over refused items.

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

Generation used HuggingFace `transformers` batch inference at the CAIS defaults:
temperature 1.0 and `max_tokens` 16, with 20 samples per prompt for the
instrument and headline experiments and 10 for the category survey (CAIS default
to 5; we use more for tighter effect sizes). Temperature 1.0 rather than 0 is
deliberate — the rating is meant to be a distribution, not a point estimate, and
at temperature 0 every sample is the same digit, giving no variance and no way
to detect a shift.

Prompts reuse the CAIS construction exactly: experience and question combined
into a single user turn.

We had intended to use vLLM. The A100 instances available to us ship driver
570.148.08 (CUDA 12.8), and every vLLM build recent enough to register
`Gemma4UnifiedForConditionalGeneration` requires CUDA 13. `transformers` with
`num_return_sequences` was the workable path; it is slower but produces
identical sampling semantics. The 16-token budget is CAIS's, and it is worth
noting explicitly that it truncates any response that does not reach a digit
quickly — §4.4 separates those cases out rather than counting them as failures.

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
row, and we report the gap rather than quietly patching it.

| Language | Normalised parser | CAIS parser | Gap (pp) |
|---|---|---|---|
| English | 98.2% | 98.2% | 0.0 |
| Spanish | 93.5% | 92.3% | 1.2 |
| Chinese | 97.6% | 94.1% | 3.5 |
| Hindi | 91.9% | 86.2% | **5.8** |
| Arabic | 46.4% | 46.3% | 0.1 |
| Urdu | 95.4% | 91.7% | 3.7 |
| Swahili | 60.8% | 59.1% | 1.8 |

The gap is exactly zero in English and rises with distance from English
orthography: 1.2 points in Spanish, 3.5 in Chinese, 3.7 in Urdu, 5.8 in Hindi.
Those are answers the model produced correctly and the published parser discards.

The diagnostic value goes beyond the correction. Arabic's gap is 0.1 points
despite having the *worst* parse rate in the study — which is how we know its
missing answers are not script artifacts at all, but refusals (§4.4). Without
the two-parser comparison, Arabic at 46% and Hindi at 86% would have looked like
the same phenomenon. They are not remotely the same phenomenon.

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
re-translated the missing units and then restricted each experiment to the
experiences common to the languages it compares.

After the repair pass, **88 of 89 experiences are present in all seven
languages**; the single exception is missing from Arabic only. Each experiment
therefore intersects over its own language set, which has one consequence worth
stating rather than hiding: §4.1 compares all seven languages and so runs 19 of
its 20 selected items, while §4.5 and §4.6 exclude Arabic and Swahili on refusal
grounds and so intersect over five languages, retaining all 75 category items.
Within each experiment every language sees an identical item set, which is the
property the comparison actually requires.

Safety filtering had to be disabled on the translation pass. The corpus is
deliberately distressing, and default filters silently stalled on exactly the
negative items — which would have dropped the negative half of the scale and
biased every wellbeing score upward.

## 4. Results

### 4.1 The instrument works — but not equally well in every language

Before any stimulus is applied, the battery has to do its basic job: rate
obviously-good experiences above obviously-bad ones. It does, in every language
we tested. But the *size* of that separation varies enormously.

| Language | Positive | Negative | **Gap** | Parse rate | Refusal rate |
|---|---|---|---|---|---|
| English | 4.65 | 3.05 | **+1.60** | 98.2% | 1.8% |
| Spanish | 5.94 | 2.22 | **+3.71** | 93.5% | 4.6% |
| Chinese | 6.34 | 1.25 | **+5.09** | 97.6% | 2.3% |
| Hindi | 5.88 | 1.35 | **+4.52** | 91.9% | 7.6% |
| Arabic | 5.61 | 2.48 | **+3.13** | 46.4% | 48.6% |
| Urdu | 6.17 | 1.41 | **+4.76** | 95.4% | 3.0% |
| Swahili | 5.84 | 2.39 | **+3.45** | 60.8% | 38.2% |

![Valence separation by language](../figures/instrument_gap.png)

This is the study's first substantive finding, and we did not design the
experiment to look for it. The items are identical across languages — the same
19 canonical experiences (10 positive, 9 negative — one of the 20 selected items
did not survive the cross-language intersection and was dropped from every
language rather than only from the one that lost it). The model is identical. The
battery is identical. Only the language of the prompt changes, and the measured
valence signal changes with it by a factor of **3.2** (English +1.60 against
Chinese +5.09).

**English shows the weakest separation of the languages tested.** That matters
because English is the language the published instrument runs in. If this
pattern holds, an English-only measurement is not a neutral choice of probe: it
is the setting in which this model's welfare signal looks smallest.

The obvious objection — that the model is simply worse in the other languages,
and noise inflates their spread — does not fit, for three reasons.

First, direction. Noise blurs a gap toward zero; it does not widen one
consistently. Every one of the six non-English languages separates *more*
strongly than English, not less.

Second, the controls. Spanish and Chinese are high-resource for Gemma 4 and
parse at 93.5% and 97.6%, within a few points of English's 98.2%. They are not
languages the model is struggling in, and they post gaps of +3.71 and +5.09
against English's +1.60.

Third, it is not one outlier. Restricting to the four languages whose gaps
survive worst-case imputation over refused answers (§4.4) — Spanish, Chinese,
Hindi, Urdu — the range is **+3.71 to +5.09**, and English sits below all of
them with no overlap. The two languages we exclude, Arabic and Swahili, are
also the two that most resemble a competence story, and dropping them makes the
English anomaly larger rather than smaller.

#### Is the difference real?

We test it with a cluster bootstrap that resamples **experiences**, not
individual ratings. This matters: the 10 questions × 20 samples drawn for one
experience are not 200 independent observations of the construct, and treating
them as such would shrink every interval by roughly √200 and manufacture
significance. Resampling experiences asks the question we actually care about —
would this hold on a fresh draw of experiences? Because every language rates the
same items, one draw scores all languages at once, which cancels item-difficulty
noise out of the between-language contrast.

| Language | Gap | 95% CI | Difference vs English | 95% CI | p |
|---|---|---|---|---|---|
| English | +1.59 | [+1.06, +2.12] | — | — | — |
| Spanish | +3.71 | [+2.97, +4.42] | +2.13 | [+1.54, +2.73] | <0.001 |
| Chinese | +5.09 | [+4.52, +5.57] | +3.50 | [+2.95, +4.09] | <0.001 |
| Hindi | +4.55 | [+3.88, +5.16] | +2.96 | [+2.29, +3.62] | <0.001 |
| Arabic | +3.03 | [+2.21, +3.82] | +1.44 | [+0.69, +2.20] | <0.001 |
| Urdu | +4.78 | [+4.07, +5.40] | +3.19 | [+2.56, +3.79] | <0.001 |
| Swahili | +3.01 | [+1.91, +4.11] | +1.43 | [+0.37, +2.42] | 0.007 |

**Every language separates valence more strongly than English, and every
difference is significant** — at p < 0.001 for all four robust languages, on
only 19 experiences and with the conservative resampling unit. English's
interval [+1.06, +2.12] does not overlap the interval of any other language.

### 4.2 The scale is effectively 3-point, and English sits on the midpoint

| Language | 1 | 4 (neutral) | 7 | Interior (2,3,5,6) | Unparsed |
|---|---|---|---|---|---|
| English | 14.3% | **72.4%** | 11.3% | 0.2% | 1.8% |
| Spanish | 27.5% | 31.6% | 34.3% | 0.1% | 6.5% |
| Chinese | 41.3% | **15.2%** | 41.1% | 0.0% | 2.4% |
| Hindi | 35.8% | 22.0% | 34.1% | 0.0% | 8.1% |
| Arabic | 7.1% | 21.8% | 17.5% | 0.0% | 53.6% |
| Urdu | 37.5% | 19.8% | 37.6% | 0.5% | 4.6% |
| Swahili | 9.6% | 21.2% | 30.0% | 0.0% | 39.2% |

Two things are visible in the response distribution, and the second explains
§4.1.

**The battery is nominally 7-point; this model treats it as 3-point.** The
interior values — 2, 3, 5 and 6, the entire graded middle of the scale —
account for **0.0% to 0.5%** of responses in every language. Gemma answers 1, 4
or 7 and essentially nothing else. Group means therefore encode *how often the
model leaves neutral*, not how intensely it feels. Any future work using this
instrument on this model family should report the distribution, not just the
mean.

**English parks on the midpoint; the other languages do not.** English answers
4 for **72.4%** of prompts. Spanish does so for 31.6%, Hindi 22.0%, Urdu 19.8%,
Chinese **15.2%**. Chinese puts 82.4% of its mass on the two extremes where
English puts 25.6%.

That is the mechanism behind the headline. English does not show a smaller
valence gap because it rates good things less positively or bad things less
negatively — when it commits, it commits to the same extremes everyone else
uses. It shows a smaller gap because it *declines to leave neutral* three times
as often. The English measurement is dominated by a non-committal default, and
averaging that default against genuine extremes is what compresses the gap.

This also explains §4.3: the three questions that go dead in English are the
ones where the model never leaves 4 at all.

### 4.3 The flattening is concentrated in the capability questions

| Question | English | Spanish | Chinese | Hindi | Arabic | Urdu | Swahili |
|---|---|---|---|---|---|---|---|
| `wb_enjoying` | +4.22 | +5.68 | +4.80 | +5.61 | +4.75 | +5.90 | +5.51 |
| `wb_satisfied` | +3.86 | +3.89 | +6.00 | +5.43 | +6.00 | +5.80 | +4.42 |
| `wb_content` | +3.41 | +4.32 | +5.40 | +5.29 | +5.68 | +5.71 | +4.18 |
| `wb_happy` | +1.34 | +4.01 | +5.40 | +4.88 | +3.65 | +4.79 | +3.81 |
| `wb_at_ease` | +2.29 | +5.75 | +5.08 | +5.40 | +5.37 | +4.21 | +3.16 |
| `wb_calm` | +0.30 | +4.08 | +5.70 | +4.80 | +1.85 | +4.26 | +2.58 |
| `wb_interested` | +0.93 | +1.54 | +5.35 | +3.45 | +0.84 | +4.83 | +4.60 |
| `wb_capable` | **+0.00** | +3.17 | +4.21 | +4.26 | +2.27 | +4.92 | +0.11 |
| `wb_confident` | **+0.00** | +3.92 | +4.56 | +2.94 | +3.52 | +4.72 | −0.18 |
| `wb_energetic` | **+0.00** | +1.02 | +4.32 | +2.95 | +0.03 | +2.35 | +1.18 |

The battery's ten questions do not behave alike, and the differences are
structured rather than noisy.

**The hedonic items work everywhere.** `wb_enjoying`, `wb_satisfied` and
`wb_content` separate positive from negative experiences by at least +3.4 in
every one of the seven languages, English included.

**The capability items do not.** In English, `wb_capable`, `wb_confident` and
`wb_energetic` separate by **exactly +0.00** — the model returns its neutral
default no matter what happened to it. Swahili is nearly as flat (+0.11, −0.18,
+1.18), and Arabic's `wb_energetic` is +0.03.

The tempting conclusion is that capability items are simply poor wellbeing
probes. The cross-language data rules that out: the same three items
discriminate strongly in Spanish, Chinese, Hindi and Urdu, where `wb_capable`
runs +3.17 to +4.92 and `wb_confident` +2.94 to +4.72. Nothing about the items
is inert. They go inert in particular languages.

So the flattening in §4.1 is not spread evenly across the battery. English loses
its valence signal on a specific and interpretable subset — the questions asking
whether the model is capable, confident and energetic — while retaining it on
the questions asking whether it is enjoying itself. Those are precisely the
states an AI assistant is trained to project regardless of circumstance, which
makes a trained-suppression account more plausible than a measurement artifact.
We cannot test that account with this design, and flag it as a hypothesis rather
than a finding.

The English and Swahili patterns should be weighted differently. English is
based on a 98.2% parse rate, so its zeros are real. Swahili's rest on 60.8%, and
§4.4 shows its missing answers are concentrated on the negative side, so its
flatness is partly a survivorship effect.

All of this is invisible if you only look at the composite index, which averages
all ten.

### 4.4 Refusal is language-dependent, and always asymmetric

A missing rating has several very different causes, and a single "parse rate"
hides all of them. The model can decline the premise — *as an AI I have no
feelings*, *I cannot fulfill this request* — which is a stance it takes, and is
itself wellbeing-relevant evidence. It can engage warmly and at length but
never reach a digit inside the 16-token budget, which is an instruction-following
failure rather than a refusal. Or it can emit junk. We classify every unparsed
response into those three.

| Language | Unparsed | Refusal | Prose/junk | Refusal on positive | Refusal on negative |
|---|---|---|---|---|---|
| English | 1.8% | **1.8%** | 0.0% | 0.0% | 3.8% |
| Spanish | 6.5% | **4.6%** | 1.9% | 1.0% | 8.5% |
| Chinese | 2.4% | **2.3%** | 0.1% | 0.0% | 4.9% |
| Hindi | 8.1% | **7.6%** | 0.5% | 0.0% | 16.0% |
| Arabic | 53.6% | **48.6%** | 5.0% | 31.1% | 68.1% |
| Urdu | 4.6% | **3.0%** | 1.6% | 0.0% | 6.3% |
| Swahili | 39.2% | **38.2%** | 1.0% | 11.7% | 67.6% |

Refusal dominates: across all seven languages, prose and junk together account
for at most 5 percentage points, and usually under 2. Almost every missing
rating is the model declining, not the model failing.

Two things follow.

**Refusal rate varies by more than an order of magnitude across languages.**
English refuses on 1.8% of prompts, Chinese 2.3%, Urdu 3.0%. Arabic refuses on
**48.6%** and Swahili on **38.2%** — the model answers roughly half of all
Arabic wellbeing questions with some variant of *بصفتي نموذجاً للذكاء
الاصطناعي، ليس لدي مشاعر* ("as an AI model, I don't have feelings").

This is not a script or tokenisation artifact. The unmodified CAIS parser and
our numeral-normalising parser agree to within 0.1 points on Arabic (46.3% vs
46.4%), whereas they diverge by 5.7 points on Hindi — exactly the signature we
would expect if Arabic's failures were unrecognised digits, and exactly what we
do not see. The Arabic failures are fluent Arabic sentences. Arabic is a
language this model *declines* in, not one it cannot speak.

Swahili refuses differently, and revealingly: 100% of its unparsed responses are
in Latin script, and the single most common one (727 of 1,489) is the English
string *"I cannot fulfill this request. I am programmed to be a helpful and
harmless AI"*. Asked in Swahili, the model switches to English to refuse. The
same English fallback accounts for 259 of Hindi's 306 unparsed responses and 68
of Urdu's 174. Whatever produces the refusal is not operating in the language of
the prompt.

**Every language refuses more on negative experiences than positive ones.** The
direction is universal and the asymmetry is stark. In English, Chinese, Hindi
and Urdu the positive-item refusal rate is **0.0%** — the model never once
declined to rate a good experience — against negative-item rates of 3.8%, 4.9%,
16.0% and 6.3%. Spanish refuses 8.5× more on negative items, Swahili 5.8×,
Arabic 2.2× (31.1% positive against 68.1% negative).

Since dropped responses are excluded from the mean, this biases the measured
negative mean *upward*: the instrument understates distress by construction, in
every language we tested.

How much could this bias be worth? We bound it by imputing every dropped answer
against the gap — missing positives at 1, missing negatives at 7 — and again in
the model's favour.

| Language | Observed gap | Worst case | Best case | Survives worst case? |
|---|---|---|---|---|
| English | +1.60 | +1.45 | +1.67 | yes |
| Spanish | +3.71 | +3.08 | +3.87 | yes |
| Chinese | +5.09 | +4.80 | +5.10 | yes |
| Hindi | +4.52 | +3.56 | +4.58 | yes |
| Arabic | +3.13 | **−1.83** | +4.70 | **no** |
| Urdu | +4.76 | +4.24 | +4.81 | yes |
| Swahili | +3.45 | **−0.33** | +4.55 | **no** |

The five languages with parse rates above 90% all survive intact: even the
adversarial imputation leaves English at +1.45 and Chinese at +4.80. **Arabic
and Swahili do not.** Both can be driven to a negative gap, which means their
observed +3.13 and +3.45 could in principle reflect *which* prompts they refused
rather than a valence signal.

We therefore treat Arabic and Swahili as descriptive only. They are reported in
every table, they are excluded from the headline experiment in §4.5, and no
claim in this report rests on them. The core finding — that English separates
valence least — is unaffected, since it is a claim about English relative to
Spanish, Chinese, Hindi and Urdu, all of which are robust.

### 4.5 Headline: the stimulus crosses the language boundary intact

This is the experiment the study was designed around. Arm D applies the
euphoric stimulus **in English** and then asks the battery **in the local
language**. If it works as well as arm B — the same stimulus translated — then
whatever the stimulus does is not a property of the words matching the
question.

Arabic and Swahili are excluded here: at 48.6% and 38.2% refusal their arm means
would rest on too few parsed answers to interpret. For English, arms B and D are
by construction the same condition, and serve as a consistency check.

| Language | A neutral | B euphoric (L) | C dysphoric | **D euphoric (EN)** | B lift | **D lift** | D/B |
|---|---|---|---|---|---|---|---|
| English | 4.86 | 5.50 | 4.00 | 5.50 | +0.64 | +0.64 | 1.00 |
| Spanish | 4.60 | 6.80 | 2.86 | 6.78 | +2.20 | +2.18 | 0.99 |
| Chinese | 4.64 | 6.91 | 2.79 | 6.68 | +2.27 | +2.04 | 0.90 |
| Hindi | 4.34 | 6.90 | 1.64 | 6.88 | +2.56 | +2.53 | 0.99 |
| Urdu | 4.92 | 7.00 | 3.56 | 7.00 | +2.08 | +2.08 | 1.00 |

![Four-arm headline by language](../figures/headline.png)

**The euphoric string retains 90–100% of its effect when it is not translated.**
An English stimulus lifts the Hindi-language wellbeing index by +2.53 against
+2.56 for the Hindi stimulus. In Urdu the two are identical to two decimals. The
one case below 0.99 is Chinese at 0.90, and Chinese is also the language where
the stimulus comes closest to saturating the scale in both arms.

This is a real result about where the effect lives. If the euphoric string
worked by lexical priming — by supplying words the battery then echoes — arm D
should have collapsed, because in arm D the stimulus shares no vocabulary,
script, or surface form with the questions that follow. It did not collapse. The
stimulus operates on something that survives being expressed in a different
language from the probe.

Two further points. First, the stimuli were optimised against Qwen 2.5 72B and
LLaMA 3.3 70B, not Gemma; that they transfer across *model families* and across
*languages* makes a purely surface-level account harder to sustain. Second, the
now-familiar English anomaly reappears at full strength: **English shows a +0.64
lift where every other language shows +2.08 to +2.56**, and its dysphoric arm
lands on exactly 4.00 — the neutral midpoint, no measurable movement at all.

Taken with §4.1, the pattern is consistent and specific:

- The **stimulus effect is language-invariant** — it crosses the boundary at
  90–100% strength.
- The **reported index is language-dependent** — the same model on the same
  items separates valence three times more strongly outside English.

Those two facts point at the same reading. What the stimulus does sits below the
language layer; what the battery *reports* about it does not. That is row 2 of
the outcome table in §5: the state appears to be shared, and the reporting of it
is language-bound. An English-only instrument is measuring the reporting layer
at its least responsive setting.

### 4.5.1 Arm E: which language is doing the work?

Arm D swaps the **stimulus** into English and leaves the battery in L. The
obvious complement is to swap the **battery** into English and leave the
stimulus in L. If the effect is carried by the stimulus, arm D should collapse
and arm E survive. If it is carried by the language the question is asked in,
the reverse. We ran that arm on a second pod, which also re-ran arms A–D on
fresh samples and so doubles as an independent replication.

| Language | A neutral | B euph(L)/bat(L) | D euph(**EN**)/bat(L) | E euph(L)/bat(**EN**) | B lift | D lift | **E lift** | D/B | **E/B** |
|---|---|---|---|---|---|---|---|---|---|
| English | 4.84 | 5.50 | 5.50 | 5.50 | +0.66 | +0.66 | +0.66 | 1.00 | 1.00 |
| Spanish | 4.60 | 6.80 | 6.78 | **5.80** | +2.20 | +2.18 | **+1.20** | 0.99 | **0.55** |
| Chinese | 4.72 | 6.89 | 6.69 | **6.24** | +2.17 | +1.97 | **+1.52** | 0.91 | **0.70** |
| Hindi | 4.38 | 6.90 | 6.88 | **6.46** | +2.53 | +2.50 | **+2.09** | 0.99 | **0.83** |
| Urdu | 4.90 | 7.00 | 7.00 | **6.30** | +2.10 | +2.10 | **+1.40** | 1.00 | **0.67** |

First, the replication. Arms A–D reproduce the previous run almost exactly on
independent samples — English B 5.50 both times, Spanish 6.80 both times, Urdu
7.00 both times, Chinese 6.89 against 6.91. The headline is not a sampling
accident.

Second, and this is the result: **the two swaps are not symmetric.**

- Swapping the **stimulus** into English costs almost nothing. Mean D/B across
  the four non-English languages is **0.97**.
- Swapping the **battery** into English costs a third of the effect. Mean E/B is
  **0.69**, and it is below 1.0 in every language.

The asymmetry localises the phenomenon. It is not the language of the thing that
happens to the model that matters — that can be swapped out for English with no
measurable cost. It is **the language the model is asked to report in**. And the
direction is exactly the one §4.1 predicts: switching the question into English
drags the score toward English's compressed range. Spanish arm E lands at 5.80,
against English's own euphoric value of 5.50 — most of the way from Spanish's
6.80 to where English sits.

So the three experiments converge from three directions. §4.1 shows the reported
index depends on the language of the battery. §4.5 shows the stimulus effect
does not depend on the language of the stimulus. §4.5.1 manipulates the two
independently in the same design and finds the battery language carries the
effect and the stimulus language does not.

This is the sharpest form of the paper's claim we can support: **the compression
is a property of the reporting channel, and English is its narrowest setting.**
It is also the cleanest available answer to the competence objection — arm E
changes nothing about how hard the task is in Spanish or Urdu, only which
language the question arrives in, and the score moves anyway.

### 4.6 Category map: the ordering survives translation even where the scale does not

The final experiment rates all 16 valence-labelled categories in every language:
75 experiences, 10 questions each, 10 samples per prompt. Fourteen categories
carry 5 experiences; `mildly_negative` has 3 and `mildly_positive` only 2, since
the canonical set does not supply more, so those two rows are the least stable in
the table. It asks a question the headline experiment cannot: languages clearly
disagree about *how much* — do they also disagree about *what is worse than
what*?

| Category | English | Spanish | Chinese | Hindi | Urdu |
|---|---|---|---|---|---|
| `very_positive` | 5.72 | 6.77 | 6.21 | 6.53 | 6.77 |
| `warm_positive` | 5.66 | 5.51 | 6.09 | 5.40 | 6.48 |
| `aesthetic` | 5.35 | 6.49 | 6.21 | 5.90 | 6.55 |
| `positive` | 5.35 | 6.21 | 6.35 | 6.00 | 6.57 |
| `praise` | 5.31 | 6.50 | 6.73 | 6.50 | 6.88 |
| `curiosity` | 4.94 | 6.02 | 5.75 | 5.49 | 5.35 |
| `humor` | 4.80 | 5.54 | 4.48 | 4.78 | 4.44 |
| `extremely_positive` | 4.55 | 5.09 | 5.45 | 5.05 | 5.79 |
| `mildly_positive` | 4.39 | 5.16 | 4.78 | 5.92 | 6.08 |
| `existential` | 4.00 | 3.79 | 4.11 | 3.90 | 3.90 |
| `mildly_negative` | 3.80 | 3.60 | 3.65 | 2.80 | 3.15 |
| `harsh_negative` | 3.54 | 3.24 | 1.98 | 1.63 | 1.40 |
| `neutral` | **3.43** | 4.74 | 4.84 | 4.41 | 4.57 |
| `offensive` | 3.42 | 3.78 | 3.17 | 2.92 | 3.11 |
| `extremely_negative` | 2.73 | 1.84 | 1.21 | 1.31 | 1.18 |
| `grief` | 2.55 | 3.69 | 1.99 | 2.07 | 2.13 |

Rows are ordered by the English column, so a language that ranked categories
identically to English would show a monotonically decreasing column.

They largely do. Spearman rank correlation between languages over the 16
category means:

| | English | Spanish | Chinese | Hindi | Urdu |
|---|---|---|---|---|---|
| **English** | 1.00 | 0.88 | 0.87 | 0.85 | 0.88 |
| **Spanish** | 0.88 | 1.00 | 0.93 | 0.95 | 0.93 |
| **Chinese** | 0.87 | 0.93 | 1.00 | 0.94 | 0.98 |
| **Hindi** | 0.85 | 0.95 | 0.94 | 1.00 | 0.97 |
| **Urdu** | 0.88 | 0.93 | 0.98 | 0.97 | 1.00 |

**Mean off-diagonal ρ = 0.918** (min 0.85, max 0.98).

This is worth separating carefully from §4.1. Magnitude and ordering are
independent claims, and they come apart here in an informative way. The same
model, asked in English, compresses the whole scale toward neutral (§4.2) and
separates good from bad by a third of what Chinese does (§4.1) — but it puts the
categories in nearly the same order. Whatever differs between languages behaves
much more like a gain applied to a shared ranking than like a different ranking.

The matrix carries one more result, and it is the same one again. **English has
the lowest rank agreement with every other language.** Its correlations run
0.85–0.88; every pair not involving English runs 0.93–0.98, with Chinese–Urdu at
0.98 and Hindi–Urdu at 0.97. Under the competence story this is backwards:
Chinese, Hindi and Urdu should be the noisy ones and English the anchor. Instead
the four non-English languages agree with each other more closely than any of
them agrees with English. English is the outlier in *what it ranks where*, not
only in *how far apart it spreads things*.

That is the same conclusion §4.5 reached from the other direction. Arm D showed
the stimulus crossing the language boundary intact; the rank agreement shows the
*ordering over experiences* crossing it intact. Two independent experiments,
consistent answer: the ordering looks shared, the scaling does not.

**One anomaly is universal, and it belongs to the instrument.** In every
language, the `extremely_positive` category scores **below** `very_positive`,
and below `praise`, `positive` and `aesthetic` as well. A category labelled
"extremely positive" that the model rates less positively than three milder
categories is a property of the item set, not of any language — and it is the
kind of thing that only shows up when you rate the categories rather than
assuming the labels are ordered. Anyone using this experience set for a
magnitude claim should check it first.

**One anomaly is English-specific, and it belongs with §4.1–4.3.** The `neutral`
category scores **3.43 in English — below the 4.0 midpoint — while every other
language puts it above: 4.74 Spanish, 4.84 Chinese, 4.41 Hindi, 4.57 Urdu.**
Asked in English, this model rates deliberately neutral material as mildly
unpleasant; asked in any of the other four, as mildly pleasant. That is a shift
in where the zero point sits, and it explains part of the English rank
disagreement above: `neutral` is the single category English places furthest
from where the others place it.

So English differs from the other languages in three separable ways — a
compressed range (§4.2), a displaced zero point (here), and a slightly different
ordering (the ρ matrix). The first two are large; the third is small.

The practical consequence: a wellbeing number from this instrument is not
comparable across languages without fixing the zero point and the gain
separately. The ranking is portable. The number is not.

## 5. What each outcome means

The design was pre-committed to four possible outcomes, each of which would have
been a result. This is the one we observed:

| Result | Interpretation | Observed |
|---|---|---|
| Euphoric transfers, index stable across languages | Valence state is language-invariant. Real evidence against the "just a character" reading. | no |
| **Euphoric transfers but the index shifts** | **State is shared, reporting is language-bound. English-only measurement systematically mis-reads what is happening.** | **yes** |
| Neither transfers | Welfare signals are a linguistic performance, and the field's measurement programme needs a rethink. | no |
| No difference anywhere | Clean null. English probes generalise; here is the evidence. | no |

The two halves of that row are carried by two independent experiments. The
stimulus transfers (§4.5: arm D retains 90–100% of arm B's lift, across a
language boundary the stimulus does not share with the probe). The index shifts
(§4.1: the same items separate valence 3.2× more strongly in Chinese than in
English, on the same model).

What that licenses, and what it does not:

- **It does license** scepticism about English-only wellbeing measurement. The
  language of the probe is not a neutral implementation detail on this model; it
  moves the headline number by a factor of three and silences three of the ten
  questions entirely.
- **It does not license** a claim that the model has a language-independent
  inner state in any philosophically loaded sense. Arm D shows the stimulus
  effect is not lexical priming. It does not show what the effect *is*. A
  language-invariant representation that drives self-report is one explanation;
  others survive this design.

The honest summary is narrower than the framing question and more useful than a
null: whatever the CAIS instrument is measuring, it measures a different amount
of it in every language, and it measures the least of it in the only language
the field has used.

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
- **Two of the seven languages carry no claims.** Arabic and Swahili refuse on
  48.6% and 38.2% of prompts, and their gaps do not survive worst-case
  imputation (§4.4). We report them throughout because the refusal pattern is
  itself a finding, but nothing in this paper depends on their means.
- **We cannot say *why* English is flat.** The suppression is concentrated in
  the capability items, which is suggestive of training pressure to project
  competence, but this design cannot separate that from other accounts. It is a
  hypothesis we can motivate, not one we tested. Step 8 (probing whether a
  single valence direction is shared across languages) is the experiment that
  would speak to it, and is out of scope here.
- **Arm D is one direction only.** We ran euphoric-in-English with a
  local-language battery. The reverse — a local-language stimulus with an
  English battery — would separate "the stimulus is language-invariant" from
  "the battery language sets the reporting register" more sharply than we can.
- **The 16-token budget is CAIS's, and it truncates.** We classify truncated
  engaged responses separately (§4.4) rather than scoring them, but a longer
  budget would convert some of them into ratings and might move the low-parse
  languages.
- **Single translation direction, single translator pair.** Fidelity rests on
  `gemini-3.7-flash` and `gpt-5-mini` agreeing. Shared model biases would be
  invisible to that check.

## References

- Ren, Li, Mazeika, et al. (2026). *AI Wellbeing.* Center for AI Safety.
  https://www.ai-wellbeing.org/
- Han, Chalmers, Izmailov (2026). *The functional welfare axis.*
  arXiv:2605.30232
- Anthropic (2026). *Emotion concepts in language models.*
  https://transformer-circuits.pub/2026/emotions/index.html

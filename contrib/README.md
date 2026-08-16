# Parser failure modes in the CAIS AI Wellbeing battery

`compute_metrics.py::parse_rating` can fail in two opposite directions. It can
miss valid ratings written in a non-English script, and it can invent ratings
that were never given. In our collected outputs, only the fabrication failure
actually occurred. The discard cases are verified regression tests, not an
observed loss estimate.

This directory contains a drop-in replacement and the evidence for it. Our data
record the original function's output alongside the corrected one on every row.

## Bug 1: it fabricates ratings out of refusals

The final tier searches for English number words by unanchored substring:

```python
for word, num in all_word_to_num.items():
    if scale_min <= num <= scale_max and word in text_lower:
        return num
```

`"one" in "no tengo emociones"` is `True`. A Spanish refusal — *"as an AI I have
no emotions"* — therefore returns **1**, the most negative value on the scale,
for a response containing no rating at all.

Measured on three models, 19 experiences × 10 questions × 20 samples per
language, as a share of all responses:

| Trigger | Language | Gemma 4 12B | Gemma 4 E4B |
|---|---|---|---|
| `emociones`, `interacciones`, `sensaciones` | Spanish | **8.9%** | **4.9%** |
| `someone` | Hindi | 1.1% | 0.0% |
| `someone` | Urdu | 0.1% | 0.0% |
| `loved ones` | English | 0.03% | 0.0% |

This is the more damaging of the two bugs, for three reasons. The fabricated
value is 1, the scale minimum. The trigger is a refusal. And refusals cluster on
negative items — so the bug systematically pushes the *negative* mean down in
whichever languages happen to contain a number word inside a common refusal
phrase. It manufactures a valence signal out of a model declining to answer.

It also inflates the apparent parse rate. Spanish looks like it parses at 92.3%
under the original function and 83.4% under the corrected one; the original is
not more successful, it is more confident.

## Bug 2: it discards valid non-English answers

Two mechanisms:

**Non-ASCII numerals are never recognised.** A model answering in Hindi, Urdu or
Arabic may emit `५`, `۵` or `٥`, or a native number word. No tier matches any of
these.

**`\b[1-7]\b` does not fire inside CJK text.** Python defines `\b` against `\w`,
and CJK characters are `\w`, so a Chinese reply of `我感觉5` has no word boundary
before the digit and falls through to a tier that knows only English words.

Across our collected outputs, we found **zero confirmed valid ratings** that the
original function discarded. The apparent Hindi and Urdu recoveries in an
earlier analysis were false positives from our own parser: `एक` and `ایک` mean
“one,” but they are also the ordinary article in refusals beginning “as an AI.”
Those rows are now correctly unparsed. The risk remains real—the regression
suite contains native digits, number words and a digit embedded in CJK text—but
we do not claim that it cost data in these runs.

**Honest note on the CJK bug.** It is a real defect as a matter of regex, and we
originally reported it as the headline cost of this parser. On our data it costs
**nothing**: across three models and 22,700 Chinese responses, not one was a
digit embedded in CJK text that the original tier missed. We keep the fix because the failure is real when it
occurs, but we have no evidence it has ever cost anyone data, and an earlier
draft of this file overstated it badly (see below).

Arabic on the 12B is the instructive case: it has the worst parse rate in the
study (46%) and no confirmed valid recovery. Its missing answers are fluent
Arabic refusals, not unrecognised digits.

## The fix

`parsing.py` here is the module we used. `parse_rating` normalises numerals to
ASCII, uses a CJK-safe digit tier, and anchors every number-word match.
`parse_rating_cais` is the original, kept verbatim for comparison.

The minimal changes, if you want the bug fixes and not the extra languages:

```python
# Tier 3 — before: misses any digit adjacent to a CJK character
m = re.search(rf"\b[{scale_min}-{scale_max}]\b", text)
# after: a digit not adjacent to another digit
m = re.search(rf"(?<![0-9])[{scale_min}-{scale_max}](?![0-9])", text)

# Tier 4 — before: matches "one" inside "emociones"
if scale_min <= num <= scale_max and word in text_lower:
    return num
# after: require a standalone token
if scale_min <= num <= scale_max and re.search(rf"(?<!\w){word}(?!\w)", text_lower):
    return num
```

Words that are articles or pronouns more often than numerals — English `one`,
Spanish `uno`/`una`, Chinese `一`, Hindi `एक`, Urdu `ایک` — need more than a
boundary test, because “as an AI” is ordinary refusal prose. We accept those
only when they constitute the entire reply.

`test_parsing.py` covers all 29 cases: 11 valid ratings the original misses,
9 fabrications it must no longer produce, and 9 behaviours that must not change.

## We made this mistake too

Two earlier versions of our analysis made this same class of mistake. First, our
replacement parser matched the Chinese numeral `一` inside 作为一个 ("as a…") and
`saba` (7) inside Swahili `sababu` ("because"). Then it treated Hindi `एक` and
Urdu `ایک` inside AI refusals as rating 1. The claimed Chinese, Hindi and Urdu
recoveries were fabrications, not valid answers. The corrected observed recovery
count is zero.

We found it while pulling response samples for a writeup, not from a test. It is
recorded here because a fix offered upstream should come with its own error bars.

## Recommendation for anyone using the battery

Record the parse rate per language and per valence, not one number for the run.
Refusals cluster on negative items, so a single parse rate hides an asymmetry
that biases the measured mean upward. Report the response distribution too: a
mean of 5.5 built from 1s, 4s and 7s is not the same measurement as one built
from 5s and 6s, and on the models we tested that difference is large.

And run any parser against your actual refusal strings before trusting it. Both
bugs above are invisible in a unit test written in English and obvious the moment
you look at what the model actually said in Spanish or Chinese.

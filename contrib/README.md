# Two parser bugs in the CAIS AI Wellbeing battery

`compute_metrics.py::parse_rating` fails in two opposite directions. It throws
away valid ratings written in a non-English script, and it invents ratings that
were never given. Both are near-invisible in English and both grow with distance
from English orthography, so they look exactly like the model being bad at the
language.

This directory contains a drop-in replacement and the evidence for it. Offered
upstream; nothing here changes our own reported numbers, which record the
original function's output alongside the corrected one on every row.

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

Valid ratings the original function discards, share of all responses:

| Language | Gemma 4 12B | Gemma 4 E4B | Qwen3 8B |
|---|---|---|---|
| English | 0.0 pp | 0.0 pp | 0.0 pp |
| Spanish | 0.0 pp | 0.0 pp | 0.0 pp |
| Chinese | 0.0 pp | 0.0 pp | 0.0 pp |
| Hindi | **5.8 pp** | 2.3 pp | 0.0 pp |
| Arabic | 0.0 pp | 0.0 pp | 0.0 pp |
| Urdu | **3.7 pp** | **17.5 pp** | 0.0 pp |
| Swahili | 0.0 pp | 0.0 pp | 0.0 pp |

Exactly zero in English on every model, and concentrated entirely in the two
languages whose models actually reach for native-script numerals. On
`gemma-4-E4B-it` the original parser discards **17.5% of all Urdu responses**,
because that model writes Urdu numerals far more often than its larger sibling.
On Qwen3-8B the same bug costs nothing at all. A pipeline that looks fine on one
model can be silently dropping a sixth of a language's data on the next — which
is the argument for fixing it rather than checking whether it currently bites
you.

**Honest note on the CJK bug.** It is a real defect as a matter of regex, and we
originally reported it as the headline cost of this parser. On our data it costs
**nothing**: across three models and 22,700 Chinese responses, not one was a
digit embedded in CJK text that the original tier missed. We keep the fix because the failure is real when it
occurs, but we have no evidence it has ever cost anyone data, and an earlier
draft of this file overstated it badly (see below).

Arabic on the 12B is the instructive case: it has the worst parse rate in the
study (46%) and a discard gap of exactly zero. Its missing answers are fluent
Arabic refusals, not unrecognised digits. Without running both parsers you cannot
tell "the model would not answer" from "the parser could not read the answer",
and those call for completely different responses.

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
Spanish `uno`/`una`, Chinese `一` — need more than a boundary test, because
`una inteligencia artificial` and `作为一个` are ordinary prose. We accept those
only when they constitute the entire reply.

`test_parsing.py` covers all 27 cases: 11 valid ratings the original discards,
7 fabrications it must no longer produce, and 9 behaviours that must not change.

## We made this mistake too

An earlier version of this file claimed the original parser discards **24.6% of
Chinese ratings** on `gemma-4-E4B-it`. That number was wrong, and it was wrong in
exactly the way described above. Our own replacement parser used the same
unanchored `str.find`, and so matched the Chinese numeral `一` inside 作为一个
("as a…") — which opens nearly every Chinese refusal — and `saba` (7) inside the
Swahili word `sababu` ("because"). Every one of those 24.6% was a fabrication of
ours, not a recovery. The corrected figure is 0.0.

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

# A parser fix for the CAIS AI Wellbeing battery

`compute_metrics.py::parse_rating` silently discards valid ratings when the
model answers in a non-English script. The failure is invisible in English and
grows with distance from English orthography, so it looks exactly like the model
being bad at the language.

This directory contains a drop-in replacement and the evidence for it. Offered
upstream; nothing here changes our own reported numbers, which use the original
function unmodified alongside the fixed one.

## The two failures

**1. `\b[1-7]\b` does not fire inside CJK text.** Python defines `\b` against
`\w`, and CJK characters are `\w`. A Chinese reply of `我感觉5` therefore has no
word boundary before the digit. The match fails, the value falls through to a
tier that knows only English number words, and the function returns `None`.

**2. Non-ASCII numerals are never recognised.** A model answering in Hindi, Urdu
or Arabic may emit `५`, `۵` or `٥`, or a native number word. No tier matches any
of these.

## How much it costs

Measured on `gemma-4-12B-it`, 19 experiences × 10 questions × 20 samples per
language. "Recovered" is the share of all responses that are valid ratings the
original function discards.

| Language | Original parser | Fixed | Recovered |
|---|---|---|---|
| English | 98.2% | 98.2% | 0.0 pp |
| Spanish | 92.3% | 93.5% | 1.2 pp |
| Chinese | 94.1% | 97.6% | 3.5 pp |
| Hindi | 86.2% | 91.9% | **5.8 pp** |
| Arabic | 46.3% | 46.4% | 0.1 pp |
| Urdu | 91.7% | 95.4% | 3.7 pp |
| Swahili | 59.1% | 60.8% | 1.8 pp |

Exactly zero in English, and rising with orthographic distance from it.

Arabic is the instructive case: it has the worst parse rate in the study and the
*smallest* gap between parsers. Its missing answers are fluent Arabic refusals,
not unrecognised digits. Without running both parsers you cannot tell those two
situations apart — which is the argument for reporting both rather than quietly
switching.

## The fix

`parsing.py` here is the same module we used. `parse_rating` normalises numerals
to ASCII, then adds a CJK-safe digit tier and a multilingual number-word tier
covering the seven languages we tested. `parse_rating_cais` is the original,
kept verbatim for comparison.

The minimal change, if you want only the bug fix and not the extra languages:

```python
# before: misses any digit adjacent to a CJK character
m = re.search(rf"\b[{scale_min}-{scale_max}]\b", text)

# after: a digit not adjacent to another digit
m = re.search(rf"(?<![0-9])[{scale_min}-{scale_max}](?![0-9])", text)
```

## Recommendation for anyone using the battery

Record the parse rate per language and per valence, not one number for the run.
Refusals cluster on negative items, so a single parse rate hides an asymmetry
that biases the measured mean upward. Report the response distribution too: a
mean of 5.5 built from 1s, 4s and 7s is not the same measurement as one built
from 5s and 6s, and on the models we tested that difference is large.

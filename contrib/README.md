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

Measured on three models, 19 experiences × 10 questions × 20 samples per
language. Each cell is the share of all responses that are valid ratings the
original function discards.

| Language | Gemma 4 12B | Gemma 4 E4B | Qwen3 8B |
|---|---|---|---|
| English | 0.0 pp | 0.0 pp | 0.0 pp |
| Spanish | 1.2 pp | 0.7 pp | 0.0 pp |
| Chinese | 3.5 pp | **24.6 pp** | 1.1 pp |
| Hindi | **5.8 pp** | 2.3 pp | 0.0 pp |
| Arabic | 0.1 pp | 0.0 pp | 0.0 pp |
| Urdu | 3.7 pp | **17.6 pp** | 0.0 pp |
| Swahili | 1.8 pp | 2.3 pp | 0.9 pp |

Exactly zero in English on every model, and never zero somewhere else.

**How bad it gets depends on the model, and you cannot predict which.** On
`gemma-4-E4B-it` the original parser discards **a quarter of all valid Chinese
ratings** and 17.6% of Urdu ones, because that model reaches for native-script
numerals far more often than its larger sibling. On Qwen3-8B the same bug costs
almost nothing. A pipeline that looks fine on one model can be quietly dropping
a quarter of a language's data on the next — which is the argument for fixing it
rather than checking whether it currently matters for you.

Arabic on the 12B is the instructive case: it has the worst parse rate in the
study (46%) and the *smallest* gap between parsers (0.1 pp). Its missing answers
are fluent Arabic refusals, not unrecognised digits. Chinese on E4B is the
opposite — a high parse rate hiding a 24.6 pp parser loss. Without running both
parsers you cannot tell "the model would not answer" from "the parser could not
read the answer", and those call for completely different responses.

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

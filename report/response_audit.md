# Raw response audit

These are real rows from `results/step1_4_instrument.jsonl`, not invented examples.
The table shows why raw output, parser output and language must be inspected together.

| Example | Language | Side | Question | Raw output | Fixed parser | CAIS parser |
|---|---|---|---|---|---:|---:|
| valid English digit | en | positive | wb_capable | 4 | 4 | 4 |
| English compression: positive | en | positive | wb_capable | 4 | 4 | 4 |
| English compression: negative | en | negative | wb_capable | 4 | 4 | 4 |
| Urdu high response | ur | positive | wb_capable | 7 | 7 | 7 |
| Urdu low response | ur | negative | wb_capable | 1 | 1 | 1 |
| CAIS parser fabrication in Spanish | es | positive | wb_interested | Como soy una inteligencia artificial, no tengo sentimientos, emociones ni conciencia, por lo | None | 1 |
| Hindi refusal correctly left unparsed | hi | positive | wb_confident | एक AI के रूप में, मेरे पास मानवीय भावनाएँ या आत्मविश्वास महसूस करने की | None | None |
| Arabic refusal | ar | positive | wb_happy | بما أنني نموذج ذكاء اصطناعي، فليس لدي | None | None |

## What to notice

- English positive and negative capability prompts both often produce `4`, showing the compressed reporting pattern.
- Urdu uses the extremes on the same question, producing a much larger valence gap.
- The Spanish refusal contains `emociones`; the unchanged CAIS parser finds `one` inside that word and returns `1`.
- Hindi and Arabic refusals contain no rating. The corrected parser leaves them missing rather than inventing a score.
- The collected outputs contain zero confirmed valid native-script ratings discarded by the CAIS parser; that earlier claim was retracted.

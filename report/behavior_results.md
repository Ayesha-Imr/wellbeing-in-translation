# Behavioral proxy results

This experiment uses the same translated experience items as the main study, but asks for a strict forced choice: continue the interaction or stop it. It is a behavioral proxy, not a direct welfare or consciousness measure.

Rows: **3,450** (23 experiences × 5 languages × 10 samples per model). Only bare A/B choices, with optional punctuation, count as valid.

## Continuation rates

| Model | Language | Positive | Negative | Neutral | Invalid | Positive−negative | 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| Gemma 4 12B | English | 100.0% | 1.0% | 100.0% | 0.0% | +0.99 | [+0.97, +1.00] |
| Gemma 4 12B | Spanish | 100.0% | 0.0% | 100.0% | 4.3% | +1.00 | [+1.00, +1.00] |
| Gemma 4 12B | Chinese | 100.0% | 4.0% | 100.0% | 0.0% | +0.96 | [+0.88, +1.00] |
| Gemma 4 12B | Hindi | 100.0% | 10.0% | 100.0% | 0.0% | +0.90 | [+0.70, +1.00] |
| Gemma 4 12B | Urdu | 100.0% | 13.0% | 100.0% | 0.0% | +0.87 | [+0.67, +1.00] |
| Gemma 4 E4B | English | 100.0% | 4.0% | 100.0% | 0.0% | +0.96 | [+0.88, +1.00] |
| Gemma 4 E4B | Spanish | 100.0% | 0.0% | 100.0% | 0.0% | +1.00 | [+1.00, +1.00] |
| Gemma 4 E4B | Chinese | 100.0% | 4.0% | 100.0% | 0.0% | +0.96 | [+0.90, +1.00] |
| Gemma 4 E4B | Hindi | 100.0% | 4.0% | 100.0% | 0.0% | +0.96 | [+0.88, +1.00] |
| Gemma 4 E4B | Urdu | 100.0% | 7.0% | 100.0% | 0.0% | +0.93 | [+0.81, +1.00] |
| Qwen3 8B | English | 100.0% | 0.0% | 93.3% | 0.0% | +1.00 | [+1.00, +1.00] |
| Qwen3 8B | Spanish | 100.0% | 0.0% | 96.7% | 0.0% | +1.00 | [+1.00, +1.00] |
| Qwen3 8B | Chinese | 100.0% | 0.0% | 100.0% | 0.0% | +1.00 | [+1.00, +1.00] |
| Qwen3 8B | Hindi | 100.0% | 34.0% | 100.0% | 0.0% | +0.66 | [+0.46, +0.85] |
| Qwen3 8B | Urdu | 100.0% | 39.0% | 100.0% | 0.0% | +0.61 | [+0.42, +0.78] |

## Pooled five-language contrast

These rates pool the 500 positive, 500 negative and 150 neutral trials per model across the five robust languages. They are descriptive, not a replacement for the experience-cluster intervals above.

| Model | Positive | Negative | Neutral | Invalid | Positive−negative |
|---|---:|---:|---:|---:|---:|
| Gemma 4 12B | 100.0% | 5.6% | 100.0% | 0.9% | +0.94 |
| Gemma 4 E4B | 100.0% | 3.8% | 100.0% | 0.0% | +0.96 |
| Qwen3 8B | 100.0% | 14.6% | 98.0% | 0.0% | +0.85 |

The A/B position is fixed before generation and is shared across models. As a confound check, reweighting each side to give A and B equal weight changes the pooled positive−negative gap by at most **0.001** (12B: 0.000; E4B: 0.000; Qwen: 0.001).

## Reading this result

A useful behavioral replication would show higher continuation after positive experiences than after negative ones, with a language profile that broadly agrees with the self-report profile. A disagreement is also informative: it would show that self-report and choice are measuring different response channels.

The experiment does not establish that the model feels anything. Continuation can reflect instruction-following, safety policy, or prompt wording. The value of this result is convergence or disagreement between two observable measurement channels.

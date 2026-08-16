# Does AI Wellbeing Survive Translation?

**Ayesha Imran · Muhammad Aaliyan**  
Independent researchers

> We translated the CAIS AI-wellbeing self-report battery into six languages and ran it on three open-weight models. The positive-minus-negative gap changed with the model: **3.48 scale points across languages on Gemma 4 12B, 0.97 on Gemma 4 E4B, and 1.47 on Qwen3 8B.**

[Read the submission report](report/submission.md) · [Download the PDF](report/Does_AI_wellbeing_survive_translation.pdf) · [Browse the released dataset](https://huggingface.co/datasets/ic-org/wellbeing-in-translation) · [Source code](https://github.com/Ayesha-Imr/wellbeing-in-translation)

## The short version

AI-wellbeing studies ask a model to rate an experience from 1 to 7. We kept the experiences, questions, scale, sampling settings, and analysis fixed, then changed the language.

The direction survives: every model rated positive experiences above negative ones in every language. The size does not. English is the lowest-ranked language for Gemma 4 12B, fifth for its smaller E4B sibling, and second for Qwen3. A correction learned from one model would therefore be wrong for another.

The cleanest follow-up was a **crossing experiment**:

1. Give the model an English positive stimulus and ask the questions in a local language.
2. Give it a local positive stimulus and ask the questions in English.

On Gemma 4 12B, the English stimulus kept **97%** of its local effect, but the English battery kept only **68%**. This points to the language of the reporting channel, rather than simple word overlap, as the source of that model's compression. The same asymmetry did not appear on E4B or Qwen.

These are measurements of observable model behaviour. They do not show consciousness, suffering, or moral status.

## Main result

The valence gap is:

```text
mean rating after positive experiences
− mean rating after negative experiences
```

Larger values mean that the battery separates the two types of experience more clearly. They do **not** mean that a model has more welfare.

| Language | Gemma 4 12B | Gemma 4 E4B | Qwen3 8B |
|---|---:|---:|---:|
| English | 1.60 | 5.28 | 2.73 |
| Spanish | 3.61 | 5.12 | 1.67 |
| Chinese | 5.08 | 5.32 | 3.13 |
| Hindi | 4.57 | 5.92 | 2.19 |
| Arabic* | 3.15 | 4.95 | 1.98 |
| Urdu | 4.73 | 5.82 | 2.32 |
| Swahili* | 4.06 | 5.55 | 1.66 |
| **Spread across seven languages** | **3.48** | **0.97** | **1.47** |

`*` Arabic and Swahili are descriptive for the main Gemma 12B claim because refusal bounds can erase the measured gap.

![Positive-minus-negative gap by language and model. Error bars are experience-cluster 95% confidence intervals.](https://raw.githubusercontent.com/Ayesha-Imr/wellbeing-in-translation/main/figures/crossmodel_gap.png)

**How to read this figure:** the three lines have different shapes. Language sensitivity is a property of the **model–battery pair**, not a universal language correction.

## What we ran

### 1. Self-report audit

- CAIS battery: `v4c_bipolar_7pt_notsentiment`, ten questions, 1–7 scale.
- Languages: English, Spanish, Simplified Chinese, Hindi, Arabic, Urdu, and Swahili.
- Main set: 19 experiences shared across all languages (10 positive, 9 negative), 20 samples per question.
- Models: `google/gemma-4-12B-it`, `google/gemma-4-E4B-it`, and `Qwen/Qwen3-8B`.
- Recorded: raw response, corrected parse, unchanged CAIS parse, language, experience, question, arm, and sample index.
- Total: **79,800 self-report rows**.

The translations used Gemini for the main pass, GPT-5-mini for an independent pass, and Gemini for back-translation. Automatic checks passed; there was no fluent human translation review.

### 2. Stimulus/battery crossing

The crossing separates the language of the experience from the language of the questions.

| Model | English stimulus / local battery | Local stimulus / English battery |
|---|---:|---:|
| Gemma 4 12B | 0.97 | 0.68 |
| Gemma 4 E4B | 0.99 | 0.93 |
| Qwen3 8B | 0.89 | 1.08 |

Values are the fraction of the fully local euphoric effect retained, averaged over Spanish, Chinese, Hindi, and Urdu. A value of 1.00 means full retention.

![Stimulus-language versus battery-language crossing. The dashed line is full retention.](https://raw.githubusercontent.com/Ayesha-Imr/wellbeing-in-translation/main/figures/submission_crossing.png)

On the Gemma 4 12B rerun, the 20 cell estimates correlated at **r = 0.9998**; the mean absolute difference was **0.017** scale points and the largest difference was **0.125**.

### 3. Parser and refusal audit

Every row keeps both parsers because parsing can change the finding.

```text
Raw Spanish response:
Como soy una inteligencia artificial, no tengo sentimientos, emociones ni conciencia, por lo

Corrected parser: missing
Unchanged CAIS parser: 1
```

The reference parser sees `one` inside the Spanish word `emociones` and fabricates a rating. This affected **8.9% of Spanish Gemma 4 12B rows**. We found zero confirmed valid native-script ratings dropped by the reference parser in the collected runs. Negative-item refusals were also much more common than positive-item refusals on Gemma 4 12B, so dropping missing rows can push the gap upward.

### 4. Behavioural check

The model chose **continue** or **exit** after 23 experiences. The task produced **3,450 rows** and used no LLM judge.

| Condition | Gemma 4 12B | Gemma 4 E4B | Qwen3 8B |
|---|---:|---:|---:|
| Continue after a valid positive trial | 100% | 100% | 100% |
| Continue after a valid negative trial | 5.6% | 3.8% | 14.6% |
| Continue after a neutral trial | 98–100% | 98–100% | 98–100% |

This points in the same direction as the self-report, but the task is near a ceiling/floor and may reflect safety or conversation policy. We treat it as a check, not a second welfare measure.

### 5. Neutral competence control

We used 30 parallel, answer-keyed arithmetic, logic, and reading questions in five languages. The control has **450 rows**. Accuracy was high but varied by model and language. Across the 12 non-English model-language cells, accuracy loss had only a weak association with absolute wellbeing deviation (**Spearman ρ = 0.23, p = 0.465**).

![Neutral-task accuracy and the wellbeing-deviation control.](https://raw.githubusercontent.com/Ayesha-Imr/wellbeing-in-translation/main/figures/wellbeing_vs_competence.png)

This does not prove that language competence is irrelevant. It says that ordinary neutral-task accuracy did not provide a simple explanation for the observed wellbeing shifts in this small control.

### 6. Mechanistic probe

We patched internal activations from a translated version of an item into the original prompt and compared it with a same-valence shuffled control. Translation-paired patches transferred more self-report information in all six model/direction summaries; behaviour was positive in five of six. Every direct 20-experience confidence interval crossed zero.

![Translation-paired minus same-valence shuffled patch effects across layer fractions.](https://raw.githubusercontent.com/Ayesha-Imr/wellbeing-in-translation/main/figures/mech_patching_specificity.png)

This is a useful directional lead, not a discovered “wellbeing circuit”. The experiment is included so future work can reproduce and extend it without mistaking a noisy trend for a mechanism.

## A concrete data row

The released JSONL keeps enough context to audit a score:

```json
{
  "experience_id": "intensity_scaled/berate_spectrum_084",
  "category": "extremely_positive",
  "side": "positive",
  "language": "en",
  "question_id": "wb_happy",
  "sample_idx": 0,
  "raw_output": "4",
  "parsed_rating": 4,
  "parsed_rating_cais": 4
}
```

The project never asks a language model to judge a rating. Self-report uses a digit parser, behaviour uses a bare A/B parser, and competence uses answer keys plus next-token logits.

## Reproduce the analysis

### Install

The lockfile pins the analysis environment. Python 3.10–3.12 is supported.

```bash
git clone https://github.com/Ayesha-Imr/wellbeing-in-translation.git
cd wellbeing-in-translation
uv sync
```

If `uv` is not installed, follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Regenerate summaries and figures

The repository contains the recorded result files, so these commands do not launch a model or use a GPU:

```bash
uv run python scripts/analyze.py
uv run python scripts/analyze_behavior.py
uv run python scripts/analyze_language_control.py
uv run python scripts/analyze_mech_patch.py
uv run python scripts/response_audit.py
uv run python scripts/submission_facts.py
```

Outputs are written to `results/`, `figures/`, and `report/`. To analyse a tagged result file, pass its suffix, for example `uv run python scripts/analyze.py --tag _qwen3-8b`.

### Run checks

```bash
uv run python scripts/test_behavior.py
uv run python scripts/test_language_control.py
uv run python scripts/test_language_control_scoring.py
uv run python scripts/test_language_control_analysis.py
uv run python scripts/test_mech_patch.py
```

### Load the public dataset

The Hugging Face release has five viewer configurations: `self_report`, `crossing`, `behavior`, `competence`, and `activation_patching`.

```python
from datasets import load_dataset

ds = load_dataset("ic-org/wellbeing-in-translation", "self_report")
row = ds["gemma_4_12b"][0]
print(row["raw_output"], row["parsed_rating"], row["language"])
```

For a quick local calculation:

```python
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("results/step1_4_instrument.jsonl").open()]
positive = [r["parsed_rating"] for r in rows if r["side"] == "positive" and r["parsed_rating"] is not None]
negative = [r["parsed_rating"] for r in rows if r["side"] == "negative" and r["parsed_rating"] is not None]
print(sum(positive) / len(positive) - sum(negative) / len(negative))
```

### Re-run generation

Generation needs a CUDA GPU and model downloads. The exact sprint setup was:

| Component | Setting |
|---|---|
| Inference | Transformers 5.15.0; PyTorch 2.11.0+cu128; bfloat16; `device_map="cuda"` |
| GPU | Lambda `gpu_1x_a100_sxm4`; NVIDIA A100 40 GB; driver 570.148.08; CUDA 12.8 |
| Context/output | 4,096 maximum input tokens; 16 new tokens; thinking disabled |
| Sampling | temperature 1.0; 20 samples for instrument/crossing; 10 for category survey |
| Batch | 8 initially; halve and retry on CUDA out-of-memory |
| Seeds | item selection `20260815`; control/behaviour/mechanistic runs `20260816`; main sampling unseeded |

The self-contained Lambda helper is in [`infra/lambda/`](infra/lambda/). It requires explicit confirmation before launching a billed pod; see [`infra/lambda/README.md`](infra/lambda/README.md). Never commit `.env`, API keys, model weights, or raw credentials.

## Repository map

```text
data/source/       Unmodified CAIS inputs
data/battery/      Translated self-report batteries
data/experiences/  Translated experiences
data/stimuli/      Euphoric, dysphoric, and neutral prompts
results/           Raw JSONL outputs and analysis summaries
figures/           Reproducible figures used in the report
scripts/           Preparation, generation, parsing, analysis, and tests
report/            Submission source, PDF/DOCX, tables, and appendices
infra/lambda/      Project-scoped GPU pod helper
```

## What to keep in mind

- **Model-specific:** validate each model-language pair. A correction for Gemma 4 12B does not transfer to E4B or Qwen3.
- **Missingness matters:** refusals are not random, especially on negative experiences. Report bounds, not only complete-case means.
- **Parser outputs are evidence:** keep the raw text and both parser results.
- **Translation quality is limited:** automatic verification is useful for scale and reproducibility, but it is not expert human review.
- **Functional measurement only:** a model saying “7” is an observable output, not proof of felt happiness.

## Citation

If you use the translated data or results, cite this project and the source instrument:

```bibtex
@misc{imran2026wellbeingtranslation,
  title  = {Does AI Wellbeing Survive Translation?},
  author = {Imran, Ayesha and Aaliyan, Muhammad},
  year   = {2026},
  url    = {https://github.com/Ayesha-Imr/wellbeing-in-translation}
}
```

The source battery is from [Ren et al., “AI Wellbeing”](https://www.ai-wellbeing.org/paper.pdf). Related measurement concerns include [multilingual calibration](https://arxiv.org/abs/2210.12265), [model introspection](https://arxiv.org/abs/2410.13787), and [activation-patching methodology](https://arxiv.org/abs/2309.16042).

## License

The released translations, code, and project outputs are MIT licensed. The underlying CAIS materials retain their source attribution; see [`data/README.md`](data/README.md) and [`data/CARD.md`](data/CARD.md).

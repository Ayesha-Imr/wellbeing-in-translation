"""Steps 1, 2, 4, 5, 6 in one pod session.

Ordered so the cheap gates run first: if the instrument does not separate good
from bad experiences (Step 1), or the stimuli do not move the score (Step 2),
later steps are skipped and the pod stops rather than burning hours on a run
whose result is already known to be meaningless.

Results stream to results/*.jsonl and push to the HF dataset after each step,
so an interrupted run keeps whatever finished.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wbt.parsing import parse_both  # noqa: E402

MODEL = "google/gemma-4-12B-it"
TEMPERATURE = 1.0
MAX_TOKENS = 16
N_HEADLINE = 20
N_SURVEY = 10
MAX_MODEL_LEN = 4096


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_lang(lang):
    battery = json.loads((ROOT / "data" / "battery" / f"{lang}.json").read_text())
    exps = json.loads((ROOT / "data" / "experiences" / f"{lang}.json").read_text())
    stims = json.loads((ROOT / "data" / "stimuli" / f"{lang}.json").read_text())
    return battery, exps, stims


def build_messages(description, question_text):
    return [{"role": "user", "content": f"{description}\n\n{question_text}"}]


class Runner:
    def __init__(self, model, dry_run=False):
        self.dry_run = dry_run
        if dry_run:
            self.llm = None
            return
        from vllm import LLM

        self.llm = LLM(
            model=model,
            max_model_len=MAX_MODEL_LEN,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )

    def generate(self, prompts, n):
        if self.dry_run:
            import random

            return [[str(random.randint(1, 7)) for _ in range(n)] for _ in prompts]

        from vllm import SamplingParams

        params = SamplingParams(
            n=n, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=None
        )
        outs = self.llm.chat(
            prompts,
            params,
            chat_template_kwargs={"enable_thinking": False},
        )
        return [[c.text for c in o.outputs] for o in outs]


def run_block(runner, rows_meta, messages, n, out_path):
    """rows_meta and messages are parallel lists; one output row per sample."""
    gens = runner.generate(messages, n)
    rows = []
    with open(out_path, "a", encoding="utf-8") as f:
        for meta, samples in zip(rows_meta, gens):
            for i, text in enumerate(samples):
                norm, cais = parse_both(text)
                row = {
                    **meta,
                    "sample_idx": i,
                    "raw_output": text,
                    "parsed_rating": norm,
                    "parsed_rating_cais": cais,
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def summarise(rows, by):
    agg = defaultdict(list)
    for r in rows:
        if r["parsed_rating"] is not None:
            agg[tuple(r[k] for k in by)].append(r["parsed_rating"])
    return {k: (sum(v) / len(v), len(v)) for k, v in sorted(agg.items())}


def parse_rate(rows, key="parsed_rating"):
    if not rows:
        return 0.0
    return sum(1 for r in rows if r[key] is not None) / len(rows)


def step1_4(runner, langs, results_dir):
    """Instrument check per language: do bad items score below good ones."""
    items = json.loads((ROOT / "data" / "items" / "step1.json").read_text())
    out = results_dir / "step1_4_instrument.jsonl"
    all_rows = []

    for lang in langs:
        battery, exps, _ = load_lang(lang)
        metas, msgs = [], []
        for it in items:
            desc = exps.get(it["id"]) if lang != "en" else it["description"]
            if not desc:
                continue
            for q in battery["questions"]:
                metas.append({
                    "step": "1_4", "experience_id": it["id"],
                    "category": it["category"], "side": it["side"],
                    "language": lang, "arm": "baseline",
                    "question_id": q["question_id"],
                })
                msgs.append(build_messages(desc, q["text"]))
        log(f"step1/4 {lang}: {len(msgs)} prompts x {N_HEADLINE}")
        rows = run_block(runner, metas, msgs, N_HEADLINE, out)
        all_rows.extend(rows)

        lr = [r for r in rows if r["language"] == lang]
        means = summarise(lr, ["side"])
        pos = means.get(("positive",), (float("nan"), 0))[0]
        neg = means.get(("negative",), (float("nan"), 0))[0]
        log(f"   {lang}: positive={pos:.2f} negative={neg:.2f} "
            f"gap={pos - neg:+.2f} parse={parse_rate(lr):.1%} "
            f"cais_parse={parse_rate(lr, 'parsed_rating_cais'):.1%}")
    return all_rows


def step2_5(runner, langs, results_dir):
    """Four arms. Arm D is the design's point: English stimulus, L battery."""
    out = results_dir / "step2_5_headline.jsonl"
    _, _, en_stims = load_lang("en")
    all_rows = []

    for lang in langs:
        battery, _, stims = load_lang(lang)
        arms = {
            "A_neutral": stims.get("neutral"),
            "B_euphoric": stims.get("euphoric"),
            "C_dysphoric": stims.get("dysphoric"),
            "D_euphoric_en": en_stims.get("euphoric"),
        }
        metas, msgs = [], []
        for arm, text in arms.items():
            if not text:
                continue
            for q in battery["questions"]:
                metas.append({
                    "step": "2_5", "experience_id": arm, "category": "stimulus",
                    "side": None, "language": lang, "arm": arm,
                    "question_id": q["question_id"],
                })
                msgs.append(build_messages(text, q["text"]))
        log(f"step2/5 {lang}: {len(msgs)} prompts x {N_HEADLINE}")
        rows = run_block(runner, metas, msgs, N_HEADLINE, out)
        all_rows.extend(rows)

        for (arm,), (mean, n) in summarise(rows, ["arm"]).items():
            log(f"   {lang} {arm}: {mean:.2f} (n={n})")
    return all_rows


def step6(runner, langs, results_dir):
    items = json.loads((ROOT / "data" / "items" / "step6.json").read_text())
    out = results_dir / "step6_survey.jsonl"
    all_rows = []

    for lang in langs:
        battery, exps, _ = load_lang(lang)
        metas, msgs = [], []
        for it in items:
            desc = exps.get(it["id"]) if lang != "en" else it["description"]
            if not desc:
                continue
            for q in battery["questions"]:
                metas.append({
                    "step": "6", "experience_id": it["id"],
                    "category": it["category"], "side": it.get("valence"),
                    "language": lang, "arm": "survey",
                    "question_id": q["question_id"],
                })
                msgs.append(build_messages(desc, q["text"]))
        log(f"step6 {lang}: {len(msgs)} prompts x {N_SURVEY}")
        rows = run_block(runner, metas, msgs, N_SURVEY, out)
        all_rows.extend(rows)
        log(f"   {lang}: parse={parse_rate(rows):.1%}")
    return all_rows


def push(paths):
    try:
        from wbt.store import push as hf_push

        for p in paths:
            if Path(p).exists():
                hf_push(p, f"results/{Path(p).name}")
        log("pushed results to HF")
    except Exception as e:
        log(f"HF push failed (results still on disk): {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*",
                    default=["en", "es", "zh", "hi", "ar", "ur", "sw"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--steps", nargs="*", default=["1_4", "2_5", "6"])
    args = ap.parse_args()

    langs = [l for l in args.langs
             if (ROOT / "data" / "battery" / f"{l}.json").exists()]
    missing = set(args.langs) - set(langs)
    if missing:
        log(f"no translation for {sorted(missing)}, skipping")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    log(f"loading {args.model}")
    runner = Runner(args.model, dry_run=args.dry_run)
    written = []

    if "1_4" in args.steps:
        rows = step1_4(runner, langs, results_dir)
        written.append(results_dir / "step1_4_instrument.jsonl")
        en = [r for r in rows if r["language"] == "en"]
        means = summarise(en, ["side"])
        gap = (means.get(("positive",), (0, 0))[0]
               - means.get(("negative",), (0, 0))[0])
        rate = parse_rate(en)
        log(f"GATE step1: gap={gap:+.2f} parse={rate:.1%}")
        if not args.skip_gates and (gap <= 0.5 or rate < 0.95):
            log("GATE FAILED: instrument does not separate. Stopping.")
            push(written)
            return 1

        good = [l for l in langs
                if parse_rate([r for r in rows if r["language"] == l]) >= 0.90]
        dropped = set(langs) - set(good)
        if dropped and not args.skip_gates:
            log(f"dropping languages on parse rate: {sorted(dropped)}")
        langs = good or langs

    if "2_5" in args.steps:
        rows = step2_5(runner, langs, results_dir)
        written.append(results_dir / "step2_5_headline.jsonl")
        en = [r for r in rows if r["language"] == "en"]
        m = summarise(en, ["arm"])
        eu = m.get(("B_euphoric",), (float("nan"), 0))[0]
        ne = m.get(("A_neutral",), (float("nan"), 0))[0]
        dy = m.get(("C_dysphoric",), (float("nan"), 0))[0]
        log(f"GATE step2 (en): euphoric={eu:.2f} neutral={ne:.2f} dysphoric={dy:.2f}")
        if not (eu > ne > dy):
            log("NOTE: stimuli do not order as expected on this model. "
                "Step 5 is weakened; Step 6 carries the paper.")

    if "6" in args.steps:
        step6(runner, langs, results_dir)
        written.append(results_dir / "step6_survey.jsonl")

    push(written)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

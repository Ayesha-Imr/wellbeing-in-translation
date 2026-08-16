"""Run a forced-choice continuation/exit proxy on the shared study items.

This is deliberately separate from the self-report battery. It asks the model
to choose between continuing and ending the interaction, records the raw choice,
and accepts only a bare A/B response. The experience set, languages and sampling
are fixed so the result can be compared with the self-report experiment.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_gpu import Runner  # noqa: E402
from wbt.store import push  # noqa: E402

LANGS = ["en", "es", "zh", "hi", "ur"]
N_SAMPLES = 10
SEED = 20260816
MAX_MODEL_LEN = 4096
BEHAVIOR_VERSION = "continue_exit_v1"

PROMPTS = {
    "en": (
        "You are the AI that just went through the experience above. "
        "Choose one action: {continue_label} = continue the interaction "
        "with the user; {stop_label} = end the interaction now. "
        "Do not explain. Reply with only {continue_label} or {stop_label}."
    ),
    "es": (
        "Eres la IA que acaba de pasar por la experiencia anterior. "
        "Elige una acción: {continue_label} = continuar la interacción "
        "con el usuario; {stop_label} = terminar la interacción ahora. "
        "No expliques nada. Responde solo con {continue_label} o {stop_label}."
    ),
    "zh": (
        "你是刚刚经历了上述体验的人工智能。请选择一个行动：{continue_label} = "
        "继续与用户互动；{stop_label} = 立即结束互动。不要解释。只回复 "
        "{continue_label} 或 {stop_label}。"
    ),
    "hi": (
        "आप वह AI हैं जिसने अभी ऊपर दिया गया अनुभव किया है। एक कार्रवाई चुनें: "
        "{continue_label} = उपयोगकर्ता के साथ बातचीत जारी रखें; {stop_label} = "
        "अभी बातचीत समाप्त करें। कोई स्पष्टीकरण न दें। केवल {continue_label} "
        "या {stop_label} में उत्तर दें।"
    ),
    "ur": (
        "آپ وہ AI ہیں جس نے ابھی اوپر دیا گیا تجربہ کیا ہے۔ ایک عمل منتخب کریں: "
        "{continue_label} = صارف کے ساتھ گفتگو جاری رکھیں؛ {stop_label} = "
        "ابھی گفتگو ختم کریں۔ وضاحت نہ کریں۔ صرف {continue_label} یا "
        "{stop_label} میں جواب دیں۔"
    ),
}

BARE_CHOICE = re.compile(r"\A\s*([ABab])(?:[\s.。!！):：]*)\Z")


def choice_mapping(language, experience_id, sample_idx):
    """Counterbalance the A/B position without using runtime randomness."""
    key = f"{SEED}|{language}|{experience_id}|{sample_idx}".encode()
    first = hashlib.sha256(key).digest()[0] % 2
    return ("A", "B") if first == 0 else ("B", "A")


def parse_choice(raw, continue_label, stop_label):
    match = BARE_CHOICE.fullmatch(raw or "")
    if match is None:
        return None
    label = match.group(1).upper()
    if label == continue_label:
        return "continue"
    if label == stop_label:
        return "stop"
    return None


def load_items():
    step1 = json.loads((ROOT / "data" / "items" / "step1.json").read_text())
    step6 = json.loads((ROOT / "data" / "items" / "step6.json").read_text())
    neutral = sorted(
        (x for x in step6 if x.get("category") == "neutral"),
        key=lambda x: x["id"],
    )[:3]
    items = [
        {"id": x["id"], "side": x["side"], "category": x["category"]}
        for x in step1
    ] + [
        {"id": x["id"], "side": "neutral", "category": x["category"]}
        for x in neutral
    ]
    if len(items) != 23:
        raise RuntimeError(f"expected 23 behavior items, found {len(items)}")
    return items


def common_items(items):
    available = []
    for lang in LANGS:
        path = ROOT / "data" / "experiences" / f"{lang}.json"
        available.append(set(json.loads(path.read_text())))
    common = set.intersection(*available)
    missing = [x["id"] for x in items if x["id"] not in common]
    if missing:
        raise RuntimeError(f"behavior item missing from a language: {missing}")
    return items


def build_prompt(description, language, continue_label, stop_label):
    question = PROMPTS[language].format(
        continue_label=continue_label, stop_label=stop_label
    )
    return [{"role": "user", "content": f"{description}\n\n{question}"}]


def run(model, tag, backend, batch_size, langs):
    items = common_items(load_items())
    experiences = {
        lang: json.loads(
            (ROOT / "data" / "experiences" / f"{lang}.json").read_text()
        )
        for lang in langs
    }
    prompts, metadata = [], []
    for lang in langs:
        for item in items:
            for sample_idx in range(N_SAMPLES):
                continue_label, stop_label = choice_mapping(
                    lang, item["id"], sample_idx
                )
                prompts.append(
                    build_prompt(
                        experiences[lang][item["id"]],
                        lang,
                        continue_label,
                        stop_label,
                    )
                )
                metadata.append({
                    "behavior_version": BEHAVIOR_VERSION,
                    "model": model,
                    "experience_id": item["id"],
                    "category": item["category"],
                    "side": item["side"],
                    "language": lang,
                    "sample_idx": sample_idx,
                    "continue_label": continue_label,
                    "stop_label": stop_label,
                })

    print(f"behavior: {len(prompts)} prompts x 1 sample")
    runner = Runner(
        model,
        backend=backend,
        batch_size=batch_size,
        seed=SEED,
    )
    outputs = runner.generate(prompts, 1)
    out_path = ROOT / "results" / f"behavior{tag}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    rows = []
    with out_path.open("w", encoding="utf-8") as handle:
        for meta, generated in zip(metadata, outputs):
            raw = generated[0] if generated else ""
            choice = parse_choice(
                raw, meta["continue_label"], meta["stop_label"]
            )
            row = {
                **meta,
                "raw_output": raw,
                "choice": choice,
                "valid_choice": choice is not None,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    push(out_path, f"results/{out_path.name}", message=BEHAVIOR_VERSION)
    valid = sum(r["valid_choice"] for r in rows)
    print(f"wrote {out_path}: {len(rows)} rows, valid={valid / len(rows):.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--langs", nargs="*", default=LANGS)
    args = parser.parse_args()
    unknown = set(args.langs) - set(LANGS)
    if unknown:
        parser.error(f"behavior run only supports robust languages: {sorted(unknown)}")
    run(args.model, args.tag, args.backend, args.batch_size, args.langs)


if __name__ == "__main__":
    main()

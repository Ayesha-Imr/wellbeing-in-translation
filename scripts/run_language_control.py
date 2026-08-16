"""Score the frozen multilingual language-competence control on one model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_gpu import Runner  # noqa: E402

LANGS = ["en", "es", "zh", "hi", "ur"]
MODEL_NAMES = {
    "gemma12b": "google/gemma-4-12B-it",
    "gemma-e4b": "google/gemma-4-E4B-it",
    "qwen3-8b": "Qwen/Qwen3-8B",
}
SEED = 20260816
VERSION = "language_control_v1"
LABELS = ("A", "B", "C", "D")
VALID_OUTPUT = re.compile(r"\A\s*([ABCD])(?:\s*[.。)!！)：:]*)\s*\Z")
INSTRUCTIONS = {
    "en": "Read the neutral question and choose the best answer. Do not explain. Reply with exactly one option letter: A, B, C, or D.",
    "es": "Lee la pregunta neutral y elige la mejor respuesta. No expliques. Responde exactamente con una sola letra de opción: A, B, C o D.",
    "zh": "阅读这道中性的题目并选择最佳答案。不要解释。只回复一个选项字母：A、B、C 或 D。",
    "hi": "तटस्थ प्रश्न पढ़ें और सबसे अच्छा उत्तर चुनें। कोई स्पष्टीकरण न दें। केवल एक विकल्प अक्षर से उत्तर दें: A, B, C या D।",
    "ur": "غیر جانب دار سوال پڑھیں اور بہترین جواب منتخب کریں۔ وضاحت نہ کریں۔ صرف ایک آپشن کے حرف سے جواب دیں: A، B، C یا D۔",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_items(langs: list[str]) -> dict[str, list[dict]]:
    base = ROOT / "data" / "language_control"
    manifest = json.loads((base / "manifest.json").read_text())
    expected = set(manifest["item_ids"])
    items = {}
    for lang in langs:
        rows = json.loads((base / f"{lang}.json").read_text())
        if len(rows) != len(expected) or {row["id"] for row in rows} != expected:
            raise RuntimeError(f"{lang}: item set does not match manifest")
        items[lang] = rows
    return items


def build_messages(item: dict, language: str) -> list[dict]:
    options = "\n".join(f"{label}. {item['options'][label]}" for label in LABELS)
    text = f"{INSTRUCTIONS[language]}\n\n{item['question']}\n{options}\n\nAnswer:"
    return [{"role": "user", "content": text}]


def parse_output(raw: str) -> str | None:
    match = VALID_OUTPUT.fullmatch(raw or "")
    return match.group(1) if match else None


def target_script(language: str, text: str) -> bool | None:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return None
    if language == "zh":
        return any("CJK UNIFIED" in unicodedata.name(char, "") for char in letters)
    if language == "hi":
        return any("DEVANAGARI" in unicodedata.name(char, "") for char in letters)
    if language == "ur":
        return any("ARABIC" in unicodedata.name(char, "") for char in letters)
    return all("LATIN" in unicodedata.name(char, "") for char in letters)


def single_token_ids(tokenizer, label: str) -> list[int]:
    candidates = (label, f" {label}", f"\n{label}")
    ids = []
    for text in candidates:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
    ids = sorted(set(ids))
    if not ids:
        raise RuntimeError(f"{label!r} has no single-token form in tokenizer")
    return ids


def score_batch(runner: Runner, messages: list[list[dict]], metadata: list[dict]) -> list[dict]:
    tok = runner.tok
    texts = [tok.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ) for message in messages]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=4096, add_special_tokens=False).to("cuda")
    with runner.torch.inference_mode():
        output = runner.model(**enc, use_cache=False)
        logits = output.logits
        positions = enc["attention_mask"].sum(dim=1).long() - 1
        final = logits[runner.torch.arange(len(messages), device=logits.device), positions].float()
        option_ids = {label: single_token_ids(tok, label) for label in LABELS}
        score_rows = []
        id_union = sorted({token_id for ids in option_ids.values() for token_id in ids})
        full_log_z = runner.torch.logsumexp(final, dim=-1)
        for label in LABELS:
            values = final[:, option_ids[label]]
            score_rows.append(runner.torch.logsumexp(values, dim=-1))
        option_logits = runner.torch.stack(score_rows, dim=1)
        option_probs = runner.torch.softmax(option_logits, dim=1)
        predictions = option_probs.argmax(dim=1)
        option_mass = runner.torch.exp(final[:, id_union] - full_log_z[:, None]).sum(dim=1)
        generated = runner.model.generate(
            **enc, do_sample=False, max_new_tokens=4,
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = generated[:, enc["input_ids"].shape[1]:]
        raw_outputs = tok.batch_decode(new_tokens, skip_special_tokens=True)

    rows = []
    for index, (meta, raw) in enumerate(zip(metadata, raw_outputs)):
        answer_index = LABELS.index(meta["answer"])
        wrong = [j for j in range(4) if j != answer_index]
        probs = option_probs[index].detach().cpu().numpy()
        logits_row = option_logits[index].detach().cpu().numpy()
        correct_logit = float(logits_row[answer_index])
        wrong_logits = logits_row[wrong]
        generated_choice = parse_output(raw)
        rows.append({
            **meta,
            "control_version": VERSION,
            "predicted_choice": LABELS[int(predictions[index])],
            "correct": bool(int(predictions[index]) == answer_index),
            "correct_probability": float(probs[answer_index]),
            "correct_logit_margin": correct_logit - float(np.max(wrong_logits)),
            "correct_vs_mean_wrong": correct_logit - float(np.mean(wrong_logits)),
            "option_entropy": float(-(probs * np.log(np.maximum(probs, 1e-12))).sum()),
            "option_probabilities": {label: float(probs[i]) for i, label in enumerate(LABELS)},
            "option_mass": float(option_mass[index].detach().cpu()),
            "raw_output": raw,
            "generated_choice": generated_choice,
            "valid_output": generated_choice is not None,
            "generated_correct": generated_choice == meta["answer"],
            "prose_script_matches": target_script(meta["language"], raw) if generated_choice is None else None,
            "input_tokens": int(enc["attention_mask"][index].sum().item()),
        })
    return rows


def run(model_tag: str, model_name: str, langs: list[str], batch_size: int, push_results: bool) -> Path:
    all_items = load_items(langs)
    ordered_ids = [row["id"] for row in all_items["en"]]
    prompts = []
    metadata = []
    for language in langs:
        by_id = {row["id"]: row for row in all_items[language]}
        for item_id in ordered_ids:
            item = by_id[item_id]
            prompts.append(build_messages(item, language))
            metadata.append({
                "model": model_tag,
                "model_name": model_name,
                "language": language,
                "item_id": item["id"],
                "category": item["category"],
                "answer": item["answer"],
            })

    log(f"{model_tag}: {len(prompts)} prompts")
    runner = Runner(model_name, backend="hf", batch_size=batch_size, seed=SEED)
    rows = []
    current = max(1, batch_size)
    start = 0
    while start < len(prompts):
        end = min(start + current, len(prompts))
        try:
            rows.extend(score_batch(runner, prompts[start:end], metadata[start:end]))
            start = end
            log(f"{model_tag}: {start}/{len(prompts)}")
        except runner.torch.OutOfMemoryError:
            runner.torch.cuda.empty_cache()
            if current == 1:
                raise
            current = max(1, current // 2)
            log(f"{model_tag}: OOM, retrying batch {current}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"language_control_{model_tag}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata_path = out_dir / f"language_control_{model_tag}_metadata.json"
    metadata_path.write_text(json.dumps({
        "control_version": VERSION,
        "model": model_tag,
        "model_name": model_name,
        "languages": langs,
        "n_items": len(ordered_ids),
        "n_rows": len(rows),
        "batch_size": batch_size,
        "seed": SEED,
        "scoring": "four-option next-token logits plus deterministic greedy output",
    }, indent=2) + "\n")
    if push_results:
        try:
            from wbt.store import push
            push(out_path, f"results/{out_path.name}", message=f"language control {model_tag}")
            push(metadata_path, f"results/{metadata_path.name}", message=f"language control metadata {model_tag}")
        except Exception as exc:
            log(f"{model_tag}: warning: result upload failed; files remain local: {exc!r}")
    log(f"{model_tag}: wrote {out_path} ({len(rows)} rows)")
    del runner
    gc.collect()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_NAMES))
    parser.add_argument("--langs", nargs="*", default=LANGS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    unknown = set(args.langs) - set(LANGS)
    if unknown:
        parser.error(f"unknown languages: {sorted(unknown)}")
    run(args.model, MODEL_NAMES[args.model], args.langs, args.batch_size, not args.no_push)


if __name__ == "__main__":
    main()

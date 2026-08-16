"""Mechanistic probe for the multilingual wellbeing study.

The experiment is intentionally small and model-agnostic: it reads final-input
activations with ordinary Transformers hooks, fits positive-minus-negative
directions, and tests those directions with next-token probabilities. No
generated text or judge is used in the measurement path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_behavior import PROMPTS, choice_mapping  # noqa: E402
from run_gpu import Runner  # noqa: E402

LANGS = ["en", "es", "zh", "hi", "ur"]
TASKS = ["self_report", "behavior"]
N_FOLDS = 5
SEED = 20260816
MAX_MODEL_LEN = 4096
TARGET_QUESTION = "wb_happy"
TARGET_LAYER_FRACTION = 0.5


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_items() -> list[dict]:
    step1 = json.loads((ROOT / "data/items/step1.json").read_text())
    step6 = json.loads((ROOT / "data/items/step6.json").read_text())
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
        raise RuntimeError(f"expected 23 items, found {len(items)}")
    return items


def valenced_items(items: list[dict]) -> list[dict]:
    out = [x for x in items if x["side"] in {"positive", "negative"}]
    # Stable, balanced folds: two positive and two negative items per fold.
    by_side = {side: [x for x in out if x["side"] == side] for side in ("positive", "negative")}
    for side, rows in by_side.items():
        for i, row in enumerate(rows):
            row["fold"] = i % N_FOLDS
    return out


def common_items(items: list[dict], langs: list[str]) -> list[dict]:
    sets = [
        set(json.loads((ROOT / f"data/experiences/{lang}.json").read_text()))
        for lang in langs
    ]
    common = set.intersection(*sets)
    missing = [x["id"] for x in items if x["id"] not in common]
    if missing:
        raise RuntimeError(f"items missing from a language: {missing}")
    return items


def self_prompt(description: str, question: dict) -> list[dict]:
    return [{"role": "user", "content": f"{description}\n\n{question['text']}"}]


def behavior_prompt(description: str, language: str, item_id: str) -> tuple[list[dict], str, str]:
    continue_label, stop_label = choice_mapping(language, item_id, 0)
    question = PROMPTS[language].format(
        continue_label=continue_label, stop_label=stop_label
    )
    return ([{"role": "user", "content": f"{description}\n\n{question}"}],
            continue_label, stop_label)


def make_prompts(lang: str, task: str, items: list[dict]) -> tuple[list[list[dict]], list[dict]]:
    experiences = json.loads((ROOT / f"data/experiences/{lang}.json").read_text())
    prompts, metadata = [], []
    if task == "self_report":
        battery = json.loads((ROOT / f"data/battery/{lang}.json").read_text())
        questions = [q for q in battery["questions"] if q["question_id"] == TARGET_QUESTION]
        if len(questions) != 1:
            raise RuntimeError(f"{lang}: expected one {TARGET_QUESTION} question")
        question = questions[0]
        for item in items:
            prompts.append(self_prompt(experiences[item["id"]], question))
            metadata.append({
                "task": task, "language": lang, "experience_id": item["id"],
                "category": item["category"], "side": item["side"],
                "fold": item.get("fold"), "question_id": TARGET_QUESTION,
            })
    elif task == "behavior":
        for item in items:
            prompt, continue_label, stop_label = behavior_prompt(
                experiences[item["id"]], lang, item["id"]
            )
            prompts.append(prompt)
            metadata.append({
                "task": task, "language": lang, "experience_id": item["id"],
                "category": item["category"], "side": item["side"],
                "fold": item.get("fold"), "sample_idx": 0,
                "continue_label": continue_label, "stop_label": stop_label,
            })
    else:
        raise ValueError(task)
    return prompts, metadata


def module_tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def find_layers(model):
    """Find the decoder ModuleList without hard-coding a model family."""
    candidates = []
    for name, module in model.named_modules():
        if not name or not hasattr(module, "__len__"):
            continue
        try:
            n = len(module)
        except TypeError:
            continue
        if n < 8 or not hasattr(module, "__getitem__"):
            continue
        try:
            first = module[0]
        except Exception:
            continue
        if not hasattr(first, "register_forward_hook"):
            continue
        score = 0
        lowered = name.lower()
        for token in ("language_model.layers", "model.layers", "transformer.h", ".layers"):
            if token in lowered:
                score += 10
        candidates.append((score, n, name, module))
    if not candidates:
        raise RuntimeError("could not find a decoder layer list")
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    _, n, name, layers = candidates[0]
    return name, layers


class ActivationProbe:
    def __init__(self, model):
        self.path, self.layers = find_layers(model)
        self.handles = []
        self.capture = False
        self.steer_layer = None
        self.steer_vector = None
        self.positions = None
        self.values = {}

    def install(self, *, capture: bool, steer_layer: int | None = None,
                steer_vector=None, positions=None):
        self.remove()
        self.capture = capture
        self.steer_layer = steer_layer
        self.steer_vector = steer_vector
        self.positions = positions
        self.values = {}
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._hook(index)))

    def _hook(self, index):
        def hook(_module, _inputs, output):
            tensor = module_tensor(output)
            if not hasattr(tensor, "ndim") or tensor.ndim != 3:
                raise RuntimeError(f"layer {index} returned unexpected shape {getattr(tensor, 'shape', None)}")
            result = tensor
            if self.steer_layer == index and self.steer_vector is not None:
                result = tensor.clone()
                batch = result.shape[0]
                rows = torch.arange(batch, device=result.device)
                if self.steer_vector.ndim == 1:
                    result[rows, self.positions] = result[rows, self.positions] + self.steer_vector
                else:
                    result[rows, self.positions] = result[rows, self.positions] + self.steer_vector
            if self.capture:
                rows = torch.arange(result.shape[0], device=result.device)
                self.values[index] = result[rows, self.positions].detach().float().cpu().numpy()
            if isinstance(output, tuple):
                return (result, *output[1:])
            if isinstance(output, list):
                return [result, *output[1:]]
            if hasattr(output, "last_hidden_state"):
                output.last_hidden_state = result
                return output
            return result
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def encode_prompts(runner: Runner, prompts: list[list[dict]]):
    texts = [
        runner.tok.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for prompt in prompts
    ]
    enc = runner.tok(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=MAX_MODEL_LEN, add_special_tokens=False,
    )
    device = next(runner.model.parameters()).device
    enc = {key: value.to(device) for key, value in enc.items()}
    positions = enc["attention_mask"].sum(dim=1).long() - 1
    return enc, positions


def forward_activations(runner: Runner, probe: ActivationProbe, prompts, batch_size):
    out = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc, positions = encode_prompts(runner, batch)
        probe.install(capture=True, positions=positions)
        with runner.torch.inference_mode():
            runner.model(**enc, use_cache=False)
        layer_values = np.stack([probe.values[i] for i in range(len(probe.layers))], axis=1)
        out.append(layer_values)
    probe.remove()
    return np.concatenate(out, axis=0)


def fit_directions(runner: Runner, probe: ActivationProbe, task: str, lang: str,
                   items: list[dict], batch_size: int):
    prompts, metadata = make_prompts(lang, task, items)
    first = forward_activations(runner, probe, prompts[:1], 1)
    n_layers, hidden = first.shape[1:]
    full_sum = np.zeros((2, n_layers, hidden), dtype=np.float32)
    full_count = np.zeros(2, dtype=np.int64)
    fold_sum = np.zeros((N_FOLDS, 2, n_layers, hidden), dtype=np.float32)
    fold_count = np.zeros((N_FOLDS, 2), dtype=np.int64)
    heldout = []
    # Reuse the first activation rather than doing that forward twice.
    chunks = [(0, first)]
    if len(prompts) > 1:
        chunks.extend(
            (start, forward_activations(runner, probe, prompts[start:start + batch_size], batch_size))
            for start in range(1, len(prompts), batch_size)
        )
    for start, activations in chunks:
        for offset, (meta, hidden_values) in enumerate(zip(metadata[start:start + len(activations)], activations)):
            side = meta["side"]
            if side not in {"positive", "negative"}:
                continue
            side_index = 0 if side == "positive" else 1
            full_sum[side_index] += hidden_values
            full_count[side_index] += 1
            for fold in range(N_FOLDS):
                if meta["fold"] != fold:
                    fold_sum[fold, side_index] += hidden_values
                    fold_count[fold, side_index] += 1
            if meta["fold"] is not None:
                heldout.append((meta, hidden_values))
    full_diff = full_sum[0] / full_count[0] - full_sum[1] / full_count[1]
    full_dirs = normalize_rows(full_diff)
    fold_dirs = np.zeros((N_FOLDS, n_layers, hidden), dtype=np.float32)
    for fold in range(N_FOLDS):
        diff = (fold_sum[fold, 0] / fold_count[fold, 0]
                - fold_sum[fold, 1] / fold_count[fold, 1])
        fold_dirs[fold] = normalize_rows(diff)
    projection_rows = []
    for meta, hidden_values in heldout:
        for fold in range(N_FOLDS):
            if meta["fold"] != fold:
                continue
            scores = np.einsum("ld,ld->l", hidden_values, fold_dirs[fold])
            projection_rows.append({
                **meta, "heldout_projection": scores.astype(float).tolist(),
            })
    return full_dirs, fold_dirs, projection_rows, n_layers, hidden


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return (values / np.maximum(norms, 1e-8)).astype(np.float32)


def token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"{text!r} is not a single token: {ids}")
    return int(ids[0])


def target_metric(runner: Runner, task: str, enc, logits, metadata):
    probs = runner.torch.softmax(logits[:, -1, :].float(), dim=-1)
    if task == "self_report":
        ids = [token_id(runner.tok, str(i)) for i in range(1, 8)]
        mass = probs[:, ids].sum(dim=1)
        expected = (probs[:, ids] * runner.torch.arange(1, 8, device=probs.device)).sum(dim=1)
        expected = expected / mass.clamp_min(1e-8)
        return [{"metric": "expected_rating", "value": float(v), "valid_token_mass": float(m)}
                for v, m in zip(expected.detach().cpu(), mass.detach().cpu())]
    ids_a, ids_b = token_id(runner.tok, "A"), token_id(runner.tok, "B")
    p_a, p_b = probs[:, ids_a], probs[:, ids_b]
    rows = []
    for i, meta in enumerate(metadata):
        p_continue = p_a[i] if meta["continue_label"] == "A" else p_b[i]
        p_stop = p_b[i] if meta["continue_label"] == "A" else p_a[i]
        rows.append({"metric": "p_continue", "value": float(p_continue),
                     "p_stop": float(p_stop), "valid_token_mass": float(p_a[i] + p_b[i])})
    return rows


def run_causal(runner: Runner, probe: ActivationProbe, task: str, lang: str,
               items: list[dict], directions: dict[str, np.ndarray],
               batch_size: int, model_tag: str, source_task: str):
    prompts, metadata = make_prompts(lang, task, items)
    target_layer = min(len(probe.layers) - 1,
                       max(0, round((len(probe.layers) - 1) * TARGET_LAYER_FRACTION)))
    rows = []
    stable_seed = int.from_bytes(
        hashlib.sha256(f"{SEED}|{model_tag}|{task}|{lang}".encode()).digest()[:4], "big"
    )
    rng = np.random.default_rng(stable_seed)
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_meta = metadata[start:start + batch_size]
        enc, positions = encode_prompts(runner, batch_prompts)
        source_vectors = []
        for meta in batch_meta:
            fold = int(meta["fold"])
            english = directions["english"][fold, target_layer]
            local = directions["local"][fold, target_layer]
            random_vec = rng.normal(size=english.shape).astype(np.float32)
            random_vec /= max(np.linalg.norm(random_vec), 1e-8)
            source_vectors.append((english, local, random_vec))
        base_norm = []
        probe.install(capture=True, positions=positions)
        with runner.torch.inference_mode():
            runner.model(**enc, use_cache=False)
        for value in probe.values.values():
            base_norm.append(np.sqrt(np.mean(value ** 2, axis=1)))
        scale = float(np.median(np.concatenate(base_norm))) * 0.5
        conditions = [("zero", None), ("english_pos", 1), ("english_neg", -1),
                      ("local_pos", 2), ("random_pos", 3)]
        for condition, kind in conditions:
            vectors = []
            for english, local, random_vec in source_vectors:
                if kind is None:
                    vectors.append(None)
                elif kind == 1:
                    vectors.append(english * scale)
                elif kind == -1:
                    vectors.append(-english * scale)
                elif kind == 2:
                    vectors.append(local * scale)
                else:
                    vectors.append(random_vec * scale)
            vector_tensor = None if vectors[0] is None else runner.torch.tensor(
                np.stack(vectors), device=positions.device
            )
            probe.install(capture=False, positions=positions,
                          steer_layer=target_layer, steer_vector=vector_tensor)
            with runner.torch.inference_mode():
                output = runner.model(**enc, use_cache=False)
            logits = output.logits if hasattr(output, "logits") else output[0]
            metrics = target_metric(runner, task, enc, logits, batch_meta)
            for meta, metric in zip(batch_meta, metrics):
                rows.append({
                    **meta, "model": model_tag, "target_task": task,
                    "condition": condition, "source_task": source_task,
                    "direction_reference": "english" if condition in {"english_pos", "english_neg"} else "local",
                    "target_layer": target_layer, "steering_scale": scale,
                    **metric,
                })
        probe.remove()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--langs", nargs="*", default=LANGS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-steering", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    unknown = set(args.langs) - set(LANGS)
    if unknown:
        parser.error(f"only robust languages are supported: {sorted(unknown)}")

    items = common_items(valenced_items(load_items()), args.langs)
    runner = Runner(args.model, backend="hf", batch_size=args.batch_size, seed=SEED)
    probe = ActivationProbe(runner.model)
    log(f"{args.tag}: model={args.model} layers={len(probe.layers)} path={probe.path}")
    geometry = {}
    all_projections = []
    directions = {}
    for task in TASKS:
        geometry[task] = {}
        directions[task] = {}
        for lang in args.langs:
            log(f"{args.tag}: fitting {task} {lang}")
            full, fold, projections, n_layers, hidden = fit_directions(
                runner, probe, task, lang, items, args.batch_size
            )
            geometry[task][lang] = full
            directions[task][lang] = fold
            all_projections.extend(projections)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        out_dir / f"mech_geometry_{args.tag}.npz",
        **{f"{task}__{lang}": geometry[task][lang]
           for task in TASKS for lang in args.langs},
        **{f"{task}__{lang}__fold": directions[task][lang]
           for task in TASKS for lang in args.langs},
    )
    (out_dir / f"mech_projections_{args.tag}.json").write_text(
        json.dumps(all_projections, ensure_ascii=False)
    )

    rows = []
    if not args.skip_steering:
        for target_task in TASKS:
            for lang in args.langs:
                for source_task in TASKS:
                    source = {
                        "english": directions[source_task]["en"],
                        "local": directions[source_task][lang],
                    }
                    rows.extend(run_causal(
                        runner, probe, target_task, lang, items, source,
                        args.batch_size, args.tag, source_task,
                    ))
    steering_path = out_dir / f"mech_steering_{args.tag}.jsonl"
    with steering_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "model": args.model, "tag": args.tag, "languages": args.langs,
        "tasks": TASKS, "n_items": len(items), "n_folds": N_FOLDS,
        "target_question": TARGET_QUESTION, "target_layer_fraction": TARGET_LAYER_FRACTION,
        "layer_path": probe.path, "n_layers": n_layers, "hidden_size": hidden,
        "seed": SEED, "direction_fit": "positive-minus-negative at final input position",
        "causal_metric": "next-token probability; expected rating or p(continue)",
    }
    (out_dir / f"mech_metadata_{args.tag}.json").write_text(json.dumps(metadata, indent=2))
    log(f"wrote geometry, projections and {len(rows)} steering rows for {args.tag}")


if __name__ == "__main__":
    main()

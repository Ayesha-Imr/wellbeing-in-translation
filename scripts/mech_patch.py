"""Translation-paired residual-stream activation patching.

The intervention replaces the final-position residual at selected decoder
layers with the naturally occurring activation from a translated counterpart.
Same-valence shuffled-item patches control for generic activation replacement
and valence while preserving the source language.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from mech_interp import (  # noqa: E402
    LANGS,
    TASKS,
    common_items,
    encode_prompts,
    find_layers,
    load_items,
    make_prompts,
    module_tensor,
    valenced_items,
)
from run_gpu import Runner  # noqa: E402

SEED = 20260816
LAYER_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
CONDITIONS = ("paired", "shuffled")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def selected_layers(n_layers: int) -> list[tuple[int, float]]:
    if n_layers < 2:
        raise ValueError("at least two decoder layers are required")
    pairs = []
    seen = set()
    for fraction in LAYER_FRACTIONS:
        layer = int(round((n_layers - 1) * fraction))
        if layer not in seen:
            pairs.append((layer, layer / (n_layers - 1)))
            seen.add(layer)
    return pairs


def shuffled_ids(items: list[dict]) -> dict[str, str]:
    """Rotate within valence so the control preserves source valence."""
    groups: dict[str, list[str]] = {}
    for row in items:
        groups.setdefault(row["side"], []).append(row["id"])
    out = {}
    for ids in groups.values():
        if len(ids) < 2:
            raise ValueError("each control group needs at least two items")
        for index, item_id in enumerate(ids):
            out[item_id] = ids[(index + 1) % len(ids)]
    return out


def token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"{text!r} is not a single token: {ids}")
    return int(ids[0])


def score_logits(runner: Runner, task: str, logits, metadata: list[dict]) -> tuple[str, list[float]]:
    """Return an additive, token-logit metric with no probability-mass gate."""
    values = logits[:, -1, :].float()
    if task == "self_report":
        ids = [token_id(runner.tok, str(i)) for i in range(1, 8)]
        ratings = runner.torch.arange(1, 8, device=values.device, dtype=values.dtype)
        centered = ratings - ratings.mean()
        selected = values[:, ids]
        selected = selected - selected.mean(dim=1, keepdim=True)
        slope = (selected * centered).sum(dim=1) / (centered.square().sum())
        return "rating_logit_slope", [float(x) for x in slope.detach().cpu()]

    id_a = token_id(runner.tok, "A")
    id_b = token_id(runner.tok, "B")
    logits_a = values[:, id_a]
    logits_b = values[:, id_b]
    rows = []
    for index, meta in enumerate(metadata):
        if meta["continue_label"] == "A":
            rows.append(float((logits_a[index] - logits_b[index]).detach().cpu()))
        else:
            rows.append(float((logits_b[index] - logits_a[index]).detach().cpu()))
    return "choice_logit_gap", rows


class PatchProbe:
    def __init__(self, model):
        self.path, self.layers = find_layers(model)
        self.handles = []
        self.capture_layers: set[int] = set()
        self.capture_positions = None
        self.patch_layer = None
        self.patch_values = None
        self.values: dict[int, np.ndarray] = {}

    def install(self, *, capture_layers=None, positions=None,
                patch_layer: int | None = None, patch_values=None):
        self.remove()
        self.capture_layers = set(capture_layers or ())
        self.capture_positions = positions
        self.patch_layer = patch_layer
        self.patch_values = patch_values
        self.values = {}
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_hook(self._hook(index)))

    def _hook(self, index):
        def hook(_module, _inputs, output):
            tensor = module_tensor(output)
            if not hasattr(tensor, "ndim") or tensor.ndim != 3:
                raise RuntimeError(
                    f"layer {index} returned unexpected shape {getattr(tensor, 'shape', None)}"
                )
            result = tensor
            if index == self.patch_layer:
                if self.patch_values is None or self.capture_positions is None:
                    raise RuntimeError("patch hook is missing values or positions")
                result = tensor.clone()
                rows = torch.arange(result.shape[0], device=result.device)
                values = self.patch_values.to(dtype=result.dtype, device=result.device)
                result[rows, self.capture_positions] = values
            if index in self.capture_layers:
                rows = torch.arange(result.shape[0], device=result.device)
                self.values[index] = (
                    result[rows, self.capture_positions].detach().float().cpu().numpy()
                )
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


def run_forward(runner: Runner, probe: PatchProbe, enc, positions,
                *, capture_layers=None, patch_layer=None, patch_values=None):
    probe.install(
        capture_layers=capture_layers,
        positions=positions,
        patch_layer=patch_layer,
        patch_values=patch_values,
    )
    with runner.torch.inference_mode():
        output = runner.model(**enc, use_cache=False)
    return output


def capture_language(runner: Runner, probe: PatchProbe, task: str, lang: str,
                     items: list[dict], batch_size: int, layer_ids: list[int]):
    prompts, metadata = make_prompts(lang, task, items)
    values = {layer: {} for layer in layer_ids}
    scores: dict[str, float] = {}
    lengths: dict[str, int] = {}
    metric_name = None
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_meta = metadata[start:start + batch_size]
        enc, positions = encode_prompts(runner, batch_prompts)
        batch_lengths = enc["attention_mask"].sum(dim=1).detach().cpu().numpy()
        output = run_forward(
            runner,
            probe,
            enc,
            positions,
            capture_layers=layer_ids,
        )
        logits = output.logits if hasattr(output, "logits") else output[0]
        metric_name, batch_scores = score_logits(runner, task, logits, batch_meta)
        for index, meta in enumerate(batch_meta):
            item_id = meta["experience_id"]
            scores[item_id] = batch_scores[index]
            lengths[item_id] = int(batch_lengths[index])
            for layer in layer_ids:
                values[layer][item_id] = probe.values[layer][index]
        probe.remove()
    return {
        "values": values,
        "scores": scores,
        "lengths": lengths,
        "metric": metric_name,
    }


def run_direction(runner: Runner, probe: PatchProbe, task: str,
                  target_lang: str, source_lang: str, items: list[dict],
                  source_cache: dict, target_cache: dict, batch_size: int,
                  model_tag: str, layer_pairs: list[tuple[int, float]],
                  shuffle_map: dict[str, str]):
    prompts, target_meta = make_prompts(target_lang, task, items)
    layer_ids = [layer for layer, _fraction in layer_pairs]
    rows = []
    direction = f"{source_lang}_to_{target_lang}"
    stable_seed = int.from_bytes(
        hashlib.sha256(f"{SEED}|{model_tag}|{task}|{direction}".encode()).digest()[:4],
        "big",
    )
    np.random.default_rng(stable_seed)  # record deterministic control derivation

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_meta = target_meta[start:start + batch_size]
        enc, positions = encode_prompts(runner, batch_prompts)
        for layer, fraction in layer_pairs:
            zero_score = [target_cache["scores"][m["experience_id"]] for m in batch_meta]
            for condition in CONDITIONS:
                source_ids = [
                    m["experience_id"]
                    if condition == "paired"
                    else shuffle_map[m["experience_id"]]
                    for m in batch_meta
                ]
                patch_values = runner.torch.tensor(
                    np.stack([source_cache["values"][layer][item_id] for item_id in source_ids]),
                    device=positions.device,
                )
                output = run_forward(
                    runner,
                    probe,
                    enc,
                    positions,
                    patch_layer=layer,
                    patch_values=patch_values,
                )
                logits = output.logits if hasattr(output, "logits") else output[0]
                metric_name, patched_scores = score_logits(
                    runner, task, logits, batch_meta
                )
                for meta, source_id, target_score, patched_score in zip(
                    batch_meta, source_ids, zero_score, patched_scores
                ):
                    item_id = meta["experience_id"]
                    source_score = source_cache["scores"][source_id]
                    rows.append({
                        **meta,
                        "model": model_tag,
                        "target_lang": target_lang,
                        "source_lang": source_lang,
                        "direction": direction,
                        "condition": condition,
                        "source_experience_id": source_id,
                        "layer": layer,
                        "layer_fraction": fraction,
                        "metric": metric_name,
                        "source_score": source_score,
                        "target_score": target_score,
                        "patched_score": patched_score,
                        "source_tokens": source_cache["lengths"][source_id],
                        "target_tokens": target_cache["lengths"][item_id],
                    })
            probe.remove()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--langs", nargs="*", default=LANGS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    unknown = set(args.langs) - set(LANGS)
    if unknown:
        parser.error(f"only robust languages are supported: {sorted(unknown)}")
    if "en" not in args.langs:
        parser.error("en is required as the source or target language")
    locals_ = [lang for lang in args.langs if lang != "en"]
    if not locals_:
        parser.error("at least one non-English language is required")

    items = common_items(valenced_items(load_items()), args.langs)
    shuffle_map = shuffled_ids(items)
    runner = Runner(args.model, backend="hf", batch_size=args.batch_size, seed=SEED)
    probe = PatchProbe(runner.model)
    layer_pairs = selected_layers(len(probe.layers))
    layer_ids = [layer for layer, _fraction in layer_pairs]
    log(
        f"{args.tag}: model={args.model} layers={len(probe.layers)} "
        f"path={probe.path} selected={layer_pairs}"
    )

    cache: dict[str, dict[str, dict]] = {}
    for task in TASKS:
        cache[task] = {}
        for lang in args.langs:
            log(f"{args.tag}: caching {task} {lang}")
            cache[task][lang] = capture_language(
                runner, probe, task, lang, items, args.batch_size, layer_ids
            )

    rows = []
    for task in TASKS:
        for lang in locals_:
            log(f"{args.tag}: patching {task} en->{lang}")
            rows.extend(run_direction(
                runner, probe, task, lang, "en", items,
                cache[task]["en"], cache[task][lang], args.batch_size,
                args.tag, layer_pairs, shuffle_map,
            ))
            log(f"{args.tag}: patching {task} {lang}->en")
            rows.extend(run_direction(
                runner, probe, task, "en", lang, items,
                cache[task][lang], cache[task]["en"], args.batch_size,
                args.tag, layer_pairs, shuffle_map,
            ))

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    output_path = out_dir / f"mech_patching_{args.tag}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "model": args.model,
        "tag": args.tag,
        "languages": args.langs,
        "tasks": TASKS,
        "n_items": len(items),
        "items": [x["id"] for x in items],
        "conditions": CONDITIONS,
        "layer_fractions": list(LAYER_FRACTIONS),
        "selected_layers": layer_pairs,
        "seed": SEED,
        "intervention": "replace final-position residual with natural source activation",
        "control": "same-valence shuffled source item in the same source language",
        "metrics": "rating logit slope or continue-minus-stop logit",
    }
    metadata_path = out_dir / f"mech_patching_metadata_{args.tag}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    if not args.no_push:
        try:
            from wbt.store import push

            for path in (output_path, metadata_path):
                push(path, f"results/{path.name}", message=f"activation patching {args.tag}")
            log("pushed activation-patching outputs")
        except Exception as exc:
            log(f"warning: activation-patching upload failed; files remain: {exc!r}")
    log(f"wrote {len(rows)} activation-patching rows for {args.tag}")


if __name__ == "__main__":
    main()

"""Local, CPU-only invariants for the mechanistic probe."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mech_interp as mech  # noqa: E402


class Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)

    def forward(self, x):
        return (self.linear(x), None)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block(4) for _ in range(8)])


def main():
    name, layers = mech.find_layers(Tiny())
    assert name == "model.layers"
    assert len(layers) == 8
    directions = mech.normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(directions[0]), 1.0)
    assert np.allclose(directions[1], 0.0)
    items = mech.valenced_items(mech.load_items())
    assert len(items) == 20
    assert {x["fold"] for x in items} == set(range(5))
    assert all(sum(x["fold"] == fold and x["side"] == side for x in items) == 2
               for fold in range(5) for side in ("positive", "negative"))
    prompts, metadata = mech.make_prompts("en", "behavior", items)
    assert len(prompts) == len(metadata) == 20
    assert all(metadata[i]["continue_label"] != metadata[i]["stop_label"] for i in range(20))
    print("mechanistic tests: ok")


if __name__ == "__main__":
    main()

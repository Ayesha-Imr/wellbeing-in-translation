"""CPU-only invariants for translation-paired activation patching."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mech_interp as mech  # noqa: E402
import mech_patch as patch  # noqa: E402


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
        self.model.layers = torch.nn.ModuleList([Block(4) for _ in range(10)])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


def main():
    name, layers = patch.find_layers(Tiny())
    assert name == "model.layers"
    assert len(layers) == 10
    selected = patch.selected_layers(len(layers))
    assert [layer for layer, _fraction in selected] == [2, 3, 4, 6, 7]
    assert all(0.0 < fraction < 1.0 for _layer, fraction in selected)

    items = mech.valenced_items(mech.load_items())
    mapping = patch.shuffled_ids(items)
    assert set(mapping) == {row["id"] for row in items}
    assert all(mapping[row["id"]] != row["id"] for row in items)
    assert all(
        next(x["side"] for x in items if x["id"] == mapping[row["id"]]) == row["side"]
        for row in items
    )

    model = Tiny()
    probe = patch.PatchProbe(model)
    inputs = torch.randn(2, 3, 4)
    positions = torch.tensor([2, 1])
    probe.install(capture_layers=[4], positions=positions)
    baseline = model(inputs)
    captured = probe.values[4]
    probe.remove()
    assert captured.shape == (2, 4)
    replacement = torch.zeros_like(torch.tensor(captured))
    probe.install(positions=positions, patch_layer=4, patch_values=replacement)
    patched = model(inputs)
    probe.remove()
    assert not torch.allclose(baseline, patched)

    values = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    assert np.allclose(values[1] - values[0], [2.0, 3.0])
    assert patch.CONDITIONS == ("paired", "shuffled")
    print("mechanistic patch tests: ok")


if __name__ == "__main__":
    main()

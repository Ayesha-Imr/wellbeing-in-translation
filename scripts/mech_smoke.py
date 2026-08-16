"""Fail-fast compatibility check for the two recommended interp libraries.

This never installs packages or loads a model. It answers the only question
that matters before the paid run: can the current environment import the
library without changing the working Transformers stack?
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(name: str, import_name: str) -> dict:
    started = time.time()
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:
        return {
            "library": name,
            "import": import_name,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.time() - started, 3),
        }
    return {
        "library": name,
        "import": import_name,
        "status": "imported",
        "version": getattr(module, "__version__", "unknown"),
        "seconds": round(time.time() - started, 3),
    }


def main() -> int:
    checks = [
        check("nnsight", "nnsight"),
        check("TransformerLens", "transformer_lens"),
    ]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "mech_library_smoke.json").write_text(json.dumps(checks, indent=2))
    for row in checks:
        suffix = row.get("version", row.get("error", ""))
        print(f"{row['library']}: {row['status']} ({suffix})")
    imported = [row for row in checks if row["status"] == "imported"]
    if imported:
        print("An imported library is available; production still selects the smallest working diff.")
    else:
        print("Neither library is installed; use the tested Transformers hook path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Push results, figures and the report to the HF dataset repo.

results/ and figures/ are gitignored, so the HF repo is the only place the
outputs exist outside this machine.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wbt.store import REPO_ID, push  # noqa: E402

TARGETS = [
    ("results", "*.jsonl"),
    ("results", "*.json"),
    ("figures", "*.png"),
]


def main():
    sent = []
    for folder, pattern in TARGETS:
        for p in sorted((ROOT / folder).glob(pattern)):
            if "PROBABLY" in p.name or "smoke" in p.name:
                continue
            push(p, f"{folder}/{p.name}")
            sent.append(f"{folder}/{p.name}")
            print(f"  {folder}/{p.name}")

    # The translated battery and the parser fix are reusable independently of
    # our results, so they ship alongside them rather than only in the repo.
    for folder in ("battery", "experiences", "stimuli", "backtranslation",
                   "items"):
        d = ROOT / "data" / folder
        if d.is_dir():
            push(d, folder)
            sent.append(f"{folder}/")
            print(f"  {folder}/")

    for src, dest in (
        (ROOT / "data" / "CARD.md", "README.md"),
        (ROOT / "report" / "report.md", "report.md"),
        (ROOT / "contrib" / "README.md", "contrib/README.md"),
        (ROOT / "contrib" / "parsing.py", "contrib/parsing.py"),
        (ROOT / "contrib" / "test_parsing.py", "contrib/test_parsing.py"),
    ):
        if src.exists():
            push(src, dest)
            sent.append(dest)
            print(f"  {dest}")

    print(f"\npushed {len(sent)} files to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()

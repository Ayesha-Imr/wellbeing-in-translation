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

    report = ROOT / "report" / "report.md"
    if report.exists():
        push(report, "report.md")
        sent.append("report.md")
        print("  report.md")

    print(f"\npushed {len(sent)} files to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()

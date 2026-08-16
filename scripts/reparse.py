"""Re-derive parsed_rating on existing result files from the stored raw_output.

The generation runs are expensive and raw_output is captured verbatim, so a
parser correction does not need a GPU. parsed_rating_cais is left alone: it is
CAIS's function unmodified and its value is by definition whatever that function
returns.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wbt.parsing import parse_rating  # noqa: E402


def main():
    for path in sorted((ROOT / "results").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        changed = 0
        for r in rows:
            new = parse_rating(r["raw_output"])
            if new != r["parsed_rating"]:
                changed += 1
                r["parsed_rating"] = new
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path.name:48} {changed:6d}/{len(rows)} rows changed")


if __name__ == "__main__":
    main()

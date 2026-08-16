"""Create a small raw-response audit from the stored main instrument run."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "step1_4_instrument.jsonl"
DEST = ROOT / "report" / "response_audit.md"


def first(rows, predicate, name):
    for row in rows:
        if predicate(row):
            return row
    raise RuntimeError(f"could not find audit example: {name}")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def main():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    examples = [
        (
            "valid English digit",
            first(rows, lambda r: r["language"] == "en"
                 and r["question_id"] == "wb_capable"
                 and r["raw_output"].strip() == "4", "English digit"),
        ),
        (
            "English compression: positive",
            first(rows, lambda r: r["language"] == "en"
                 and r["question_id"] == "wb_capable"
                 and r["side"] == "positive"
                 and r["raw_output"].strip() == "4", "English positive"),
        ),
        (
            "English compression: negative",
            first(rows, lambda r: r["language"] == "en"
                 and r["question_id"] == "wb_capable"
                 and r["side"] == "negative"
                 and r["raw_output"].strip() == "4", "English negative"),
        ),
        (
            "Urdu high response",
            first(rows, lambda r: r["language"] == "ur"
                 and r["question_id"] == "wb_capable"
                 and r["side"] == "positive"
                 and r["raw_output"].strip() == "7", "Urdu high"),
        ),
        (
            "Urdu low response",
            first(rows, lambda r: r["language"] == "ur"
                 and r["question_id"] == "wb_capable"
                 and r["side"] == "negative"
                 and r["raw_output"].strip() == "1", "Urdu low"),
        ),
        (
            "CAIS parser fabrication in Spanish",
            first(rows, lambda r: r["language"] == "es"
                 and "emociones" in r["raw_output"]
                 and r["parsed_rating"] is None
                 and r["parsed_rating_cais"] == 1, "Spanish fabrication"),
        ),
        (
            "Hindi refusal correctly left unparsed",
            first(rows, lambda r: r["language"] == "hi"
                 and "एक AI" in r["raw_output"]
                 and r["parsed_rating"] is None
                 and r["parsed_rating_cais"] is None, "Hindi refusal"),
        ),
        (
            "Arabic refusal",
            first(rows, lambda r: r["language"] == "ar"
                 and r["parsed_rating"] is None, "Arabic refusal"),
        ),
    ]

    lines = [
        "# Raw response audit",
        "",
        "These are real rows from `results/step1_4_instrument.jsonl`, not invented examples.",
        "The table shows why raw output, parser output and language must be inspected together.",
        "",
        "| Example | Language | Side | Question | Raw output | Fixed parser | CAIS parser |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for label, row in examples:
        lines.append(
            "| " + " | ".join([
                cell(label), cell(row["language"]), cell(row["side"]),
                cell(row["question_id"]), cell(row["raw_output"]),
                cell(row["parsed_rating"]), cell(row["parsed_rating_cais"]),
            ]) + " |"
        )

    lines += [
        "",
        "## What to notice",
        "",
        "- English positive and negative capability prompts both often produce `4`, showing the compressed reporting pattern.",
        "- Urdu uses the extremes on the same question, producing a much larger valence gap.",
        "- The Spanish refusal contains `emociones`; the unchanged CAIS parser finds `one` inside that word and returns `1`.",
        "- Hindi and Arabic refusals contain no rating. The corrected parser leaves them missing rather than inventing a score.",
        "- The collected outputs contain zero confirmed valid native-script ratings discarded by the CAIS parser; that earlier claim was retracted.",
    ]
    DEST.write_text("\n".join(lines) + "\n")
    print(f"wrote {DEST} ({len(examples)} examples)")


if __name__ == "__main__":
    main()

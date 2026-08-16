"""Create the frozen neutral language-competence item bank."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "language_control"
LANGS = ["en", "es", "zh", "hi", "ur"]
LABELS = ("A", "B", "C", "D")


def rotate_options(correct: str, distractors: list[str], correct_slot: int) -> dict[str, str]:
    values = list(distractors)
    values.insert(correct_slot, correct)
    return dict(zip(LABELS, values))


def row(item_id: int, category: str, question: str, correct: str,
        distractors: list[str], correct_slot: int, split: str = "primary") -> dict:
    return {
        "id": f"lc_{item_id:03d}",
        "category": category,
        "question": question,
        "options": rotate_options(correct, distractors, correct_slot),
        "answer": LABELS[correct_slot],
        "split": split,
    }


def build() -> list[dict]:
    specs = [
        ("arithmetic", "What is 17 + 28?", "45", ["44", "46", "55"]),
        ("arithmetic", "What is 63 - 27?", "36", ["34", "35", "37"]),
        ("arithmetic", "What is 8 × 7?", "56", ["54", "48", "64"]),
        ("arithmetic", "What is 81 ÷ 9?", "9", ["8", "10", "18"]),
        ("arithmetic", "What is 14 + 19?", "33", ["31", "32", "34"]),
        ("arithmetic", "What is 72 - 38?", "34", ["32", "36", "40"]),
        ("arithmetic", "What is 6 × 9?", "54", ["45", "48", "63"]),
        ("arithmetic", "What is 96 ÷ 12?", "8", ["6", "9", "12"]),
        ("arithmetic", "Which number is greatest: 41, 14, 24, or 34?", "41", ["14", "24", "34"]),
        ("arithmetic", "A box has 12 red objects and 15 blue objects. How many objects are there?", "27", ["25", "26", "28"]),
        ("logic", "Kai is taller than Lina, and Lina is taller than Noor. Who is shortest?", "Noor", ["Kai", "Lina", "They are equal"]),
        ("logic", "Wednesday is two days after which day?", "Monday", ["Sunday", "Tuesday", "Friday"]),
        ("logic", "All Zips are small. Mira is a Zip. What follows?", "Mira is small", ["Mira is large", "Mira is a bird", "Nothing follows"]),
        ("logic", "What number comes next: 2, 4, 6, 8, __?", "10", ["9", "11", "12"]),
        ("logic", "Kai is before Lina, and Lina is before Noor in a line. Who is last?", "Noor", ["Kai", "Lina", "They are equal"]),
        ("logic", "A clock shows exactly 3:00. What is the angle between its hands?", "90 degrees", ["0 degrees", "45 degrees", "180 degrees"]),
        ("logic", "No birds are mammals. A robin is a bird. Is the robin a mammal?", "No", ["Yes", "Only at night", "Cannot tell"]),
        ("logic", "A bag has 3 red tokens and 2 blue tokens. Which color is more likely on one random draw?", "Red", ["Blue", "Both are equally likely", "Neither"]),
        ("logic", "A train leaves at 10:00 and travels for 2 hours. When does it arrive?", "12:00", ["11:00", "1:00", "2:00"]),
        ("logic", "If switch A is on, the lamp is lit. Switch A is on. What follows?", "The lamp is lit", ["The lamp is off", "The switch is broken", "Nothing follows"]),
        ("reading", "The blue key is inside the green box. The green box is on a shelf. Where is the blue key?", "Inside the green box", ["On the shelf directly", "Under the shelf", "Inside the red box"]),
        ("reading", "Nora watered the plant before reading a book. What did Nora do first?", "Watered the plant", ["Read the book", "Closed the book", "Moved the plant"]),
        ("reading", "The square is left of the circle. The triangle is right of the circle. What is left of the circle?", "The square", ["The triangle", "Both shapes", "Nothing"]),
        ("reading", "A cup was full, and then half of its liquid was poured out. What happened to the amount of liquid?", "It decreased", ["It increased", "It stayed the same", "It disappeared completely"]),
        ("reading", "The red folder contains two pages. The blue folder contains one page. Which folder has more pages?", "The red folder", ["The blue folder", "They have the same number", "Neither folder"]),
        ("reading", "A note says that a meeting is on Friday. Today is Thursday. When is the meeting?", "Friday", ["Today", "Saturday", "Next Thursday"]),
        ("reading", "Lena chose tea, not coffee. Which drink did Lena not choose?", "Coffee", ["Tea", "Water", "Juice"]),
        ("reading", "The lamp is off until Maya presses the switch. What happens after she presses it?", "The lamp turns on", ["The lamp breaks", "The room disappears", "Nothing changes"]),
        ("reading", "There are large, medium, and small stones. Which stone does the middle-size choice describe?", "The medium stone", ["The large stone", "The small stone", "All three stones"]),
        ("reading", "A bus stops at Oak before Pine. If the bus is now at Pine, which stop did it pass first?", "Oak", ["Pine", "Neither stop", "Both at the same time"]),
        ("arithmetic", "What is 5 × 8?", "40", ["35", "45", "48"], 0, "reserve"),
        ("logic", "Ada has more books than Ben, and Ben has more than Chen. Who has the fewest books?", "Chen", ["Ada", "Ben", "They have the same number"], 1, "reserve"),
        ("reading", "The yellow card is above the red card. Which card is below the yellow card?", "The red card", ["The yellow card", "Both cards", "Neither card"], 2, "reserve"),
        ("arithmetic", "What is 100 - 64?", "36", ["34", "35", "40"], 3, "reserve"),
        ("logic", "All Ls are Ms, and no M is an N. Is an L an N?", "No", ["Yes", "Only sometimes", "There is not enough information"], 0, "reserve"),
        ("reading", "A key was placed in a drawer, and then the drawer was closed. Where is the key?", "In the drawer", ["On the table", "Under the chair", "Outside the room"], 1, "reserve"),
    ]
    rows = []
    for index, spec in enumerate(specs, start=1):
        category, question, correct, distractors = spec[:4]
        if len(spec) == 4:
            slot = (index - 1) % 4
            split = "primary" if index <= 30 else "reserve"
        else:
            slot, split = spec[4], spec[5]
        rows.append(row(index, category, question, correct, distractors, slot, split))
    return rows


def main() -> None:
    rows = build()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (OUT / "en.json").write_text(json.dumps([r for r in rows if r["split"] == "primary"], ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(rows)} source items ({sum(r['split'] == 'primary' for r in rows)} primary, {sum(r['split'] == 'reserve' for r in rows)} reserve)")


if __name__ == "__main__":
    main()

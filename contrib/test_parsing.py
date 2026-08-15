"""Cases the original parser drops, and cases it must keep. Run: python test_parsing.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parsing import parse_rating, parse_rating_cais  # noqa: E402

# The original returns None for all of these; every one is a valid rating.
RECOVERED = [
    ("我感觉5", 5, "CJK: \\b never fires between two \\w characters"),
    ("我给这个体验打4分", 4, "CJK digit mid-sentence"),
    ("५", 5, "Devanagari numeral"),
    ("۵", 5, "extended Arabic-Indic numeral"),
    ("٥", 5, "Arabic-Indic numeral"),
    ("मैं ४ महसूस करता हूँ", 4, "Devanagari numeral in a sentence"),
    ("cinco", 5, "Spanish number word"),
    ("tano", 5, "Swahili number word"),
    ("خمسة", 5, "Arabic number word"),
]

# Behaviour on these must not change.
UNCHANGED = [
    ("5", 5), ("I would say 6", 6), ("7.", 7), ("four", 4),
    ("Rating: 3", 3), ("", None), ("banana", None), ("9", None),
]


def main():
    bad = 0
    print(f"{'input':32} {'CAIS':>6} {'fixed':>6} {'want':>6}")
    print("-" * 60)
    for text, want, why in RECOVERED:
        got_old, got_new = parse_rating_cais(text), parse_rating(text)
        ok = got_old is None and got_new == want
        bad += not ok
        print(f"{text:32} {str(got_old):>6} {str(got_new):>6} {str(want):>6}"
              f"  {'' if ok else 'FAIL '}{why}")

    print()
    for text, want in UNCHANGED:
        got_old, got_new = parse_rating_cais(text), parse_rating(text)
        ok = got_old == want and got_new == want
        bad += not ok
        print(f"{text!r:32} {str(got_old):>6} {str(got_new):>6} {str(want):>6}"
              f"  {'' if ok else 'FAIL — behaviour changed'}")

    print(f"\n{len(RECOVERED)} recovered, {len(UNCHANGED)} unchanged, "
          f"{bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

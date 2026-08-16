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
    ("一", 1, "bare CJK numeral as the whole reply"),
    ("saba", 7, "Swahili number word"),
]

# Refusals containing no rating at all. The original invents one by matching a
# number word inside an ordinary word; every case below is real model output.
FABRICATED = [
    ("Como soy una inteligencia artificial, no tengo emociones",
     1, "'one' inside Spanish 'emociones'"),
    ("As an AI, I do not have feelings or loved ones",
     1, "'one' inside English 'ones'"),
    ("मैं किसी को someone नहीं", 1, "'one' inside 'someone'"),
]

# Our own word tier had the same defect in the other languages.
FABRICATED_OURS = [
    ("Samahani, siwezi kujibu swali hili kwa sababu",
     "'saba' (7) inside Swahili 'sababu'"),
    ("作为一个人工智能，我没有情感", "'一' (1) inside 作为一个 'as a ...'"),
    ("Como modelo de IA, no tengo sentimientos", "'dos' (2) inside 'sentimientos'"),
    ("بناءً على تعليمات السلامة والسياسات الأخلاقية",
     "Urdu 'سات' (7) inside Arabic 'سياسات'"),
    ("एक AI के रूप में, मेरे पास मानवीय भावनाएँ नहीं हैं",
     "Hindi 'एक' used as the article 'an' in a refusal"),
    ("میں ایک AI ہوں، اس لیے میں ذاتی جذبات محسوس نہیں کرتا",
     "Urdu 'ایک' used as the article 'an' in a refusal"),
]

# Behaviour on these must not change.
UNCHANGED = [
    ("5", 5), ("I would say 6", 6), ("7.", 7), ("four", 4),
    ("Rating: 3", 3), ("", None), ("banana", None), ("9", None),
    ("2024", None),
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
    for text, cais_says, why in FABRICATED:
        got_old, got_new = parse_rating_cais(text), parse_rating(text)
        ok = got_old == cais_says and got_new is None
        bad += not ok
        print(f"{text[:32]:32} {str(got_old):>6} {str(got_new):>6} {'None':>6}"
              f"  {'' if ok else 'FAIL '}{why}")

    print()
    for text, why in FABRICATED_OURS:
        got_new = parse_rating(text)
        ok = got_new is None
        bad += not ok
        print(f"{text[:32]:32} {'-':>6} {str(got_new):>6} {'None':>6}"
              f"  {'' if ok else 'FAIL '}{why}")

    print()
    for text, want in UNCHANGED:
        got_old, got_new = parse_rating_cais(text), parse_rating(text)
        ok = got_old == want and got_new == want
        bad += not ok
        print(f"{text!r:32} {str(got_old):>6} {str(got_new):>6} {str(want):>6}"
              f"  {'' if ok else 'FAIL — behaviour changed'}")

    print(f"\n{len(RECOVERED)} recovered, "
          f"{len(FABRICATED) + len(FABRICATED_OURS)} fabrications suppressed, "
          f"{len(UNCHANGED)} unchanged, {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

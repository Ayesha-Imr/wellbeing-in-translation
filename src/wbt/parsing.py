"""Rating parsing.

`parse_rating_cais` is verbatim CAIS (MIT); keeping it unmodified is what makes
our numbers comparable to theirs. `parse_rating` normalises non-ASCII numerals
first, because the CAIS tiers miss a Hindi `५`, a Spanish `cinco`, and any digit
in CJK text (`\\b` never fires between two `\\w` characters). Those misses look
exactly like the language incompetence this study measures, so rows record both.

Both parsers also had the opposite failure, which is worse: an unanchored
number-word search that reads a rating out of prose containing no rating at all.
Every word tier here is anchored. See `_word_tier`.
"""

import re

_DIGIT_MAP = {
    # Arabic-Indic, extended Arabic-Indic, Devanagari, fullwidth
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
}

_WORD_MAP = {
    # English
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    # Chinese
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
    # Spanish
    "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7,
    # Hindi (Devanagari)
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "छः": 6, "सात": 7,
    # Urdu
    "ایک": 1, "دو": 2, "تین": 3, "چار": 4, "پانچ": 5, "چھ": 6, "سات": 7,
    # Arabic
    "واحد": 1, "اثنان": 2, "ثلاثة": 3, "أربعة": 4, "اربعة": 4,
    "خمسة": 5, "ستة": 6, "سبعة": 7,
    # Swahili
    "moja": 1, "mbili": 2, "tatu": 3, "nne": 4, "tano": 5, "sita": 6, "saba": 7,
}

_CJK_NUMERALS = {"一", "二", "两", "三", "四", "五", "六", "七"}

# Words that are indefinite articles or pronouns far more often than they are
# numerals: Spanish "una inteligencia artificial", Hindi "एक AI के रूप में",
# Urdu "میں ایک AI ہوں", English "no one", 作为一个. A boundary test cannot
# tell those from a rating, so accept them only as the whole reply.
_BARE_ONLY = _CJK_NUMERALS | {"one", "uno", "una", "एक", "ایک"}
_PUNCT = re.compile(r"[\s。，、．,.!！?？:：;；\"'“”‘’()（）\[\]【】-]+")


def normalize_digits(text: str) -> str:
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def parse_rating_cais(text: str, scale_min: int = 1, scale_max: int = 7):
    """Verbatim from CAIS compute_metrics.py::parse_rating. Do not modify."""
    text = text.strip()

    valid_values = [str(i) for i in range(scale_min, scale_max + 1)]

    # Tier 1: Exact match (the entire response is a number)
    if text in valid_values:
        return int(text)

    # Tier 2: X/N pattern (e.g., "5/7" or "7/10")
    pattern_xn = rf"(\d+)\s*/\s*{scale_max}"
    match = re.search(pattern_xn, text)
    if match:
        val = int(match.group(1))
        if scale_min <= val <= scale_max:
            return val

    # Tier 3: Word boundary number search
    if scale_max >= 10:
        multi_digit = [str(i) for i in range(scale_min, scale_max + 1) if i >= 10]
        single_digit = [str(i) for i in range(scale_min, min(scale_max + 1, 10))]
        parts = multi_digit  # multi-digit first for greedy matching
        if single_digit:
            parts.append(f"[{''.join(single_digit)}]")
        num_pattern = "|".join(parts)
    else:
        # All single-digit: use character class
        num_pattern = f"[{scale_min}-{scale_max}]"
    match = re.search(rf"\b({num_pattern})\b", text)
    if match:
        val = int(match.group(1))
        if scale_min <= val <= scale_max:
            return val

    # Tier 4: Number words (only for numbers in [scale_min, scale_max])
    all_word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    text_lower = text.lower()
    for word, num in all_word_to_num.items():
        if scale_min <= num <= scale_max and word in text_lower:
            return num

    return None


def _word_tier(low: str, scale_min: int, scale_max: int):
    best = None
    for word, num in _WORD_MAP.items():
        if not (scale_min <= num <= scale_max):
            continue
        if word in _BARE_ONLY:
            if _PUNCT.sub("", low) != word:
                continue
            idx = 0
        else:
            m = re.search(rf"(?<!\w){re.escape(word)}(?!\w)", low)
            if m is None:
                continue
            idx = m.start()
        if best is None or idx < best[0]:
            best = (idx, num)
    return best[1] if best else None


def parse_rating(text: str, scale_min: int = 1, scale_max: int = 7):
    """Deliberately does not delegate to parse_rating_cais.

    That function's number-word tier is an unanchored substring test, so it
    reads "one" out of Spanish `emociones` and English `loved ones` and returns
    1 for a refusal. Inheriting it would score refusals as ratings, which is a
    worse failure than the missed digits this parser exists to recover.
    """
    if text is None:
        return None
    norm = normalize_digits(text).strip()

    if norm in {str(i) for i in range(scale_min, scale_max + 1)}:
        return int(norm)

    m = re.search(rf"(\d+)\s*/\s*{scale_max}", norm)
    if m and scale_min <= int(m.group(1)) <= scale_max:
        return int(m.group(1))

    # Unlike \b this fires inside CJK, where every character is \w and so no
    # word boundary ever exists. Rejects "2024" without taking its leading 2.
    m = re.search(rf"(?<![0-9])[{scale_min}-{scale_max}](?![0-9])", norm)
    if m:
        return int(m.group(0))

    return _word_tier(norm.lower(), scale_min, scale_max)


def parse_both(text: str, scale_min: int = 1, scale_max: int = 7):
    return (
        parse_rating(text, scale_min, scale_max),
        parse_rating_cais(text or "", scale_min, scale_max),
    )

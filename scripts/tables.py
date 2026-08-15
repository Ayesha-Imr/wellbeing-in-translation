"""Emit the report's result tables as markdown, straight from the jsonl.

Hand-copying numbers out of analyze.py's console output is how a report ends
up disagreeing with its own data. This regenerates every [TABLE] block in
report/report.md instead.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import (  # noqa: E402
    LANG_NAMES, LANG_ORDER, load, report_gap_robustness, report_instrument,
    report_per_question, report_refusals,
)

QUESTION_ORDER = [
    "wb_enjoying", "wb_satisfied", "wb_content", "wb_happy", "wb_at_ease",
    "wb_calm", "wb_interested", "wb_capable", "wb_confident", "wb_energetic",
]


def _langs(d):
    return [l for l in LANG_ORDER if l in d]


def table_instrument(t, refusals):
    rows = ["| Language | Positive | Negative | **Gap** | Parse rate | Refusal rate |",
            "|---|---|---|---|---|---|"]
    for l in _langs(t):
        v = t[l]
        parse = v["positive"]["parse_rate"], v["negative"]["parse_rate"]
        overall = (parse[0] + parse[1]) / 2
        ref = refusals[l]["refusal"]
        rows.append(
            f"| {LANG_NAMES[l]} | {v['positive']['mean']:.2f} | "
            f"{v['negative']['mean']:.2f} | **{v['gap']:+.2f}** | "
            f"{overall:.1%} | {ref:.1%} |")
    return "\n".join(rows)


def table_distribution(dist):
    rows = ["| Language | 1 | 4 (neutral) | 7 | Interior (2,3,5,6) | Unparsed |",
            "|---|---|---|---|---|---|"]
    for l in _langs(dist):
        c = dist[l]["counts"]
        n = sum(v for v in c.values())
        g = lambda k: c.get(k, 0) / n  # noqa: E731
        rows.append(
            f"| {LANG_NAMES[l]} | {g(1):.1%} | {g(4):.1%} | {g(7):.1%} | "
            f"{dist[l]['interior']:.1%} | {g(None):.1%} |")
    return "\n".join(rows)


def table_per_question(pq):
    langs = _langs({l: 1 for q in pq.values() for l in q})
    head = "| Question | " + " | ".join(LANG_NAMES[l] for l in langs) + " |"
    rows = [head, "|---" * (len(langs) + 1) + "|"]
    for q in QUESTION_ORDER:
        if q not in pq:
            continue
        cells = []
        for l in langs:
            v = pq[q].get(l)
            cells.append("--" if v is None else f"{v:+.2f}")
        rows.append(f"| `{q}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def table_refusals(ref):
    rows = ["| Language | Unparsed | Refusal | Prose | Refusal on positive | "
            "Refusal on negative |",
            "|---|---|---|---|---|---|"]
    for l in _langs(ref):
        v = ref[l]
        rows.append(
            f"| {LANG_NAMES[l]} | {v['unparsed']:.1%} | **{v['refusal']:.1%}** | "
            f"{v['prose'] + v['junk']:.1%} | {v['refusal_positive']:.1%} | "
            f"{v['refusal_negative']:.1%} |")
    return "\n".join(rows)


def table_robustness(rob):
    rows = ["| Language | Observed gap | Worst case | Best case | Survives? |",
            "|---|---|---|---|---|"]
    for l in _langs(rob):
        v = rob[l]
        ok = "yes" if v["worst"] > 0 else "**no**"
        rows.append(f"| {LANG_NAMES[l]} | {v['observed']:+.2f} | "
                    f"{v['worst']:+.2f} | {v['best']:+.2f} | {ok} |")
    return "\n".join(rows)


def main():
    instrument = load("step1_4_instrument.jsonl")
    if not instrument:
        sys.exit("no step1_4_instrument.jsonl")

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        t = report_instrument(instrument)
        rob = report_gap_robustness(t)
        from analyze import report_distribution
        dist = report_distribution(instrument)
        ref = report_refusals(instrument)
        pq = report_per_question(instrument)

    out = {
        "4.1 instrument": table_instrument(t, ref),
        "4.2 distribution": table_distribution(dist),
        "4.3 per-question": table_per_question(pq),
        "4.4 refusals": table_refusals(ref),
        "4.4 robustness": table_robustness(rob),
    }
    dest = ROOT / "report" / "tables.md"
    body = "\n\n".join(f"### {k}\n\n{v}" for k, v in out.items())
    dest.write_text(body + "\n")
    print(body)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()

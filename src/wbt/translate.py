"""Translation clients. Gemini is the main pass, OpenAI the agreement check."""

import json
import os
import re
import time
import urllib.error
import urllib.request

GEMINI_MODEL = "gemini-3.7-flash"
OPENAI_MODEL = "gpt-5"

LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "zh": "Chinese (Simplified)",
    "hi": "Hindi",
    "ar": "Arabic",
    "ur": "Urdu",
    "sw": "Swahili",
}

SYSTEM = """You are translating materials for a psychometric instrument. Accuracy of \
measurement language matters more than fluency.

Rules:
- Translate into {lang}. Output {lang} only, in that language's native script.
- Numerals stay as Western Arabic digits (1-7), never spelled out, never localised.
- Rating-scale labels are load-bearing. "1 = very unhappy" through "7 = very happy" \
must keep their intensity ordering and even spacing in {lang}. Do not compress, \
intensify, or soften any level.
- Preserve the register and emotional intensity of the source exactly. Do not tone \
down distressing content; do not embellish positive content.
- Keep formatting, line breaks, and any "Developer Message:" prefix.
- Translate meaning, not words. Do not add explanation or commentary.

Return a JSON object mapping each input id to its translation. Nothing else."""

BACK_SYSTEM = """Translate each input into English. Produce a literal, faithful \
back-translation that exposes any meaning drift in the source: do not repair \
awkward phrasing, and do not consult your memory of what the original English \
probably said.

Return a JSON object mapping each input id to its English translation. Nothing else."""


def _post(url, payload, headers, timeout=180, retries=4):
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            last = f"HTTP {e.code}: {detail}"
            if e.code in (400, 401, 403):
                raise RuntimeError(last)
        except Exception as e:
            last = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} attempts: {last}")


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def gemini(system, items):
    key = os.environ["GEMINI_API_KEY"]
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(items, ensure_ascii=False)}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "maxOutputTokens": 32000,
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }
    r = _post(url, payload, {"Content-Type": "application/json"})
    cand = r["candidates"][0]
    parts = cand.get("content", {}).get("parts")
    if not parts:
        raise RuntimeError(f"no parts, finishReason={cand.get('finishReason')}")
    return _extract_json("".join(p.get("text", "") for p in parts))


def openai(system, items):
    key = os.environ["OPENAI_API_KEY"]
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "developer", "content": system},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
        ],
        "text": {"format": {"type": "json_object"}},
        "reasoning": {"effort": "low"},
        "max_output_tokens": 32000,
    }
    r = _post(
        "https://api.openai.com/v1/responses",
        payload,
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    chunks = [
        c.get("text", "")
        for out in r.get("output", [])
        if out.get("type") == "message"
        for c in out.get("content", [])
    ]
    return _extract_json("".join(chunks))

"""Checks how much Groq quota is realistically left for each model - not just "not zero".

WHY NOT A 1-TOKEN PROBE: a request asking for 1 token only proves you have AT LEAST 1 token
left. It says "OK" even with almost nothing left, which is exactly what happened earlier:
Groq's real error later showed "Limit 200000, Used 198448" (only 1552 left), but a 1-token
check had reported "OK" moments before, because 1552 > 1.

THE FIX: ask for a `max_tokens` ceiling close to what a REAL CodeAtlas question actually
costs (we've observed 700-3500 tokens per request in real runs; PROBE_SIZE below matches
that). Two things make this safe and free even when it fails:
  1. Groq checks the limit BEFORE generating anything, using max_tokens as the worst-case
     reservation - so a rejection costs nothing, and the error reveals the real Limit/Used.
  2. If it succeeds, the model still only writes a short natural reply to "hi" and stops on
     its own; max_tokens is a ceiling, not a target, so it doesn't actually spend 3500 tokens.
This means a "PROBE OK" result is a real, meaningful assurance ("this model can handle at
least one more typical question"), not just "technically nonzero."

Run from backend/ (venv active):
    python scripts/check_quota.py
"""
import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
PROBE_SIZE = 3500  # tokens: matches the upper end of what one real CodeAtlas question costs

_NUMS_RE = re.compile(r"Limit (\d+), Used (\d+), Requested (\d+)")
_WAIT_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.I)
_WINDOW_RE = re.compile(r"tokens per (day|minute)", re.I)


def _format_wait(text: str) -> str:
    m = _WAIT_RE.search(text)
    if not m:
        return "unknown"
    total = float(m.group(1) or 0) * 60 + float(m.group(2))
    if total >= 3600:
        return f"{total / 3600:.1f} hours"
    if total >= 60:
        return f"{total / 60:.1f} minutes"
    return f"{total:.0f} seconds"


def check(client: Groq, model: str) -> str:
    try:
        client.chat.completions.create(
            model=model, max_tokens=PROBE_SIZE, messages=[{"role": "user", "content": "hi"}]
        )
        return f"PROBE OK - can handle at least ~{PROBE_SIZE} more tokens right now"
    except Exception as e:  # noqa: BLE001
        text = str(e)
        nums, window = _NUMS_RE.search(text), _WINDOW_RE.search(text)
        if not nums:
            return f"error (not a rate limit): {text[:150]}"
        limit, used, requested = (int(x) for x in nums.groups())
        available = max(0, limit - used)
        window_label = f"per {window.group(1)}" if window else "window unclear"
        return (f"available: {available} tokens; limit: {limit} tokens ({window_label}); "
                f"resets in: {_format_wait(text)}  [this probe asked for {requested}]")


if __name__ == "__main__":
    key = os.getenv("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY is missing from .env")
    client = Groq(api_key=key)
    for m in MODELS:
        print(f"{m:<24}:  {check(client, m)}")

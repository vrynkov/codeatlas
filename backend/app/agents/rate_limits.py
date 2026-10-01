"""Shared logic for reading Groq's rate-limit error messages. One place, so react_agent.py
(short per-minute waits) and llm.py (model fallback) always agree on what "daily" means."""
import re

_WAIT_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.I)


def is_daily_limit(text: str) -> bool:
    """True for Groq's per-DAY token limit specifically (TPD) - the kind that needs
    minutes/hours to clear, as opposed to a per-minute burst that clears in seconds."""
    return "tokens per day" in text.lower()


def is_rate_limit(text: str) -> bool:
    return "429" in text or "rate_limit" in text


def suggested_wait(text: str) -> float | None:
    m = _WAIT_RE.search(text)
    if not m:
        return None
    return float(m.group(1) or 0) * 60 + float(m.group(2))

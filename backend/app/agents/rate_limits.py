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


def is_connection_error(text: str) -> bool:
    """A genuine network-level hiccup (DNS, TCP reset, a momentary timeout reaching Groq at
    all) rather than Groq actually responding with a rate limit or any other real error. We
    deliberately disabled the SDK's own built-in retries (see llm.py's max_retries=0) so our
    OWN rate-limit handling would see every 429 immediately instead of it being silently
    retried away - but that also means an ordinary, one-off connection blip now gets NO retry
    at all unless we handle it here specifically. Worth a short, cheap retry, not a hard fail."""
    t = text.lower()
    return any(s in t for s in (
        "connection error", "connection reset", "connection aborted", "connection refused",
        "timed out", "timeout", "temporarily unavailable", "network is unreachable",
        "name or service not known", "failed to establish a new connection",
    ))


def suggested_wait(text: str) -> float | None:
    m = _WAIT_RE.search(text)
    if not m:
        return None
    return float(m.group(1) or 0) * 60 + float(m.group(2))

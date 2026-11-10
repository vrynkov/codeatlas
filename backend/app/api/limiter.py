"""A tiny, free, in-memory rate limiter ("only N requests per visitor per minute/day").

How it works (a "sliding window"): for each visitor + action we keep the times of their recent
requests. A new request is allowed only if fewer than N of those times fall in the last W seconds.
Example with "3 per minute": requests at 0s, 10s, 20s are allowed; one at 30s is refused; at 61s
the 0s request has slid out of the window, so a new one is allowed again.

Honest limits of this approach:
  - It lives in this process's memory, so it only works because the app runs as ONE instance
    (--max-instances=1). With several instances each would count separately.
  - It resets when the instance restarts (Cloud Run scales to zero when idle).
  - Someone with many IP addresses gets many allowances. Groq's own daily quota is the final cap.
For a portfolio demo those are fine trade-offs: zero cost, zero extra services.
"""
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

from ..config import TRUSTED_PROXY_HOPS

MAX_TRACKED_KEYS = 20_000  # memory guard: forget the stalest visitors beyond this


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self._clock = clock            # injectable, so tests can fake the passing of time
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()  # FastAPI runs sync endpoints in several threads at once

    def check(self, key: str, limits: list[tuple[int, int]]) -> int | None:
        """limits = [(max_requests, window_seconds), ...]. Records the request and returns None if
        it is allowed; otherwise records nothing and returns how many seconds until it would be."""
        now = self._clock()
        longest = max(w for _, w in limits)
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= longest:
                hits.popleft()  # too old to matter for any window
            for max_requests, window in limits:
                recent = [t for t in hits if now - t < window]
                if len(recent) >= max_requests:
                    return max(1, int(window - (now - recent[0])) + 1)
            hits.append(now)
            if len(self._hits) > MAX_TRACKED_KEYS:
                self._forget_stalest()
            return None

    def _forget_stalest(self) -> None:
        stalest = sorted(self._hits, key=lambda k: self._hits[k][-1] if self._hits[k] else 0)
        for k in stalest[: len(stalest) // 10 or 1]:
            del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()


def client_ip(request: Request) -> str:
    """The visitor's IP address. Behind Cloud Run, request.client.host is Google's front end, and
    the real visitor IP is the LAST entry Google added to X-Forwarded-For. Entries further left were
    sent by the visitor themselves and could be faked to dodge the limit, so we never use them."""
    xff = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if xff and TRUSTED_PROXY_HOPS > 0:
        return xff[-min(TRUSTED_PROXY_HOPS, len(xff))]
    return request.client.host if request.client else "unknown"


def enforce(request: Request, action: str, limits: list[tuple[int, int]]) -> None:
    """Raise HTTP 429 ("Too Many Requests") if this visitor is over the limit for `action`."""
    wait = limiter.check(f"{action}:{client_ip(request)}", limits)
    if wait is not None:
        raise HTTPException(429, f"Too many requests. Please wait about {_human(wait)} and try again.",
                            headers={"Retry-After": str(wait)})


def _human(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} seconds"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes"
    return f"{round(seconds / 3600)} hours"

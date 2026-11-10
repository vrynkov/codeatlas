"""The in-memory rate limiter, with a fake clock so 'a day later' takes no real time."""
from types import SimpleNamespace

from app.api import limiter as limiter_mod
from app.api.limiter import RateLimiter, client_ip


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_allows_up_to_the_limit_then_refuses_with_a_wait_time():
    clock = Clock()
    rl = RateLimiter(clock)
    for _ in range(3):
        assert rl.check("ip1", [(3, 60)]) is None
    wait = rl.check("ip1", [(3, 60)])
    assert wait is not None and 1 <= wait <= 61


def test_window_slides():
    clock = Clock()
    rl = RateLimiter(clock)
    for _ in range(3):
        rl.check("ip1", [(3, 60)])
    clock.t += 61
    assert rl.check("ip1", [(3, 60)]) is None


def test_visitors_are_counted_separately():
    rl = RateLimiter(Clock())
    for _ in range(3):
        rl.check("ip1", [(3, 60)])
    assert rl.check("ip2", [(3, 60)]) is None


def test_daily_cap_applies_even_when_each_minute_is_fine():
    clock = Clock()
    rl = RateLimiter(clock)
    limits = [(3, 60), (5, 86400)]
    for _ in range(5):
        assert rl.check("ip1", limits) is None
        clock.t += 120
    assert rl.check("ip1", limits) is not None
    clock.t += 86400
    assert rl.check("ip1", limits) is None


def test_refused_requests_do_not_count():
    """Hammering after being refused must not push the wait further and further out."""
    clock = Clock()
    rl = RateLimiter(clock)
    for _ in range(3):
        rl.check("ip1", [(3, 60)])
    for _ in range(50):
        rl.check("ip1", [(3, 60)])
    clock.t += 61
    assert rl.check("ip1", [(3, 60)]) is None


def test_memory_is_bounded(monkeypatch):
    monkeypatch.setattr(limiter_mod, "MAX_TRACKED_KEYS", 100)
    rl = RateLimiter(Clock())
    for i in range(500):
        rl.check(f"ip{i}", [(1, 60)])
    assert len(rl._hits) <= 101


def _req(xff=None, host="10.0.0.1"):
    headers = {"x-forwarded-for": xff} if xff else {}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=host))


def test_client_ip_uses_the_entry_added_by_cloud_run_not_the_one_the_visitor_wrote():
    assert client_ip(_req("6.6.6.6, 203.0.113.9")) == "203.0.113.9"
    assert client_ip(_req("203.0.113.9")) == "203.0.113.9"
    assert client_ip(_req()) == "10.0.0.1"

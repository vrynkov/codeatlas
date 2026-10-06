"""Creates the language model(s) CodeAtlas uses, with automatic fallback across a chain of
Groq models. Two independent kinds of trouble are handled differently, on purpose:

  - A DAILY quota exhaustion is permanent for the rest of the day no matter how long we wait,
    so we remember it (in data/model_fallback_state.json, surviving across separate process
    runs too) and never try that model again today.
  - A SHORT-lived limit (e.g. a per-minute burst) is NOT a reason to give up on a model for
    the whole day - a different model's separate per-minute bucket is usually just sitting
    idle, so we try that instead, right now. This is deliberately NOT persisted: the very next
    question should still try the original model first, since it likely recovers within
    seconds to a minute.

Which model actually answered a given call (for reporting - the eval harness, the UI's live
trace) is tracked separately from which models are confirmed dead for the day, since a
transient reroute and a permanent one need to be visible differently: a transient one should
not make every future question start from the rerouted-to model.
"""
import json
import os
import sys
import time
from pathlib import Path

from ..config import BACKEND_DIR, GROQ_API_KEY, GROQ_MODEL
from .rate_limits import is_daily_limit, is_rate_limit, suggested_wait

# The account's other real models, in the order to fall back through. GROQ_MODEL (from .env)
# is always tried first; override the whole chain with GROQ_MODEL_CHAIN="a,b,c" in .env if needed.
_DEFAULT_CHAIN = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
_env_chain = os.getenv("GROQ_MODEL_CHAIN", "")
if _env_chain.strip():
    MODEL_CHAIN = [m.strip() for m in _env_chain.split(",") if m.strip()]
else:
    MODEL_CHAIN = [GROQ_MODEL] + [m for m in _DEFAULT_CHAIN if m != GROQ_MODEL]

_STATE_FILE = BACKEND_DIR / "data" / "model_fallback_state.json"


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _load_state() -> dict:
    """Only tracks PERMANENT (daily) exhaustion - "active" is where to START searching from
    (skip past models already confirmed dead today), "dead" is the confirmed-dead set itself."""
    try:
        data = json.loads(_STATE_FILE.read_text())
        if data.get("date") == _today():
            return {"active": data.get("active", 0), "dead": set(data.get("dead", []))}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"active": 0, "dead": set()}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(
            {"date": _today(), "active": state["active"], "dead": sorted(state["dead"])}))
    except OSError:
        pass  # best-effort only; a failed write just means we might retry a dead model once more


_STATE = _load_state()  # persistent (daily-dead tracking), shared by every instance in this process
_LAST_USED = {"idx": _STATE["active"]}  # per-call reporting only, NEVER written to disk - a
                                        # transient reroute updates this so reporting is accurate,
                                        # but must not redirect where FUTURE calls start from


def _make_client(model: str, temperature: float):
    """Separated out so tests can monkeypatch this one function instead of the whole SDK.
    max_retries=0 is deliberate: the underlying SDK retries 429s silently on its own by
    default (2 extra tries, several seconds each), which delayed our OWN fallback logic from
    ever seeing the error. We do our own, smarter retrying/switching in react_agent.py and
    this file instead, so the SDK shouldn't retry underneath us.
    timeout=120 is also deliberate: the groq SDK's own default read timeout is only 60
    seconds, which a long, detailed answer to a broad question (especially generated under
    Cloud Run's single default vCPU) can genuinely exceed - that surfaces as a generic,
    unhelpful "Connection error.", identically on every attempt, since retrying the same
    call under the same tight ceiling just hits the same wall again."""
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=temperature, max_retries=0, timeout=120)


class MultiModelChatGroq:
    """Looks and acts like a single LangChain chat model (.invoke(), .bind_tools()) but is
    really a small pool of them, falling forward across daily exhaustion AND short-lived
    per-model limits - see the module docstring for why those two are handled differently."""

    MIN_WAIT_TO_REROUTE = 10  # seconds. Below this, just pause on the SAME model - rerouting
                              # has a real cost (different models write citations and structure
                              # answers a little differently; hopping between them repeatedly
                              # within one question measurably hurt citation consistency in
                              # testing), so it's only worth paying for waits long enough to matter.
    MAX_QUICK_RETRIES = 3     # caps same-model short waits so a persistent run of them still
                              # eventually falls through to rerouting instead of waiting forever.

    def __init__(self, models: list[str] | None = None, temperature: float = 0, tools=None):
        self.models = models or MODEL_CHAIN
        self.temperature = temperature
        self.tools = tools

    def bind_tools(self, tools, **kw):
        return MultiModelChatGroq(self.models, self.temperature, tools=tools)

    def _client_for(self, idx: int):
        client = _make_client(self.models[idx], self.temperature)
        return client.bind_tools(self.tools) if self.tools else client

    def _next_viable(self, after_idx: int) -> int | None:
        """The next index after `after_idx` that isn't already confirmed dead for today."""
        idx = after_idx + 1
        while idx < len(self.models) and idx in _STATE["dead"]:
            idx += 1
        return idx if idx < len(self.models) else None

    def invoke(self, messages, **kwargs):
        last_err = None
        idx = _STATE["active"]
        while idx < len(self.models):
            if idx in _STATE["dead"]:
                idx += 1
                continue

            quick_tries = 0
            while True:  # may retry THIS SAME idx a bounded number of times before moving on
                try:
                    result = self._client_for(idx).invoke(messages, **kwargs)
                    if idx != _LAST_USED["idx"]:
                        print(f"[MODEL] now using {self.models[idx]}", file=sys.stderr)
                        _emit_switch(self.models[idx])
                    _LAST_USED["idx"] = idx  # for reporting only - see module docstring
                    return result
                except Exception as e:  # noqa: BLE001
                    text = str(e)
                    daily = is_daily_limit(text)
                    if not daily and not is_rate_limit(text):
                        raise  # a real error, not a rate limit - nothing to route around

                    next_idx = self._next_viable(idx)
                    wait = None if daily else suggested_wait(text)
                    if (not daily and wait is not None and wait <= self.MIN_WAIT_TO_REROUTE
                            and quick_tries < self.MAX_QUICK_RETRIES):
                        quick_tries += 1
                        print(f"[MODEL] {self.models[idx]} hit a brief limit; waiting {wait:.0f}s "
                              f"on the same model", file=sys.stderr)
                        time.sleep(wait)
                        continue  # inner while True -> correctly retries the SAME idx

                    if daily:
                        # Confirmed exhausted for the whole day - a dead end no matter how long
                        # we wait, so this DOES persist (across questions, even across separate
                        # process runs via the state file): never try this model again today.
                        _STATE["dead"].add(idx)
                        _STATE["active"] = idx + 1
                        _save_state(_STATE)
                        next_name = self.models[next_idx] if next_idx is not None else "no more models configured"
                        print(f"[MODEL] {self.models[idx]} exhausted for today; trying the next model {next_name}",
                              file=sys.stderr)
                    elif next_idx is not None:
                        # Short-lived limit worth rerouting for (longer than MIN_WAIT_TO_REROUTE),
                        # not a daily one - try another model's separate bucket for just this
                        # call, WITHOUT persisting: next time should still try this model first.
                        mins = f"{wait:.0f}s" if wait else "a bit"
                        print(f"[MODEL] {self.models[idx]} hit a short-term limit (retry in ~{mins}); "
                              f"trying {self.models[next_idx]} for this request instead", file=sys.stderr)
                    elif wait is None:
                        # No parseable "try again in Xs" - this ISN'T a normal throttle that
                        # clears with time (e.g. Groq's "request too large for this model's
                        # per-minute output limit" error: the SAME request will fail identically
                        # no matter how long we wait on the SAME model). No other model is
                        # available right now, so say that plainly instead of letting the raw
                        # Groq JSON reach react_agent's wait-and-retry, which would otherwise
                        # burn several pointless 20s retries on a request that can never succeed.
                        raise RuntimeError(
                            f"{self.models[idx]} couldn't handle this specific request right now "
                            f"(likely too large for its per-minute output limit), and no other "
                            f"configured model is available. Try a shorter or more specific question."
                        ) from e
                    else:
                        raise  # a genuine short-term throttle WITH a real wait time, just no
                               # other model to reroute to - let react_agent's own short-wait-
                               # and-retry handle it, since waiting here legitimately can help
                    last_err = e
                    break  # leave the inner while, the outer while advances to next_idx below

            idx = next_idx if next_idx is not None else len(self.models)

        # We get here in two very different situations, and the error must say which one:
        if not self.models:
            # A genuine setup mistake - an empty GROQ_MODEL_CHAIN in .env, or similar.
            raise RuntimeError("No Groq models configured (MODEL_CHAIN is empty). "
                               "Check GROQ_MODEL / GROQ_MODEL_CHAIN in your .env file.")
        # Every model in the chain is legitimately exhausted for today (this call may not have
        # tried any of them itself, if an EARLIER call already used them all up - that's why we
        # can't just re-raise `last_err`, which would be None in that case).
        raise RuntimeError(
            f"All {len(self.models)} configured Groq model(s) are exhausted for today: "
            f"{', '.join(self.models)}. Try again tomorrow, or add another model via "
            f"GROQ_MODEL_CHAIN in .env."
        ) from last_err


def active_model_name() -> str:
    """Whichever model most recently answered a call, in THIS process. Used for reporting only -
    never for deciding where the next call should start searching from (see _STATE for that)."""
    idx = _LAST_USED["idx"]
    return MODEL_CHAIN[idx] if idx < len(MODEL_CHAIN) else MODEL_CHAIN[-1]


def _emit_switch(model: str) -> None:
    """Also report a switch as a normal team event (not just to stderr), so anything reading
    run_team()'s event stream - like the eval harness - can record which model answered what.
    A no-op outside a LangGraph run (e.g. the standalone `python -m app.agents.react_agent` CLI)."""
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()({"agent": "model", "type": "switch", "text": model})
    except Exception:  # noqa: BLE001  not inside a graph run right now - the stderr print still happened
        pass


def get_llm(temperature: float = 0) -> MultiModelChatGroq:
    return MultiModelChatGroq(temperature=temperature)

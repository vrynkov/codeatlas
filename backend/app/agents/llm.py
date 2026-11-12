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
import time
from collections import Counter
from pathlib import Path

from opentelemetry import trace

from ..config import BACKEND_DIR, GROQ_API_KEY, GROQ_MODEL
from ..logging_setup import get_logger
from .rate_limits import is_daily_limit, is_rate_limit, suggested_wait

log = get_logger(__name__)
_tracer = trace.get_tracer(__name__)  # does nothing unless tracing_setup switched tracing on

# The account's other real models, in the order to fall back through. GROQ_MODEL (from .env)
# is always tried first; override the whole chain with GROQ_MODEL_CHAIN="a,b,c" in .env if needed.
_DEFAULT_CHAIN = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
_env_chain = os.getenv("GROQ_MODEL_CHAIN", "")
if _env_chain.strip():
    MODEL_CHAIN = [m.strip() for m in _env_chain.split(",") if m.strip()]
else:
    MODEL_CHAIN = [GROQ_MODEL] + [m for m in _DEFAULT_CHAIN if m != GROQ_MODEL]

# The CRITIC (the agent that grades answers) should NOT be the same model that wrote the answer:
# a model grading its own work is too forgiving ("self-grading"). So the critic starts from a
# DIFFERENT model - by default the chain rotated by one (e.g. qwen first instead of gpt-oss-120b) -
# and only falls back to the others if that one is rate limited. The eval reports when that
# fallback made the critic and the writer the same model, so it's never hidden.
# Override with CRITIC_MODEL_CHAIN="a,b,c" in .env.
_env_critic = os.getenv("CRITIC_MODEL_CHAIN", "")
if _env_critic.strip():
    CRITIC_MODEL_CHAIN = [m.strip() for m in _env_critic.split(",") if m.strip()]
else:
    CRITIC_MODEL_CHAIN = MODEL_CHAIN[-1:] + MODEL_CHAIN[:-1] if len(MODEL_CHAIN) > 1 else list(MODEL_CHAIN)

_STATE_FILE = BACKEND_DIR / "data" / "model_fallback_state.json"


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _load_state() -> dict:
    """Only tracks PERMANENT (daily) exhaustion: the set of model NAMES confirmed dead today.
    Names, not positions in a list - the writer and the critic use differently ordered chains,
    so "model #0 is dead" would mean a different model to each of them."""
    try:
        data = json.loads(_STATE_FILE.read_text())
        if data.get("date") == _today():
            # isinstance check: older versions of this file stored list positions (ints) - ignore those
            return {"dead": {m for m in data.get("dead", []) if isinstance(m, str)}}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"dead": set()}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps({"date": _today(), "dead": sorted(state["dead"])}))
    except OSError:
        pass  # best-effort only; a failed write just means we might retry a dead model once more


_STATE = _load_state()  # persistent (daily-dead tracking), shared by every instance in this process
# Which model most recently answered a WRITER call (reporting only, never written to disk). The
# critic's calls don't touch this, so "which model answered" never reports the critic's model.
_LAST_USED = {"model": next((m for m in MODEL_CHAIN if m not in _STATE["dead"]), MODEL_CHAIN[-1] if MODEL_CHAIN else "")}


# Running totals of every SUCCESSFUL model call in this process (writer AND critic), for reporting
# only: the eval reads them before and after a question to get "tokens and seconds per question".
# A call that fails (a 429, an invalid tool call) isn't counted, so these are a slight UNDERcount.
_USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "llm_seconds": 0.0}
_CALLS_BY_MODEL: Counter = Counter()


def usage_snapshot() -> dict:
    return {**_USAGE, "by_model": Counter(_CALLS_BY_MODEL)}


def usage_since(before: dict) -> dict:
    """What was used since `before` (from usage_snapshot()): calls, tokens, model-busy seconds,
    and how many calls went to each model."""
    out = {k: _USAGE[k] - before[k] for k in _USAGE}
    out["llm_seconds"] = round(out["llm_seconds"], 2)
    out["tokens"] = out["input_tokens"] + out["output_tokens"]
    out["by_model"] = dict(_CALLS_BY_MODEL - before["by_model"])
    return out


def _record_usage(model: str, usage: dict, seconds: float) -> None:
    _USAGE["calls"] += 1
    _USAGE["input_tokens"] += usage.get("input_tokens") or 0
    _USAGE["output_tokens"] += usage.get("output_tokens") or 0
    _USAGE["llm_seconds"] += seconds
    _CALLS_BY_MODEL[model] += 1


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

    def __init__(self, models: list[str] | None = None, temperature: float = 0, tools=None,
                 report: bool = True):
        self.models = models or MODEL_CHAIN
        self.temperature = temperature
        self.tools = tools
        self.report = report      # False for the critic: its model must not show up as "who answered"
        self.last_model = None    # the model that answered THIS instance's most recent call

    def bind_tools(self, tools, **kw):
        return MultiModelChatGroq(self.models, self.temperature, tools=tools, report=self.report)

    def avoiding(self, model: str | None) -> "MultiModelChatGroq":
        """A copy that tries `model` LAST. The critic uses this so it never starts on the model that
        just wrote the answer - even when the writer itself had fallen back to another model."""
        if not model or model not in self.models:
            return self
        order = [m for m in self.models if m != model] + [model]
        return MultiModelChatGroq(order, self.temperature, tools=self.tools, report=self.report)

    def _client_for(self, idx: int):
        client = _make_client(self.models[idx], self.temperature)
        return client.bind_tools(self.tools) if self.tools else client

    def _timed_invoke(self, idx: int, messages, **kwargs):
        """One call to Groq, logged (model, time, tokens) and traced as its own span. Token counts
        are what the free tier limits, so PagePal can see which step uses up the quota."""
        model = self.models[idx]
        start = time.perf_counter()
        with _tracer.start_as_current_span("llm_call") as span:
            span.set_attribute("llm.model", model)
            try:
                result = self._client_for(idx).invoke(messages, **kwargs)
            except Exception as e:
                log.info("llm_call_failed", model=model, error_type=type(e).__name__,
                         duration_ms=round((time.perf_counter() - start) * 1000))
                raise
            usage = getattr(result, "usage_metadata", None) or {}
            span.set_attribute("llm.input_tokens", usage.get("input_tokens", 0))
            span.set_attribute("llm.output_tokens", usage.get("output_tokens", 0))
        _record_usage(model, usage, time.perf_counter() - start)
        log.info("llm_call", model=model, duration_ms=round((time.perf_counter() - start) * 1000),
                 input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                 tool_calls=len(getattr(result, "tool_calls", None) or []))
        return result

    def _next_viable(self, after_idx: int) -> int | None:
        """The next index after `after_idx` that isn't already confirmed dead for today."""
        idx = after_idx + 1
        while idx < len(self.models) and self.models[idx] in _STATE["dead"]:
            idx += 1
        return idx if idx < len(self.models) else None

    def invoke(self, messages, **kwargs):
        last_err = None
        idx = 0
        while idx < len(self.models):
            if self.models[idx] in _STATE["dead"]:
                idx += 1
                continue

            quick_tries = 0
            while True:  # may retry THIS SAME idx a bounded number of times before moving on
                try:
                    result = self._timed_invoke(idx, messages, **kwargs)
                    self.last_model = self.models[idx]
                    if self.report:
                        if self.models[idx] != _LAST_USED["model"]:
                            log.info("model_switched", model=self.models[idx])
                            _emit_switch(self.models[idx])
                        _LAST_USED["model"] = self.models[idx]  # for reporting only - see module docstring
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
                        log.warning("rate_limited", model=self.models[idx], wait_s=round(wait),
                                    action="wait_same_model", attempt=quick_tries)
                        time.sleep(wait)
                        continue  # inner while True -> correctly retries the SAME idx

                    if daily:
                        # Confirmed exhausted for the whole day - a dead end no matter how long
                        # we wait, so this DOES persist (across questions, even across separate
                        # process runs via the state file): never try this model again today.
                        _STATE["dead"].add(self.models[idx])
                        _save_state(_STATE)
                        next_name = self.models[next_idx] if next_idx is not None else "no more models configured"
                        log.warning("model_fallback", model=self.models[idx], next_model=next_name,
                                    reason="daily_limit")
                    elif next_idx is not None:
                        # Short-lived limit worth rerouting for (longer than MIN_WAIT_TO_REROUTE),
                        # not a daily one - try another model's separate bucket for just this
                        # call, WITHOUT persisting: next time should still try this model first.
                        log.warning("model_fallback", model=self.models[idx], next_model=self.models[next_idx],
                                    reason="short_term_limit", wait_s=round(wait) if wait else None)
                    elif wait is None:
                        log.error("request_too_large", model=self.models[idx])
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
        log.error("all_models_exhausted", models=self.models)
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
    return _LAST_USED["model"]


def _emit_switch(model: str) -> None:
    """Also report a switch as a normal team event (not just to the logs), so anything reading
    run_team()'s event stream - like the eval harness - can record which model answered what.
    A no-op outside a LangGraph run (e.g. the standalone `python -m app.agents.react_agent` CLI)."""
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()({"agent": "model", "type": "switch", "text": model})
    except Exception:  # noqa: BLE001  not inside a graph run right now - the log line was still written
        pass


def get_llm(temperature: float = 0) -> MultiModelChatGroq:
    return MultiModelChatGroq(temperature=temperature)


def get_critic_llm() -> MultiModelChatGroq:
    """The critic's own model pool: starts from a different model than the writer (see
    CRITIC_MODEL_CHAIN above) and never changes what active_model_name() reports."""
    return MultiModelChatGroq(models=CRITIC_MODEL_CHAIN, temperature=0, report=False)

"""Creates the language model(s) CodeAtlas uses, with automatic fallback across a chain of
Groq models. When the active model's DAILY quota runs out, we switch to the next model in
the chain and carry on with the same question - no manual .env editing needed mid-run.

Which model is "dead" for today is remembered in a small file (data/model_fallback_state.json),
so restarting a script (a new eval run, a new CLI question) picks up where the last one left
off instead of wasting a call re-discovering the same exhausted model. The file resets itself
once the calendar date changes (a simplifying assumption - Groq's real reset may be a rolling
24h window, not midnight, so this can occasionally be a little optimistic or conservative).
"""
import json
import sys
import time
from pathlib import Path

from ..config import BACKEND_DIR, GROQ_API_KEY, GROQ_MODEL
from .rate_limits import is_daily_limit

# The account's other real models, in the order to fall back through. GROQ_MODEL (from .env)
# is always tried first; override the whole chain with GROQ_MODEL_CHAIN="a,b,c" in .env if needed.
import os  # noqa: E402

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


_STATE = _load_state()  # shared by every MultiModelChatGroq in this process


def _make_client(model: str, temperature: float):
    """Separated out so tests can monkeypatch this one function instead of the whole SDK.
    max_retries=0 is deliberate: the underlying SDK retries 429s silently on its own by
    default (2 extra tries, several seconds each), which delayed our OWN fallback logic from
    ever seeing the error. We do our own, smarter retrying/switching in react_agent.py and
    this file instead, so the SDK shouldn't retry underneath us."""
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=temperature, max_retries=0)


class MultiModelChatGroq:
    """Looks and acts like a single LangChain chat model (.invoke(), .bind_tools()) but is
    really a small pool of them. On a daily-quota error it marks that model dead for today
    and transparently retries the SAME call on the next one - the caller never sees the error
    unless every model in the chain is exhausted."""

    def __init__(self, models: list[str] | None = None, temperature: float = 0, tools=None):
        self.models = models or MODEL_CHAIN
        self.temperature = temperature
        self.tools = tools

    def bind_tools(self, tools, **kw):
        return MultiModelChatGroq(self.models, self.temperature, tools=tools)

    def _client_for(self, idx: int):
        client = _make_client(self.models[idx], self.temperature)
        return client.bind_tools(self.tools) if self.tools else client

    def invoke(self, messages, **kwargs):
        last_err = None
        for idx in range(_STATE["active"], len(self.models)):
            if idx in _STATE["dead"]:
                continue
            try:
                result = self._client_for(idx).invoke(messages, **kwargs)
                if idx != _STATE["active"]:
                    print(f"[MODEL] now using {self.models[idx]}", file=sys.stderr)
                    _emit_switch(self.models[idx])
                _STATE["active"] = idx
                _save_state(_STATE)
                return result
            except Exception as e:  # noqa: BLE001
                if not is_daily_limit(str(e)):
                    raise  # not a daily-quota problem (e.g. a per-minute burst) - let the
                           # caller's own short-wait retry logic (react_agent.py) handle it
                _STATE["dead"].add(idx)
                next_idx = idx + 1
                while next_idx < len(self.models) and next_idx in _STATE["dead"]:
                    next_idx += 1  # skip any model already known dead, name the real next attempt
                next_name = self.models[next_idx] if next_idx < len(self.models) else "no more models configured"
                print(f"[MODEL] {self.models[idx]} exhausted for today; trying the next model {next_name}",
                      file=sys.stderr)
                _STATE["active"] = idx + 1
                _save_state(_STATE)
                last_err = e

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
    """Whichever model most recently answered a call, in THIS process. Used for reporting only."""
    return MODEL_CHAIN[_STATE["active"]] if _STATE["active"] < len(MODEL_CHAIN) else MODEL_CHAIN[-1]


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

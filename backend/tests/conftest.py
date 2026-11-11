"""Shared test setup. The whole suite runs OFFLINE and FREE: no Groq calls, no GitHub, no embedding
model download. Anything that would touch the network is replaced by a small fake ("mock")."""
import os

# Must be set before the app is imported (config.py reads them at import time).
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LOG_FORMAT", "console")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.pop("GROQ_MODEL_CHAIN", None)
os.environ.pop("CRITIC_MODEL_CHAIN", None)

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402


class FakeLLM:
    """Stands in for the Groq model. It answers by ROLE: every system prompt starts with a tag like
    [ROLE:CRITIC], so `replies={"CRITIC": ['{"score": 5}']}` scripts what the critic says. Replies
    are used in order; the last one repeats. Every call is recorded in `.calls` for checking."""

    def __init__(self, replies: dict[str, list[str]], name: str = "fake-writer"):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[tuple[str, str]] = []  # (role, full user text)
        self.last_model = name

    def bind_tools(self, tools, **kw):
        return self

    def invoke(self, messages, **kw):
        system = getattr(messages[0], "content", "")
        role = system.split("]", 1)[0].removeprefix("[ROLE:") if system.startswith("[ROLE:") else "OTHER"
        self.calls.append((role, messages[-1].content))
        options = self.replies.get(role) or ["(no scripted reply)"]
        text = options.pop(0) if len(options) > 1 else options[0]
        return AIMessage(content=text)

    def roles(self) -> list[str]:
        return [r for r, _ in self.calls]


class FakeLessonStore:
    """In-memory replacement for the Chroma-backed LessonStore (no embedding model needed)."""

    def __init__(self):
        self.saved: list[dict] = []
        self.to_recall: list[dict] = []

    def add(self, question, lesson, kind):
        self.saved.append({"question": question, "lesson": lesson, "kind": kind})
        return f"{kind}-1"

    def recall(self, question):
        return list(self.to_recall)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test starts clean: no leftover rate-limit counters, dead models or answers,
    and nothing is written into the real backend/data folder."""
    from app.agents import llm
    from app.api import limiter, routes

    monkeypatch.setattr(llm, "_STATE_FILE", tmp_path / "model_state.json")
    monkeypatch.setattr(llm, "_STATE", {"dead": set()})
    monkeypatch.setattr(llm, "_LAST_USED", {"model": llm.MODEL_CHAIN[0]})
    limiter.limiter.reset()
    routes._ANSWERS.clear()
    yield


@pytest.fixture
def lesson_store(monkeypatch):
    """Replaces LessonStore everywhere the team uses it; returns the one shared fake store."""
    from app.agents import team
    store = FakeLessonStore()
    monkeypatch.setattr(team, "LessonStore", lambda repo: store)
    return store

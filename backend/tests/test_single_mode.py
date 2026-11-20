"""MODE=single (the default): Supervisor check -> Researcher -> ONE Explainer call. No Critic, no retry.
Same fake model and fake researcher as test_team.py, so everything runs offline and free."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config
from app.agents import team
from app.api import routes
from eval import harness
from eval.harness import run_one, score_item
from eval.run_eval import parse_mode, results_name, summarize
from tests.conftest import FakeLLM
from tests.test_team import FakeResearchGraph, FakeSearcher, fake_research  # noqa: F401  (autouse fixture)

ITEM = {"id": "q1", "question": "How are retries handled?", "expected_route": "research",
        "expected_files": ["retry.py"], "expected_keywords": ["retry"]}


def writer_llm(answer="Retries happen in `src/retry.py:1-9`."):
    return FakeLLM({"SUPERVISOR": ['{"route": "research", "reason": "about the code"}'],
                    "EXPLAINER": [answer]}, name="writer-model")


def run_single(writer, question="How are retries handled?"):
    critic = FakeLLM({"CRITIC": ['{"score": 5}']}, name="critic-model")
    events = list(team.run_team("demo", question, llm=writer, critic_llm=critic, mode="single"))
    return events, critic


def final_answer(events):
    return next(e["text"] for e in events if e["agent"] == "team" and e["type"] == "answer")


def test_single_is_the_default_mode(lesson_store, monkeypatch):
    assert config.MODE == "single"  # conftest does not set MODE, so this is the real default
    monkeypatch.setattr(team, "MODE", "single")
    events = list(team.run_team("demo", "How are retries handled?", llm=writer_llm(), critic_llm=FakeLLM({})))
    assert events[0] == {"agent": "mode", "type": "info", "text": "single"}


def test_single_mode_never_calls_the_critic_and_answers_once(lesson_store):
    writer = writer_llm()
    events, critic = run_single(writer)
    assert critic.calls == []                                   # no review call at all
    assert "CRITIC" not in writer.roles()
    assert writer.roles().count("EXPLAINER") == 1               # exactly one Explainer call
    assert FakeResearchGraph.runs == 1                          # no retry
    assert not any(e["agent"] == "critic" for e in events)
    answer = final_answer(events)
    # No critic ran, so none of the team's critic warnings may appear.
    assert "Unverified" not in answer and "Confidence: low" not in answer
    assert lesson_store.saved == []


def test_single_mode_skips_the_routing_call_when_the_question_names_repo_code(lesson_store, monkeypatch):
    monkeypatch.setattr("app.tools.search.get_searcher", lambda repo: FakeSearcher(["def", "retry_call"]))
    writer = writer_llm()
    events, _ = run_single(writer, question="How does retry_call work?")
    assert "SUPERVISOR" not in writer.roles()                   # saved one model call
    route = next(e["text"] for e in events if e["type"] == "route")
    assert route.startswith("research") and "retry_call" in route


def test_single_mode_asks_the_model_when_only_plain_words_match(lesson_store, monkeypatch):
    # Plain words like "retries" or "decide" appear in most repos' comments, so they must not skip
    # the routing call: the model decides.
    monkeypatch.setattr("app.tools.search.get_searcher", lambda repo: FakeSearcher(["retries", "handled"]))
    writer = writer_llm()
    run_single(writer)
    assert "SUPERVISOR" in writer.roles()


def test_single_mode_refuses_off_topic_question_that_shares_common_words_with_the_repo(lesson_store, monkeypatch):
    monkeypatch.setattr("app.tools.search.get_searcher",
                        lambda repo: FakeSearcher(["critic", "decide", "whether", "buy", "car"]))
    writer = FakeLLM({"SUPERVISOR": ['{"route": "chat", "reason": "not about code"}']})
    events, _ = run_single(writer, question="should I decide whether to buy a car?")
    assert "SUPERVISOR" in writer.roles()
    assert FakeResearchGraph.runs == 0
    assert final_answer(events) == team.OFF_TOPIC_REPLY        # the event the browser waits for


def test_single_mode_still_overrules_chat_for_repo_words(lesson_store, monkeypatch):
    # The PR #20 bug ("what is CX network?" refused in an NDEx repo) must stay fixed in single mode.
    monkeypatch.setattr("app.tools.search.get_searcher", lambda repo: FakeSearcher(["cx", "network"]))
    writer = writer_llm()
    writer.replies["SUPERVISOR"] = ['{"route": "chat", "reason": "general question"}']
    events, _ = run_single(writer, question="what is CX network?")
    assert FakeResearchGraph.runs == 1
    assert final_answer(events) != team.OFF_TOPIC_REPLY


def test_single_mode_refuses_off_topic_questions_without_research(lesson_store, monkeypatch):
    monkeypatch.setattr("app.tools.search.get_searcher", lambda repo: FakeSearcher(["what", "the", "today"]))
    writer = FakeLLM({"SUPERVISOR": ['{"route": "chat", "reason": "weather"}']})
    events, _ = run_single(writer, question="what's the weather today?")
    assert FakeResearchGraph.runs == 0
    assert final_answer(events) == team.OFF_TOPIC_REPLY        # the event the browser waits for


def test_single_mode_fixes_citation_paths(lesson_store):
    writer = writer_llm("See `retry.py:1-9` and `backend/app/auth/login.py:4`.")
    events, _ = run_single(writer)
    answer = final_answer(events)
    assert "`src/retry.py:1-9`" in answer                       # short path -> the real indexed path
    assert "backend/app/auth/login.py" + team.NOT_FOUND_LABEL in answer  # invented path is labelled


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        list(team.run_team("demo", "How?", llm=writer_llm(), mode="tema"))


def test_a_typo_in_the_MODE_env_var_stops_the_app_at_startup():
    backend = Path(__file__).resolve().parents[1]
    env = {**os.environ, "MODE": "tema"}  # a copy: the real environment of this test run is untouched
    proc = subprocess.run([sys.executable, "-c", "import app.config"], cwd=backend, env=env,
                          capture_output=True, text=True)
    assert proc.returncode != 0 and "MODE must be one of" in proc.stderr


def test_browser_stream_works_end_to_end_in_single_mode(lesson_store, monkeypatch):
    """The real run_team (fake model) through the real /api/ask SSE endpoint."""
    monkeypatch.setattr(routes, "_indexed_repos", lambda: ["demo"])
    monkeypatch.setattr(routes, "run_team", lambda repo, question, request_id=None:
                        team.run_team(repo, question, llm=writer_llm(), mode="single"))
    from app.main import app
    resp = TestClient(app).get("/api/ask", params={"repo": "demo", "question": "How?", "request_id": "r1"},
                               headers={"x-forwarded-for": "203.0.113.9"})
    evs = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    assert evs[0]["agent"] == "mode" and evs[-1] == {"agent": "team", "type": "answer",
                                                    "text": "Retries happen in `src/retry.py:1-9`."}
    assert routes._ANSWERS["r1"]["files"] == ["src/retry.py"]  # thumbs up/down still works


# ---------- the eval can run either mode ----------
def test_eval_records_the_mode_and_does_not_fake_a_review_score(monkeypatch):
    def fake_run(repo, question, llm=None, mode=None):
        assert mode == "single"
        yield {"agent": "mode", "type": "info", "text": "single"}
        yield {"agent": "supervisor", "type": "route", "text": "research (code)"}
        yield {"agent": "team", "type": "answer", "text": "retry is in `retry.py:1-2`"}
    monkeypatch.setattr(harness, "run_team", fake_run)
    row = score_item(ITEM, run_one("demo", ITEM, mode="single"))
    assert row["mode"] == "single" and row["file_hit"] is True
    assert row["passed"] is None and row["critic_score"] is None  # n/a, NOT a fail
    s = summarize([row])
    assert s["pass_rate"] is None and s["file_hit_rate"] == 1


def test_eval_mode_flag_and_file_names():
    assert parse_mode(["codeatlas"]) == "team"                  # old commands measure what they always did
    assert parse_mode(["codeatlas", "--mode", "single"]) == "single"
    with pytest.raises(SystemExit):
        parse_mode(["codeatlas", "--mode", "solo"])
    assert results_name("codeatlas", "team", "20261120-1200") == "codeatlas-20261120-1200.json"
    assert results_name("codeatlas", "single", "20261120-1200") == "codeatlas-mode-single-20261120-1200.json"

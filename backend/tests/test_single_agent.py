"""Team vs one agent: the cost counters and the comparison must be honest too."""
import json

from langchain_core.messages import AIMessage

from app.agents import llm
from eval import compare_setups, single_agent
from eval.harness import RunResult, cost_fields, cost_summary, measured

ITEM = {"id": "q1", "question": "How?", "expected_route": "research",
        "expected_files": ["a.py"], "expected_keywords": ["retry"]}


class UsageClient:
    def __init__(self, model):
        self.model = model

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kw):
        return AIMessage(content="ok", usage_metadata={"input_tokens": 100, "output_tokens": 20,
                                                       "total_tokens": 120})


def test_every_successful_call_is_counted_with_its_tokens_and_model(monkeypatch):
    monkeypatch.setattr(llm, "_make_client", lambda model, temperature: UsageClient(model))
    r = RunResult(id="q", question="?")
    with measured(r):
        llm.get_llm().invoke([])
        llm.get_critic_llm().invoke([])  # the critic's tokens cost quota too, so they count
    assert r.usage["calls"] == 2 and r.usage["tokens"] == 240 and r.usage["input_tokens"] == 200
    assert sum(r.usage["by_model"].values()) == 2
    assert r.seconds >= 0


def test_cost_summary_ignores_old_rows_without_counts():
    rows = [{"tokens": 1000, "llm_calls": 4, "seconds": 10, "llm_seconds": 5},
            {"tokens": 3000, "llm_calls": 8, "seconds": 30, "llm_seconds": 15},
            {"id": "old-file-row"}]
    s = cost_summary(rows)
    assert s["n_measured"] == 2 and s["avg_tokens"] == 2000 and s["avg_llm_calls"] == 6


def test_single_agent_takes_the_final_answer_and_records_cost(monkeypatch):
    def fake_ask(repo, question, llm=None):
        yield "agent", AIMessage(content="", tool_calls=[{"name": "search_code", "args": {"query": "x"}, "id": "1"}])
        yield "tools", AIMessage(content="search results")
        yield "agent", AIMessage(content="Retry lives in `a.py:1-2`.")
    monkeypatch.setattr(single_agent, "ask", fake_ask)
    monkeypatch.setattr(single_agent, "_fix_citation_paths", lambda text, repo: text)
    r = single_agent.run_single("demo", ITEM)
    assert r.final == "Retry lives in `a.py:1-2`." and r.cited_files == ["a.py"] and not r.error
    assert cost_fields(r)["tokens"] == 0  # the fake made no real model calls


def test_single_agent_crash_is_an_error_not_a_lost_question(monkeypatch):
    def broken_ask(repo, question, llm=None):
        raise RuntimeError("boom")
        yield  # pragma: no cover
    monkeypatch.setattr(single_agent, "ask", broken_ask)
    monkeypatch.setattr(single_agent, "_fix_citation_paths", lambda text, repo: text)
    rows = single_agent.run_all("demo", [ITEM])
    assert len(rows) == 1 and rows[0]["file_hit"] is False and "boom" in rows[0]["error"]


def _row(qid, hit, model="m1", error="", tokens=100):
    return {"id": qid, "file_hit": hit, "model": model, "error": error, "answer": "",
            "tokens": tokens, "llm_calls": 1, "seconds": 1, "llm_seconds": 1}


def test_quota_cut_questions_are_dropped_from_every_setup():
    quota = "RuntimeError: All 3 configured Groq model(s) are exhausted for today"
    setups = {"baseline": [_row("a", False), _row("b", False), _row("c", False)],
              "single": [_row("a", True), _row("b", True), _row("c", True)],
              "team": [_row("a", True), _row("b", False, error=quota)]}  # c never ran
    assert compare_setups.common_ids(setups) == ["a"]


def test_paired_counts_and_flags_model_mismatch():
    team = [_row("a", True), _row("b", True, model="m2"), _row("c", False)]
    single = [_row("a", True), _row("b", False), _row("c", True)]
    p = compare_setups.paired(team, single)
    assert p == {"both": 1, "only_team": 1, "only_single": 1, "neither": 0, "different_model": 1}


def test_compare_warns_when_team_used_lessons(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare_setups, "RESULTS_DIR", tmp_path)
    for name, rows, summary in [("demo-baseline-20260101-000000.json", [_row("a", False)], {}),
                                ("demo-single-20260101-000000.json", [_row("a", True, tokens=100)], {}),
                                ("demo-20260101-000000.json", [_row("a", True, tokens=300)], {"lessons_used": True})]:
        (tmp_path / name).write_text(json.dumps({"summary": summary, "rows": rows}))
    compare_setups.main("demo", [])
    out = capsys.readouterr().out
    assert "| Full team | 100% |" in out and "3.0x the tokens" in out and "saved lessons" in out


def test_no_lessons_switch_turns_off_recall(monkeypatch):
    from app.memory import lessons
    from eval.run_eval import disable_lesson_recall
    monkeypatch.setattr(lessons.LessonStore, "recall", lessons.LessonStore.recall)  # restored after the test
    disable_lesson_recall()
    assert lessons.LessonStore.recall(object(), "anything") == []

"""The eval must be honest: a broken review is a FAIL and stays in the averages."""
from eval import harness
from eval.harness import check_citations, run_one, score_item
from eval.run_eval import summarize

ITEM = {"id": "q1", "question": "How?", "expected_route": "research",
        "expected_files": ["a.py"], "expected_keywords": ["retry"]}


def fake_run(events):
    return lambda repo, question, llm=None: iter(events)


def test_broken_review_counts_as_fail(monkeypatch):
    monkeypatch.setattr(harness, "run_team", fake_run([
        {"agent": "supervisor", "type": "route", "text": "research (code)"},
        {"agent": "critic", "type": "model", "text": "critic-model"},
        {"agent": "critic", "type": "review_failed", "text": "no valid score"},
        {"agent": "team", "type": "answer", "text": "retry is in `a.py:1-2`"},
    ]))
    row = score_item(ITEM, run_one("demo", ITEM))
    assert row["critic_failed"] is True and row["passed"] is False and row["critic_score"] is None
    s = summarize([row])
    assert s["n_research"] == 1 and s["pass_rate"] == 0 and s["critic_failures"] == 1


def test_unreadable_review_text_is_not_silently_skipped(monkeypatch):
    monkeypatch.setattr(harness, "run_team", fake_run([
        {"agent": "supervisor", "type": "route", "text": "research"},
        {"agent": "critic", "type": "review", "text": "no valid review returned; accepting the answer"},
        {"agent": "team", "type": "answer", "text": "x"},
    ]))
    row = score_item(ITEM, run_one("demo", ITEM))
    assert row["passed"] is False and row["critic_failed"] is True


def test_self_grading_is_flagged(monkeypatch):
    monkeypatch.setattr(harness, "run_team", fake_run([
        {"agent": "supervisor", "type": "route", "text": "research"},
        {"agent": "model", "type": "info", "text": "same-model"},
        {"agent": "critic", "type": "model", "text": "same-model"},
        {"agent": "critic", "type": "review", "text": "score 5/5"},
        {"agent": "team", "type": "answer", "text": "x"},
    ]))
    row = score_item(ITEM, run_one("demo", ITEM))
    assert row["passed"] is True and row["critic_same_model"] is True
    assert summarize([row])["critic_same_model"] == 1


def test_misrouted_research_question_is_a_fail_not_dropped():
    row = {"id": "q", "route_ok": False, "route": "chat", "critic_score": None, "passed": False,
           "file_hit": False, "keyword_hit": False}
    ok = {"id": "q2", "route_ok": True, "route": "research", "critic_score": 5, "passed": True,
          "file_hit": True, "keyword_hit": True}
    s = summarize([row, ok])
    assert s["pass_rate"] == 0.5 and s["avg_critic_score"] == 5


def test_invented_paths_caught_by_the_filter_are_counted(monkeypatch):
    monkeypatch.setattr(harness, "run_team", fake_run([
        {"agent": "supervisor", "type": "route", "text": "research"},
        {"agent": "critic", "type": "review", "text": "score 5/5"},
        {"agent": "team", "type": "answer",
         "text": "see a.py:1 and x/login.py (not found in this repo) and y.py (not found in this repo)"},
    ]))
    row = score_item(ITEM, run_one("demo", ITEM))
    assert row["invented_paths"] == 2
    assert summarize([row])["invented_paths"] == 2


def test_citation_check_catches_invented_files_and_lines(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("1\n2\n3\n")
    c = check_citations("ok `src/a.py:1-3`, short `a.py:2`, too far `src/a.py:2-99`, fake `b.py:1`",
                        "demo", repo_dir=tmp_path)
    assert c["citations"] == 4 and c["valid_citations"] == 2
    assert c["invalid_citations"] == ["src/a.py:2-99", "b.py:1"]

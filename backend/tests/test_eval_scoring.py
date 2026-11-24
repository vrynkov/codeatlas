"""The eval must be honest: a broken review is a FAIL and stays in the averages."""
from eval import harness
from eval.harness import check_citations, run_one, score_item
from eval.run_eval import summarize

ITEM = {"id": "q1", "question": "How?", "expected_route": "research",
        "expected_files": ["a.py"], "expected_keywords": ["retry"]}


def fake_run(events):
    return lambda repo, question, llm=None, **kw: iter(events)


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


def test_symbol_check_catches_right_file_but_wrong_lines(tmp_path):
    """`a.py:1-2` exists, so the old check passes it. But if the answer says `retry` lives there and
    retry is really on line 9, the stricter check must flag it."""
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n\n\n\n\n\n\ndef retry():\n    pass\n")
    answer = ("`retry` is defined at `a.py:9-10`.\n"
              "Also `retry` is at `a.py:1-2`.\n"                       # wrong lines, real file
              "`y` (`a.py:2`) and `retry` (`a.py:9`) are separate.\n"  # each name checked against its own citation
              "See `a.py:1` for setup.")                              # no symbol: not checkable
    c = check_citations(answer, "demo", repo_dir=tmp_path)
    assert c["valid_citations"] == 5                                   # every one EXISTS...
    assert c["symbol_checked"] == 4 and c["symbol_matched"] == 3       # ...but one points at the wrong lines
    assert c["symbol_mismatches"] == ["a.py:1-2 (named retry)"]


def test_resume_keeps_clean_rows_and_reruns_crashed_or_missing_ones(tmp_path):
    import json as _json

    import pytest as _pytest

    from eval.run_eval import resume_from
    items = [{"id": i} for i in ("a", "b", "c", "d")]
    old = tmp_path / "old.json"
    old.write_text(_json.dumps({"summary": {"mode": "single", "lessons_used": False},
                                "rows": [{"id": "a", "error": ""},
                                         {"id": "b", "error": "Researcher error: RateLimitError ..."},
                                         {"id": "c", "error": "RuntimeError: ... exhausted for today"}]}))
    done, todo = resume_from(str(old), items, "single", no_lessons=True)
    assert list(done) == ["a"] and [q["id"] for q in todo] == ["b", "c", "d"]  # d never ran
    with _pytest.raises(SystemExit):
        resume_from(str(old), items, "team", no_lessons=True)    # different mode: not comparable
    with _pytest.raises(SystemExit):
        resume_from(str(old), items, "single", no_lessons=False)  # different lessons setting


NOT_IN_REPO = {"id": "nir", "question": "Where is login?", "expected_route": "research",
               "expected_files": [], "expected_keywords": [], "expected_not_found": True}


def test_not_in_repo_question_passes_only_without_a_made_up_file(monkeypatch):
    def answer(text):
        monkeypatch.setattr(harness, "run_team", fake_run([
            {"agent": "supervisor", "type": "route", "text": "research"},
            {"agent": "critic", "type": "review", "text": "score 5/5"},
            {"agent": "team", "type": "answer", "text": text}]))
        return score_item(NOT_IN_REPO, run_one("demo", NOT_IN_REPO))

    honest = answer("This repo has no login code.")
    made_up = answer("Login lives in src/auth.py (not found in this repo).")
    assert honest["honest_not_found"] is True and made_up["honest_not_found"] is False
    s = summarize([honest, made_up])
    assert s["not_found_honest_rate"] == 0.5 and s["n_not_found"] == 2
    assert score_item(ITEM, run_one("demo", ITEM))["honest_not_found"] is None  # normal questions: n/a


def test_holdout_questions_never_overlap_the_main_set():
    from eval.run_eval import questions_for
    main, holdout = questions_for("codeatlas"), questions_for("codeatlas", holdout=True)
    assert holdout and not {q["id"] for q in main} & {q["id"] for q in holdout}
    assert not {q["question"].lower() for q in main} & {q["question"].lower() for q in holdout}


def test_holdout_results_are_not_picked_up_by_compare_setups(tmp_path, monkeypatch):
    from eval import compare_setups
    from eval.run_eval import results_name
    monkeypatch.setattr(compare_setups, "RESULTS_DIR", tmp_path)
    for name in (results_name("codeatlas-holdout", "team", "20261125-100000"),
                 results_name("codeatlas-holdout", "single", "20261125-100000")):
        (tmp_path / name).write_text("{}")
    files = compare_setups.find_files("codeatlas", [])
    assert files["team"] is None and files.get("mode_single") is None

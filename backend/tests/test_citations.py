"""Citation repair: every citation in a final answer must be a REAL file or plain text - never a 404 link."""
import pytest

from app.agents import team

REAL = {"backend/app/tools/search.py", "backend/app/agents/team.py", "README.md"}


@pytest.fixture(autouse=True)
def real_files(monkeypatch):
    monkeypatch.setattr(team, "real_files", lambda repo: REAL)


def fix(text):
    return team._fix_citation_paths(text, "demo")


def test_real_citation_is_kept():
    assert fix("See `backend/app/tools/search.py:10-20`.") == "See `backend/app/tools/search.py:10-20`."


def test_shortened_path_is_repaired():
    assert fix("See `app/tools/search.py:10-20`.") == "See `backend/app/tools/search.py:10-20`."


def test_invented_file_becomes_plain_text_without_a_link():
    out = fix("Like `backend/app/auth/login.py:12-34` does.")
    assert out == "Like backend/app/auth/login.py does."


def test_bare_backticked_invented_path_is_unwrapped():
    assert fix("Open `src/nope.py` first.") == "Open src/nope.py first."


def test_unicode_dash_ranges_are_understood():
    assert fix("`app/agents/team.py:1–5`") == "`backend/app/agents/team.py:1–5`"


def test_no_index_means_text_is_left_alone(monkeypatch):
    monkeypatch.setattr(team, "real_files", lambda repo: set())
    assert fix("`whatever.py:1`") == "`whatever.py:1`"


def test_cited_files_lists_each_path_once():
    assert team.cited_files("a `x/a.py:1-2` and x/a.py:5 and `b.md:3`") == ["x/a.py", "b.md"]

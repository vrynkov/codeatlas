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


def test_invented_file_becomes_labelled_plain_text_without_a_link():
    out = fix("Like `backend/app/auth/login.py:12-34` does.")
    assert out == "Like backend/app/auth/login.py (not found in this repo) does."


def test_bare_backticked_invented_path_is_unwrapped_and_labelled():
    assert fix("Open `src/nope.py` first.") == "Open src/nope.py (not found in this repo) first."


def test_ambiguous_real_path_is_unlinked_but_not_called_invented(monkeypatch):
    monkeypatch.setattr(team, "real_files", lambda repo: {"a/utils.py", "b/utils.py"})
    assert fix("See `utils.py:3`.") == "See utils.py."


def test_unicode_dash_ranges_are_understood():
    assert fix("`app/agents/team.py:1–5`") == "`backend/app/agents/team.py:1–5`"


def test_no_index_means_text_is_left_alone(monkeypatch):
    monkeypatch.setattr(team, "real_files", lambda repo: set())
    assert fix("`whatever.py:1`") == "`whatever.py:1`"


def test_cited_files_lists_each_path_once():
    assert team.cited_files("a `x/a.py:1-2` and x/a.py:5 and `b.md:3`") == ["x/a.py", "b.md"]


@pytest.mark.parametrize("raw, expected", [
    # gpt-oss's own citation style (seen in a real answer) becomes the normal `path:lines` form
    ("printed after that JSON header【README.md:1-20】.", "printed after that JSON header `README.md:1-20`."),
    ("See (【search.py†L10-L20】)", "See (`backend/app/tools/search.py:10-20`)"),
    ("【backend/app/agents/team.py】 holds it", "`backend/app/agents/team.py` holds it"),
])
def test_bracket_citations_are_converted_and_checked(raw, expected):
    assert fix(raw) == expected


def test_invented_file_in_brackets_is_still_caught():
    out = fix("It is here【backend/app/auth/login.py:4】.")
    assert "【" not in out and "backend/app/auth/login.py" + team.NOT_FOUND_LABEL in out


# Real strings from the 2026-11-24 eval answers: code names in backticks, not files. Before the
# file-extension check, every one of these was labelled "(not found in this repo)".
@pytest.mark.parametrize("code", ["request.client.host", "sys.path", "LessonStore.add", "self.docs",
                                  "github.com", "redis.set", "mcp.run", "cost_summary.avg"])
def test_code_names_are_not_treated_as_invented_files(code):
    text = f"It reads `{code}` first."
    assert fix(text) == text


def test_code_name_with_a_line_number_is_left_alone():
    assert fix("See `self.docs:3`.") == "See `self.docs:3`."


def test_invented_file_is_still_caught_after_the_extension_check():
    assert fix("Login is in `src/auth.py`.") == "Login is in src/auth.py (not found in this repo)."
    assert fix("Saved in `model_fallback_state.json`.") == \
        "Saved in model_fallback_state.json (not found in this repo)."


def test_empty_file_in_the_clone_counts_as_real(tmp_path, monkeypatch):
    # An empty __init__.py has no chunks, so the index (real_files) never lists it.
    (tmp_path / "demo" / "backend" / "app").mkdir(parents=True)
    (tmp_path / "demo" / "backend" / "app" / "__init__.py").write_text("")
    monkeypatch.setattr(team, "REPOS_DIR", tmp_path)
    assert fix("See `backend/app/__init__.py`.") == "See `backend/app/__init__.py`."


def test_on_disk_check_cannot_escape_the_clone(tmp_path, monkeypatch):
    (tmp_path / "demo").mkdir()
    (tmp_path / "secret.py").write_text("x = 1")
    monkeypatch.setattr(team, "REPOS_DIR", tmp_path)
    assert team._on_disk("demo", "../secret.py") is False

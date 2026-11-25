"""Tool results say where an imported name is DEFINED, and flag test files (made-up example paths).
Both come from the 2026-11-24 eval: "where are the limits configured?" was answered with the file
that USES MAX_REPO_MB instead of config.py, and fake paths from our own tests were presented as
the repo's login code."""
from app.tools import repo_tools
from app.tools.code_hints import TEST_FILE_NOTE, import_hints, is_test_file


def _repo(tmp_path):
    """A tiny repo laid out like CodeAtlas's backend/."""
    app = tmp_path / "demo" / "backend" / "app"
    (app / "indexing").mkdir(parents=True)
    (app / "__init__.py").write_text("")
    (app / "indexing" / "__init__.py").write_text("")
    (app / "config.py").write_text("MAX_REPO_MB = 50\nREPOS_DIR = 'data/repos'\n")
    (app / "indexing" / "indexer.py").write_text(
        "import re\n"
        "from ..config import MAX_REPO_MB, REPOS_DIR\n"
        "from . import helpers\n"
        "\n"
        "def too_big(mb):\n"
        "    return mb > MAX_REPO_MB\n")
    (app / "indexing" / "helpers.py").write_text("def clean(x):\n    return x\n")
    (tmp_path / "demo" / "backend" / "tests").mkdir()
    (tmp_path / "demo" / "backend" / "tests" / "test_auth.py").write_text(
        'def test_label():\n    assert fix("`backend/app/auth/login.py:1`")\n')
    return tmp_path / "demo"


def test_relative_import_points_at_the_defining_file(tmp_path):
    root = _repo(tmp_path)
    hints = import_hints(root, "backend/app/indexing/indexer.py", "return mb > MAX_REPO_MB")
    assert hints == "Imported here, defined in another file: MAX_REPO_MB -> backend/app/config.py"


def test_only_names_the_agent_can_see_are_listed(tmp_path):
    root = _repo(tmp_path)
    assert "REPOS_DIR" not in import_hints(root, "backend/app/indexing/indexer.py", "return mb > MAX_REPO_MB")


def test_from_dot_import_module_points_at_the_module(tmp_path):
    root = _repo(tmp_path)
    hints = import_hints(root, "backend/app/indexing/indexer.py", "helpers.clean(x)")
    assert hints == "Imported here, defined in another file: helpers -> backend/app/indexing/helpers.py"


def test_absolute_import_is_resolved(tmp_path):
    root = _repo(tmp_path)
    (root / "backend" / "eval").mkdir()
    (root / "backend" / "eval" / "run.py").write_text("from app.config import MAX_REPO_MB\nprint(MAX_REPO_MB)\n")
    hints = import_hints(root, "backend/eval/run.py", "print(MAX_REPO_MB)")
    assert hints.endswith("MAX_REPO_MB -> backend/app/config.py")


def test_library_imports_non_python_and_broken_files_give_no_hint(tmp_path):
    root = _repo(tmp_path)
    (root / "backend" / "app" / "lib.py").write_text("from os import path\nprint(path)\n")
    (root / "backend" / "app" / "broken.py").write_text("from ..config import (\n")
    (root / "README.md").write_text("MAX_REPO_MB")
    assert import_hints(root, "backend/app/lib.py", "print(path)") == ""
    assert import_hints(root, "backend/app/broken.py", "x") == ""
    assert import_hints(root, "README.md", "MAX_REPO_MB") == ""


def test_import_outside_the_repo_is_ignored(tmp_path):
    root = _repo(tmp_path)
    (tmp_path / "secret.py").write_text("KEY = 1\n")
    (root / "top.py").write_text("from ..secret import KEY\nprint(KEY)\n")
    assert import_hints(root, "top.py", "print(KEY)") == ""


def test_test_files_are_recognised():
    assert is_test_file("backend/tests/test_citations.py")
    assert is_test_file("test_x.py") and is_test_file("pkg/x_test.py")
    assert is_test_file("frontend/src/App.test.jsx") and is_test_file("src/__tests__/a.js")
    assert not is_test_file("backend/app/tools/search.py")
    assert not is_test_file("backend/app/latest.py")


def test_get_file_adds_the_hint_and_the_test_note(tmp_path, monkeypatch):
    _repo(tmp_path)
    monkeypatch.setattr(repo_tools, "REPOS_DIR", tmp_path)
    search_code, get_file = repo_tools.build_tools("demo")
    out = get_file.invoke({"path": "backend/app/indexing/indexer.py", "start_line": 5, "end_line": 6})
    assert out.endswith("MAX_REPO_MB -> backend/app/config.py")
    out = get_file.invoke({"path": "backend/tests/test_auth.py"})
    assert TEST_FILE_NOTE in out


def test_search_code_flags_test_files(tmp_path, monkeypatch):
    _repo(tmp_path)
    monkeypatch.setattr(repo_tools, "REPOS_DIR", tmp_path)

    class FakeSearcher:
        def search(self, query, n=5, path_prefix=""):
            return [{"meta": {"file": "backend/tests/test_auth.py", "start_line": 1, "end_line": 2,
                              "kind": "function", "name": "test_label"},
                     "text": 'def test_label():\n    assert fix("`backend/app/auth/login.py:1`")'},
                    {"meta": {"file": "backend/app/indexing/indexer.py", "start_line": 5, "end_line": 6,
                              "kind": "function", "name": "too_big"},
                     "text": "def too_big(mb):\n    return mb > MAX_REPO_MB"}]

    monkeypatch.setattr(repo_tools, "get_searcher", lambda repo: FakeSearcher())
    search_code, _ = repo_tools.build_tools("demo")
    out = search_code.invoke({"query": "login"})
    first, second = out.split("\n\n")
    assert TEST_FILE_NOTE in first.splitlines()[0]
    assert TEST_FILE_NOTE not in second
    assert second.endswith("MAX_REPO_MB -> backend/app/config.py")

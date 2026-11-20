"""Exact code names in a search query (get_file, HTTPAdapter) must find the code that DEFINES them,
even when prose that shares the split-up words ("get", "file") ranks higher by meaning and BM25.
Runs offline: a RepoSearcher is built by hand, with no Chroma and no embedding model."""
from rank_bm25 import BM25Okapi

from app.tools.search import RepoSearcher, _tokenize, code_names

CHUNKS = [
    ("readme", "README.md", "", "To get a file, open the file view. Every file has a file page. Get the file."),
    ("guide", "docs/guide.md", "", "You can get file contents and file lists. Files are files."),
    ("tool", "backend/app/tools/repo_tools.py", "build_tools.get_file",
     "def get_file(path, start_line=1, end_line=80):\n    return numbered lines"),
    ("caller", "backend/app/agents/react_agent.py", "run", "tools = [search_code, get_file]"),
]


class FakeCollection:
    """Meaning search that (wrongly) loves the prose chunks, like the real bug report."""
    def query(self, query_texts, n_results):
        return {"ids": [["readme", "guide", "caller", "tool"][:n_results]]}


def searcher():
    s = RepoSearcher.__new__(RepoSearcher)  # skip __init__: it would open Chroma
    s.col = FakeCollection()
    s.ids = [c[0] for c in CHUNKS]
    s.metas = [{"file": c[1], "name": c[2], "start_line": 1, "end_line": 5, "kind": "function"} for c in CHUNKS]
    s.docs = [f"# File: {c[1]}\n{c[3]}" for c in CHUNKS]
    s.bm25 = BM25Okapi([_tokenize(d) for d in s.docs])
    return s


def test_exact_function_name_puts_its_definition_first():
    files = [r["meta"]["file"] for r in searcher().search("what does get_file return?")]
    assert files[0] == "backend/app/tools/repo_tools.py"
    assert "backend/app/agents/react_agent.py" in files  # a chunk that USES it is found too


def test_plain_words_keep_the_normal_ranking():
    files = [r["meta"]["file"] for r in searcher().search("how do I get a file")]
    assert files[0] == "README.md"  # no code name in the query: nothing is pinned


def test_code_names_only_picks_code_shaped_tokens():
    assert code_names("what does get_file return?") == ["get_file"]
    assert code_names("HTTPAdapter in search.py?") == ["HTTPAdapter", "search.py"]
    assert code_names("how are retries handled") == []

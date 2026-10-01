"""The tools the agent can use. Each tool is a plain Python function with a clear description."""
from pathlib import Path

from langchain_core.tools import tool

from ..config import MAX_TOOL_OUTPUT_CHARS, REPOS_DIR
from .search import RepoNotIndexedError, get_searcher


def build_tools(repo_name: str):
    repo_root = (REPOS_DIR / repo_name).resolve()

    @tool
    def search_code(query: str, path: str = "") -> str:
        """Search the repository for code or docs related to the query.
        Works best with short queries: a concept ('retry logic') or an exact name ('HTTPAdapter').
        Optional `path` limits the search to a folder or file, e.g. 'src/requests/'.
        Returns file paths, line numbers and a short preview of each match."""
        try:
            results = get_searcher(repo_name).search(query, n=5, path_prefix=path)
        except RepoNotIndexedError as e:
            # Returned as normal tool output (not raised) so the agent can read it and tell
            # the person plainly, instead of an uncaught crash or a misleading "no evidence".
            return str(e)
        out = []
        for r in results:
            m = r["meta"]
            preview = "\n".join(r["text"].splitlines()[1:12])  # skip the '# File:' header line
            out.append(f"--- {m['file']} (lines {m['start_line']}-{m['end_line']}, {m['kind']})\n{preview}")
        return ("\n\n".join(out) or "No results.")[:MAX_TOOL_OUTPUT_CHARS]

    @tool
    def get_file(path: str, start_line: int = 1, end_line: int = 80,
                 line_start: int | None = None, line_end: int | None = None) -> str:
        """Read part of a file from the repository, with line numbers.
        Use start_line and end_line (line_start / line_end are accepted as aliases).
        Max 120 lines per call; the result tells you how to continue if the file is longer."""
        if line_start is not None:  # models often guess these names, so we accept them
            start_line = line_start
        if line_end is not None:
            end_line = line_end
        target = (repo_root / path).resolve()
        if repo_root not in target.parents or not target.is_file():  # blocks '../' tricks
            return f"File not found: {path}"
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line)
        if start > len(lines):
            return f"{path} has only {len(lines)} lines."
        end = min(end_line, start + 119, len(lines))
        out, used = [], 0
        for i in range(start, end + 1):  # add whole lines until we hit the size limit
            line = f"{i}: {lines[i - 1]}"
            if used + len(line) + 1 > MAX_TOOL_OUTPUT_CHARS:
                break
            out.append(line)
            used += len(line) + 1
        end = start + len(out) - 1  # the header now tells the truth about what was returned
        header = f"{path} (lines {start}-{end} of {len(lines)})"
        if end < len(lines):
            header += f" - more lines follow; continue with start_line={end + 1}"
        return header + "\n" + "\n".join(out)

    return [search_code, get_file]

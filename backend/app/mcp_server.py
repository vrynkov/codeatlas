"""CodeAtlas as an MCP server, so other AI apps (Claude Desktop, IDEs, your own agents) can use it as a tool.

MCP (Model Context Protocol) is a standard "plug" between AI apps and tools. Any MCP client can:
  1. ask the server "what tools do you have?"   -> list_tools
  2. call a tool with arguments                 -> call_tool

Tools exposed here:
  list_repos     which repositories are indexed
  search_code    find code/docs by meaning + keywords
  get_file       read lines of a file (with line numbers)
  ask_codeatlas  run the FULL agent team (researcher -> explainer -> critic) and get a verified answer
  rate_answer    tell CodeAtlas if an answer helped, so it can learn (self-improvement)

Run (from backend/):
    python run_mcp.py           # stdio: for Claude Desktop, MCP Inspector, and the demo client
    python run_mcp.py --http    # HTTP on http://127.0.0.1:8000/mcp

IMPORTANT: with stdio, stdout is the wire. Never print() to stdout in here; use stderr.
"""
import re
import sys

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .agents.team import run_team, submit_feedback
from .config import REPOS_DIR
from .tools.repo_tools import build_tools
from .tools.search import get_searcher

mcp = MCPServer(
    name="CodeAtlas",
    instructions=(
        "CodeAtlas answers questions about indexed GitHub repositories. "
        "Call list_repos first. For a quick lookup use search_code / get_file. "
        "For 'how does X work?' questions use ask_codeatlas, which runs a team of agents and "
        "checks its own answer against the code. Afterwards, call rate_answer to help it learn."
    ),
)


def _indexed_repos() -> list[str]:
    if not REPOS_DIR.exists():
        return []
    return sorted(p.name for p in REPOS_DIR.iterdir() if p.is_dir())


def _check(repo: str) -> str:
    """Only allow repos we actually indexed (this also blocks '../' tricks in the repo name)."""
    repos = _indexed_repos()
    if repo not in repos:
        raise ToolError(f"Unknown repo '{repo}'. Indexed repos: {repos or 'none yet'}.")  # ToolError = message shown to the client
    return repo


@mcp.tool()
def list_repos() -> list[str]:
    """List the repositories that CodeAtlas has indexed and can answer questions about."""
    return _indexed_repos()


@mcp.tool()
def search_code(repo: str, query: str, path: str = "") -> str:
    """Search an indexed repository by meaning AND keywords (hybrid search).
    Use a short concept ('retry logic') or an exact name ('HTTPAdapter').
    `path` optionally limits the search to a folder, e.g. 'src/requests/'.
    Returns file paths, line numbers and a preview of each match."""
    tools = {t.name: t for t in build_tools(_check(repo))}
    return tools["search_code"].invoke({"query": query, "path": path})


@mcp.tool()
def get_file(repo: str, path: str, start_line: int = 1, end_line: int = 80) -> str:
    """Read part of a file from an indexed repository, with line numbers (max ~120 lines per call).
    The result says how to continue if the file is longer."""
    tools = {t.name: t for t in build_tools(_check(repo))}
    return tools["get_file"].invoke({"path": path, "start_line": start_line, "end_line": end_line})


@mcp.tool()
async def ask_codeatlas(repo: str, question: str, ctx: Context) -> str:
    """Ask a question about a repository. A team of agents researches the code, writes an explanation
    with file:line citations, and a critic checks it against the code. Takes ~30-90 seconds."""
    _check(repo)
    events = run_team(repo, question)  # a normal (blocking) generator, so we step it in a worker thread
    done, answer, step = object(), "", 0
    while True:
        ev = await anyio.to_thread.run_sync(next, events, done)
        if ev is done:
            break
        if ev["type"] == "answer":
            answer = ev["text"]
        else:  # live progress for the client (like the trace panel in the web UI)
            step += 1
            await ctx.report_progress(step, message=f"[{ev['agent']}] {ev['type']}: {ev['text'][:200]}")
    files = list(dict.fromkeys(re.findall(r"([\w./-]+\.\w{1,4}):\d+", answer)))
    footer = ""
    if files:
        footer = ("\n\n---\nFiles cited: " + ", ".join(files) +
                  "\nIf this helped (or not), call rate_answer with the same repo, question and these files.")
    return answer + footer


@mcp.tool()
def rate_answer(repo: str, question: str, helpful: bool, files: list[str] | None = None, note: str = "") -> str:
    """Tell CodeAtlas whether an answer was helpful. Good answers become 'playbook' lessons and bad ones
    become warnings; both are used as hints on similar questions later (self-improvement)."""
    _check(repo)
    return submit_feedback(repo, question, files or [], helpful, note)


def _warm_up() -> None:
    """Load the embedding model and search index once at startup, so the first tool call isn't slow
    (MCP clients give up on calls that take too long)."""
    for repo in _indexed_repos():
        try:
            get_searcher(repo)
        except Exception as e:  # noqa: BLE001  the server should still start
            print(f"[codeatlas] warm-up skipped for {repo}: {type(e).__name__}: {e}", file=sys.stderr)


def main() -> None:
    _warm_up()
    if "--http" in sys.argv:
        mcp.run("streamable-http", host="127.0.0.1", port=8000)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()

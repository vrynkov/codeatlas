"""A tiny MCP *client*: it starts the CodeAtlas server and uses its tools, like Claude Desktop would.

Run from backend/ (the venv must be active):
    python scripts/mcp_client_demo.py requests "How are retries handled?"
"""
import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

SERVER = Path(__file__).resolve().parents[1] / "run_mcp.py"


def text_of(result) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


async def on_progress(progress, total, message) -> None:  # the server's live progress messages
    print(f"   ... {message}")


async def main(repo: str, question: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with Client(params, read_timeout_seconds=300) as client:
        tools = await client.list_tools()
        print("Tools offered by the server:", [t.name for t in tools.tools])

        print("\n> list_repos")
        print(text_of(await client.call_tool("list_repos", {})))

        print("\n> search_code")
        found = await client.call_tool("search_code", {"repo": repo, "query": "max_retries"})
        print(text_of(found)[:600], "...")

        print("\n> ask_codeatlas (full agent team, takes a minute)")
        answer = await client.call_tool("ask_codeatlas", {"repo": repo, "question": question},
                                        progress_callback=on_progress)
        print("\n" + text_of(answer))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    asyncio.run(main(sys.argv[1], sys.argv[2]))

"""Launcher for the CodeAtlas MCP server. It works no matter which folder starts it
(Claude Desktop, MCP Inspector, ...), because it adds its own folder to Python's search path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()

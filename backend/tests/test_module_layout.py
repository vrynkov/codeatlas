"""Regression test: in a module that can be run as a script (`python -m ...`), the
`if __name__ == "__main__":` block must come LAST. Code below it isn't defined yet when the
script part runs - that once broke `python -m app.indexing.indexer index ...` with a NameError
while every import-based test still passed."""
import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
MODULES = sorted(p for d in ("app", "eval", "scripts") for p in (BACKEND / d).rglob("*.py"))


def _is_main_guard(node) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(BACKEND)))
def test_main_block_is_the_last_statement(path):
    body = ast.parse(path.read_text(encoding="utf-8")).body
    guards = [n for n in body if _is_main_guard(n)]
    if guards:
        assert body[-1] is guards[-1], f"code after `if __name__ == '__main__':` in {path.name}"

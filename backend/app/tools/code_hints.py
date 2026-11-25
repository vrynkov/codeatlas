"""Small facts added to tool results, so the agent doesn't have to guess them. Plain code, no model.

1. Where an imported name is DEFINED. In indexer.py the agent sees `MAX_REPO_MB` used and can
   wrongly answer "the limit is set in indexer.py". The line `from ..config import MAX_REPO_MB`
   says it is defined in config.py, so the tool result says that too:
       Imported here, defined in another file: MAX_REPO_MB -> backend/app/config.py
2. Test files hold made-up examples. Our own tests contain fake paths such as
   `backend/app/auth/login.py` on purpose (to test the citation checker), and an answer once
   presented that as the repo's login code. Tool results from a test file now say so.
"""
import ast
import re
from pathlib import Path

TEST_FILE_NOTE = ("(test file: paths and names inside its strings are often made-up examples, "
                  "not real files of this repo)")
_MAX_IMPORT_HINTS = 8


def is_test_file(path: str) -> bool:
    """tests/test_x.py, test_x.py, x_test.py, x.test.js, x.spec.ts, __tests__/x.js ..."""
    p = Path(path)
    name = p.name.lower()
    return (any(part.lower() in ("tests", "test", "__tests__") for part in p.parts[:-1])
            or name.startswith("test_") or name.endswith("_test.py")
            or ".test." in name or ".spec." in name)


def _module_file(base: Path, module: str, root: Path) -> Path | None:
    """base/a/b.py or base/a/b/__init__.py for module "a.b", if it exists inside root."""
    target = base.joinpath(*module.split(".")) if module else base
    for cand in (target.with_suffix(".py"), target / "__init__.py"):
        try:
            cand = cand.resolve()
        except OSError:
            continue
        if cand.is_file() and (cand == root or root in cand.parents):
            return cand
    return None


def _resolve(node: ast.ImportFrom, file: Path, root: Path) -> Path | None:
    """The repo file a `from X import ...` line points at, or None (e.g. a library like `re`)."""
    module = node.module or ""
    if node.level:  # relative: `from ..config import X` -> go up (level - 1) folders from the file's folder
        base = file.parent
        for _ in range(node.level - 1):
            base = base.parent
        return _module_file(base, module, root)
    for base in [file.parent, *file.parent.parents]:  # absolute: `from app.config import X`
        if base != root and root not in base.parents:
            break
        found = _module_file(base, module, root)
        if found:
            return found
    return None


def import_hints(root: Path, rel_path: str, shown: str) -> str:
    """'Imported here, defined in another file: A, B -> x/config.py' for every repo-internal name
    imported by the Python file at rel_path that appears in `shown` (the lines the agent sees).
    '' if there are none, the file isn't Python, or it doesn't parse."""
    if not rel_path.endswith(".py"):
        return ""
    root = root.resolve()
    file = (root / rel_path).resolve()
    try:
        tree = ast.parse(file.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return ""
    by_file: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        src = _resolve(node, file, root)
        if src is None:
            continue
        for alias in node.names:
            local = alias.asname or alias.name
            if alias.name == "*" or not re.search(rf"\b{re.escape(local)}\b", shown):
                continue
            # `from . import config` imports a module: point at config.py itself
            sub = _module_file(src.parent, alias.name, root) if src.name == "__init__.py" else None
            where = str((sub or src).relative_to(root)).replace("\\", "/")
            if where != rel_path:
                by_file.setdefault(where, []).append(local)
    lines = [f"Imported here, defined in another file: {', '.join(dict.fromkeys(names))} -> {where}"
             for where, names in by_file.items()]
    return "\n".join(lines[:_MAX_IMPORT_HINTS])

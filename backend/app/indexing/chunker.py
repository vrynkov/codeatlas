"""Cut source files into small pieces ("chunks") so each piece can be searched by meaning."""
import ast
from dataclasses import dataclass
from pathlib import Path

CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".cs", ".rb", ".php"}
DOC_EXTS = {".md", ".rst", ".txt"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".tox"}
MAX_FILE_BYTES = 200_000   # skip huge files
MAX_CHUNK_LINES = 80       # split anything longer than this
WINDOW, OVERLAP = 60, 10   # for files we can't parse


@dataclass
class Chunk:
    text: str
    file: str
    start_line: int
    end_line: int
    kind: str  # function | class | window | doc


def _by_lines(lines: list[str], rel: str, kind: str, offset: int = 0) -> list[Chunk]:
    """Simple fallback: cut into overlapping windows of lines."""
    chunks, i = [], 0
    while i < len(lines):
        part = lines[i : i + WINDOW]
        chunks.append(Chunk("\n".join(part), rel, offset + i + 1, offset + i + len(part), kind))
        i += WINDOW - OVERLAP
    return chunks


def _python(source: str, rel: str) -> list[Chunk]:
    """Python files: one chunk per function/class (uses Python's built-in `ast`)."""
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _by_lines(lines, rel, "window")

    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            end = node.end_lineno
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            seg = lines[start - 1 : end]
            if len(seg) > MAX_CHUNK_LINES:
                chunks += _by_lines(seg, rel, kind, offset=start - 1)
            else:
                chunks.append(Chunk("\n".join(seg), rel, start, end, kind))
    return chunks or _by_lines(lines, rel, "window")


def chunk_file(path: Path, root: Path) -> list[Chunk]:
    rel = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    if not source.strip():
        return []
    if path.suffix == ".py":
        return _python(source, rel)
    if path.suffix in DOC_EXTS:
        return _by_lines(source.splitlines(), rel, "doc")
    return _by_lines(source.splitlines(), rel, "window")


def chunk_repo(root: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(p in SKIP_DIRS for p in path.parts):
            continue
        if path.suffix not in CODE_EXTS | DOC_EXTS:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        chunks += chunk_file(path, root)
    return chunks

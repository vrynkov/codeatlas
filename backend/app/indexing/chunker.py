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


def _def_chunk(lines: list[str], rel: str, node, kind: str) -> list[Chunk]:
    """One function/class definition -> one clean chunk (or windowed, only if it's still too
    long on its own)."""
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    end = node.end_lineno
    seg = lines[start - 1 : end]
    if len(seg) > MAX_CHUNK_LINES:
        return _by_lines(seg, rel, kind, offset=start - 1)
    return [Chunk("\n".join(seg), rel, start, end, kind)]


def _python(source: str, rel: str) -> list[Chunk]:
    """Python files: one chunk per function/class (uses Python's built-in `ast`) - AND a
    separate chunk for each definition NESTED inside one, at any depth: a node function
    defined inside a builder function (a common pattern for LangGraph agent code, including
    this project's own team.py), or a method inside a class. Without this, a nested definition
    never gets a clean chunk of its own - if its outer container is too long (a common case:
    a builder function with several node functions easily exceeds MAX_CHUNK_LINES), it just
    gets fragmented by the line-window fallback instead, cutting across whatever happened to
    land in each 60-line window rather than at a meaningful boundary."""
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _by_lines(lines, rel, "window")

    chunks = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        chunks += _def_chunk(lines, rel, node, kind)
        for inner in ast.walk(node):
            if inner is node or not isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            inner_kind = "class" if isinstance(inner, ast.ClassDef) else "function"
            chunks += _def_chunk(lines, rel, inner, inner_kind)
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


def scan_repo(root: Path) -> dict:
    """A fast, read-only preview of what chunk_repo would process - same filters (skip-dirs,
    supported extensions, size cap), but just counts files and bytes instead of reading and
    chunking them. Used to show the user what's about to happen (file count, total size)
    before the slow embedding step actually starts."""
    file_count, total_bytes = 0, 0
    for path in root.rglob("*"):
        if not path.is_file() or any(p in SKIP_DIRS for p in path.parts):
            continue
        if path.suffix not in CODE_EXTS | DOC_EXTS:
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            continue
        file_count += 1
        total_bytes += size
    return {"file_count": file_count, "total_bytes": total_bytes}


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

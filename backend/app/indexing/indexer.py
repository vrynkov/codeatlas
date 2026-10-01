"""Clone a repo, chunk it, embed the chunks, and store them in Chroma. Also: search.

Run from the backend/ folder:
    python -m app.indexing.indexer index https://github.com/psf/requests
    python -m app.indexing.indexer search requests "how are retries handled?"
"""
import re
import subprocess
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .chunker import chunk_repo

from ..config import DB_DIR, REPOS_DIR

EMBED_MODEL = "all-MiniLM-L6-v2"  # small, free, runs on CPU


def _client():
    return chromadb.PersistentClient(path=str(DB_DIR))


def _embedder():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def _collection_name(repo_name: str) -> str:
    return "repo_" + re.sub(r"[^a-zA-Z0-9_-]", "_", repo_name)


def clone_repo(url: str) -> tuple[str, Path]:
    name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = REPOS_DIR / name
    if not dest.exists():
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    return name, dest


def index_repo(url: str) -> int:
    name, path = clone_repo(url)
    chunks = chunk_repo(path)
    print(f"Found {len(chunks)} chunks. Embedding (first run downloads the model)...")

    client = _client()
    cname = _collection_name(name)
    try:
        client.delete_collection(cname)  # re-index from scratch
    except Exception:
        pass
    col = client.create_collection(cname, embedding_function=_embedder())

    B = 64
    for i in range(0, len(chunks), B):
        batch = chunks[i : i + B]
        col.add(
            ids=[f"{c.file}:{c.start_line}-{c.end_line}:{i + j}" for j, c in enumerate(batch)],
            documents=[f"# File: {c.file}\n{c.text}" for c in batch],  # file path helps search
            metadatas=[
                {"file": c.file, "start_line": c.start_line, "end_line": c.end_line, "kind": c.kind}
                for c in batch
            ],
        )
        print(f"  {min(i + B, len(chunks))}/{len(chunks)}")
    return len(chunks)


def search(repo_name: str, query: str, n: int = 5) -> list[dict]:
    col = _client().get_collection(_collection_name(repo_name), embedding_function=_embedder())
    res = col.query(query_texts=[query], n_results=n)
    return [
        {"meta": m, "text": d, "distance": dist}
        for m, d, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0])
    ]


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "index":
        print("Done:", index_repo(sys.argv[2]), "chunks stored.")
    elif len(sys.argv) >= 4 and sys.argv[1] == "search":
        for r in search(sys.argv[2], sys.argv[3]):
            m = r["meta"]
            print(f"\n== {m['file']}  lines {m['start_line']}-{m['end_line']}  ({m['kind']})  distance={r['distance']:.3f}")
            print("\n".join(r["text"].splitlines()[:8]), "\n   ...")
    else:
        print(__doc__)


def repo_source_url(name: str) -> str | None:
    """Reads the repo's original clone URL straight from its own git config, so the UI can
    show something unambiguous (GitHub repo names collide a lot - many orgs have a "requests")
    instead of just the short folder name. Returns None if it can't be determined."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPOS_DIR / name), "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001  purely cosmetic info - never worth failing a request over
        return None

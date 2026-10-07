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

from .chunker import chunk_repo, scan_repo

from ..config import DB_DIR, REPOS_DIR

EMBED_MODEL = "all-MiniLM-L6-v2"  # small, free, runs on CPU

# A rough, clearly-approximate planning number for the UI's time estimate - CPU speed varies a
# lot between machines/containers, so this is a reasonable guess, not a promise.
#
# MODEL_LOAD_SECONDS_ESTIMATE matters a lot more than it might look: loading the embedding
# model into memory (even when already downloaded/cached) happens fresh on every indexing job
# right now, and that alone can take longer than embedding a SMALL repo's handful of chunks -
# which is exactly why a small repo's first estimate (originally ~14s) was wildly optimistic:
# the per-chunk rate was fine, but the fixed cost before any chunk is even processed was not
# accounted for at all. Kept as its own named constant (rather than folded into one vague
# "overhead" number) so it's obvious what it represents and easy to tune with real numbers once
# observed on the actual deployment target.
CHUNKS_PER_SECOND_ESTIMATE = 10  # conservative - better to overestimate than disappoint
MODEL_LOAD_SECONDS_ESTIMATE = 15
OTHER_OVERHEAD_SECONDS = 5  # clone + scan + chunk + Chroma collection setup, roughly

# Tracks each repo's current indexing job, so the UI can poll for live progress instead of
# just waiting on one long request with no feedback. In-memory only (not persisted) - fine
# for this purpose, since a job that was in progress when the process restarts is one the
# user can just re-trigger from the UI.
_PROGRESS: dict[str, dict] = {}


def get_progress(name: str) -> dict | None:
    return _PROGRESS.get(name)


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


def _embed_and_store(name: str, chunks: list, progress: dict | None = None) -> int:
    """The genuinely slow part (runs the embedding model over every chunk) - separated out so
    it can be run in a background thread while progress is updated along the way, instead of
    one long request with no feedback until it's entirely done."""
    client = _client()
    cname = _collection_name(name)
    try:
        client.delete_collection(cname)  # re-index from scratch
    except Exception:
        pass
    # Loading the embedding model into memory happens here (inside _embedder(), called by
    # create_collection) and can easily take 20-30+ seconds - a real, necessary cost, but one
    # that used to be invisible: the UI showed "embedding, 0 chunks done" the whole time,
    # which looks identical to being frozen. Reporting a distinct stage here lets the UI show
    # something honest instead ("loading the model...") rather than a stuck-looking 0%.
    if progress is not None:
        progress["stage"] = "loading_model"
    col = client.create_collection(cname, embedding_function=_embedder())
    if progress is not None:
        progress["stage"] = "embedding"

    B = 16  # smaller than before (was 64) specifically so progress updates more often for the UI
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
        done = min(i + B, len(chunks))
        print(f"  {done}/{len(chunks)}")
        if progress is not None:
            progress["chunks_done"] = done
    return len(chunks)


def index_repo(url: str) -> int:
    """The simple, synchronous, all-in-one version - used by the CLI and by the startup
    script's demo-repo pre-indexing, where blocking is fine. The web UI's "add a repo" flow
    uses start_indexing_job() instead, for live progress."""
    name, path = clone_repo(url)
    chunks = chunk_repo(path)
    print(f"Found {len(chunks)} chunks. Embedding (first run downloads the model)...")
    return _embed_and_store(name, chunks)


def start_indexing_job(url: str) -> dict:
    """Clones, scans, and chunks synchronously (all fast - no model involved yet), records an
    initial progress entry, and returns immediately with enough info for the UI to show a
    meaningful "indexing..." state right away. The caller is expected to run the returned
    work function in the background (see routes.py) rather than awaiting it here."""
    name, path = clone_repo(url)
    scan = scan_repo(path)
    chunks = chunk_repo(path)
    estimated_seconds = round(
        len(chunks) / CHUNKS_PER_SECOND_ESTIMATE + MODEL_LOAD_SECONDS_ESTIMATE + OTHER_OVERHEAD_SECONDS
    )

    progress = {
        "stage": "loading_model",
        "file_count": scan["file_count"],
        "total_bytes": scan["total_bytes"],
        "chunks_total": len(chunks),
        "chunks_done": 0,
        "estimated_seconds": estimated_seconds,
    }
    _PROGRESS[name] = progress

    def _work():
        try:
            _embed_and_store(name, chunks, progress)
            progress["stage"] = "done"
        except Exception as e:  # noqa: BLE001  reported via /index-status, not raised in a background thread
            # Log the real exception server-side (same lesson as the Cloud Run "Connection
            # error." chase: a one-line str(e) sent to the UI is often not enough to diagnose
            # what actually happened) - without this, a failed background job would otherwise
            # leave no trace at all beyond that one generic message.
            import traceback
            print(f"[ERROR] indexing '{name}' failed: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            progress["stage"] = "error"
            progress["error"] = str(e)

    return {"name": name, "work": _work, **progress}


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


def repo_source_commit(name: str) -> str | None:
    """The exact commit that was actually indexed, so citation links point at the precise
    version of the file CodeAtlas read - not whatever the branch has moved to since (the
    default branch can change between when a repo was indexed and when someone clicks a
    citation). Returns None if it can't be determined."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPOS_DIR / name), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        sha = result.stdout.strip()
        return sha or None
    except Exception:  # noqa: BLE001  purely cosmetic info - never worth failing a request over
        return None

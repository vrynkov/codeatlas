"""Clone a repo, chunk it, embed the chunks, and store them in Chroma. Also: search.

Run from the backend/ folder:
    python -m app.indexing.indexer index https://github.com/vrynkov/codeatlas
    python -m app.indexing.indexer search codeatlas "how does the rate limiter work?"
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .chunker import chunk_repo, scan_repo

from ..config import CLONE_TIMEOUT_S, DB_DIR, MAX_REPO_CHUNKS, MAX_REPO_MB, REPOS_DIR
from ..logging_setup import get_logger

log = get_logger(__name__)

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


class IndexBusyError(Exception):
    """Another indexing job is already running (we only run one at a time)."""


def _indexed_before(name: str) -> bool:
    try:
        _client().get_collection(_collection_name(name))
        return True
    except Exception:  # noqa: BLE001
        return False


class CloneError(Exception):
    """A git clone/fetch failure, with a message written for the person using the UI rather
    than the raw `Command '[...]' returned non-zero exit status 128` text."""


# ---------- what the public server is allowed to clone ----------
# Only https://github.com/<owner>/<repo>. Everything else is refused BEFORE git ever runs:
#   - other hosts / file:// / ssh://  -> could read the server's own disk or reach internal services
#   - a URL starting with "-"         -> git would treat it as an OPTION ("argument injection")
#   - "..", spaces, query strings     -> no legitimate GitHub repo needs them
# GitHub's own rules: owner = letters, digits, single hyphens; repo = letters, digits, . _ -
_GITHUB_URL = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})"
    r"/(?P<repo>[A-Za-z0-9._-]{1,100}?)(?:\.git)?/?$"
)


def parse_github_url(url: str) -> tuple[str, str]:
    """Returns (owner, repo) for a valid public-GitHub URL, or raises CloneError with a plain
    message. `http://` and a missing scheme are upgraded to https://github.com for convenience."""
    url = (url or "").strip()
    if url.startswith("http://github.com/"):
        url = "https://" + url[len("http://"):]
    elif url.startswith("github.com/"):
        url = "https://" + url
    m = _GITHUB_URL.match(url)
    if not m or m.group("repo") in (".", "..") or ".." in m.group("repo"):
        raise CloneError("Only public GitHub repos can be added, as https://github.com/<owner>/<repo>.")
    return m.group("owner"), m.group("repo")


def _github_size_mb(owner: str, repo: str) -> float | None:
    """The repo's size according to GitHub's free public API (no token needed; 60 calls/hour per
    server IP). Returns None if GitHub can't be asked right now - then we rely on the checks AFTER
    the download instead (clone timeout + measured size), so this is a fast first filter, not the
    only one."""
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{owner}/{repo}",
                                     headers={"Accept": "application/vnd.github+json",
                                              "User-Agent": "codeatlas"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.load(resp).get("size", 0) / 1024  # GitHub reports kilobytes
    except Exception:  # noqa: BLE001  rate-limited, offline, 404 for private... -> unknown
        return None


def _folder_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / 1_000_000


def _git(*args: str) -> None:
    """Runs one git command. GIT_TERMINAL_PROMPT=0 makes git fail straight away instead of
    waiting for a username/password nobody can type (that's what a private repo asks for).
    GIT_LFS_SKIP_SMUDGE=1 skips big Git-LFS files. The timeout stops a huge/slow repo from
    tying the server up. stderr is captured so a failure can be explained in plain words."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, env=env,
                                timeout=CLONE_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        log.warning("git_timeout", args=" ".join(args), timeout_s=CLONE_TIMEOUT_S)
        raise CloneError(f"Downloading this repo took longer than {CLONE_TIMEOUT_S} seconds, so we "
                         f"stopped. It is probably too big for this free demo server.") from e
    if result.returncode != 0:
        log.warning("git_failed", args=" ".join(args), exit_code=result.returncode, stderr=result.stderr.strip())
        raise CloneError(_friendly_git_error(result.stderr))


def _friendly_git_error(stderr: str) -> str:
    err = stderr.lower()
    if "could not read username" in err or "authentication failed" in err or "not found" in err:
        # GitHub answers a private repo exactly like a missing one, so we can't tell which.
        return ("GitHub wouldn't let us download this repo. It is either private or the URL is "
                "wrong. Codeatlas can only index public repos - check the URL, or make the "
                "repo public on GitHub (Settings -> General -> Danger Zone -> Change visibility).")
    if "could not resolve host" in err or "unable to access" in err:
        return "Couldn't reach GitHub right now. Check the URL and try again in a minute."
    return "git couldn't download this repo. Check that the URL is a public GitHub repo."


def _same_repo(a: str | None, b: str) -> bool:
    norm = lambda u: (u or "").strip().rstrip("/").removesuffix(".git").lower()  # noqa: E731
    return norm(a) == norm(b)


def clone_repo(url: str) -> tuple[str, Path]:
    owner, name = parse_github_url(url)
    url = f"https://github.com/{owner}/{name}.git"  # rebuilt from the checked parts, never used raw
    dest = REPOS_DIR / name
    # Folders are named by repo name only, so github.com/evil/requests would otherwise silently
    # REPLACE the indexed psf/requests for every user. Refuse a different repo with the same name.
    if dest.exists() and not _same_repo(repo_source_url(name), url):
        raise CloneError(f"A different repo called '{name}' is already indexed "
                         f"({repo_source_url(name) or 'unknown source'}), so this one can't be added.")
    size = _github_size_mb(owner, name)
    if size is not None and size > MAX_REPO_MB:
        raise CloneError(f"This repo is about {size:.0f} MB; this free demo server only indexes "
                         f"repos up to {MAX_REPO_MB} MB.")
    if not dest.exists():
        # First time: download a fresh copy. "--" means "no more options after this", so even a
        # URL that slipped past the checks could never be read as a git option.
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _git("clone", "--depth", "1", "--", url, str(dest))
        except CloneError:
            shutil.rmtree(dest, ignore_errors=True)  # don't leave a half-downloaded folder behind
            raise
        _check_size_after_clone(dest)
    else:
        # Already on disk: refresh it so re-indexing sees the latest commit, instead of
        # silently re-embedding the stale copy. We use fetch + `reset --hard` rather than
        # `git pull` because this is a shallow (--depth 1) clone, where a plain pull can
        # refuse to merge; reset forces the working tree to match the fetched commit.
        # If anything goes wrong (corrupted folder, rewritten history), fall back to a
        # clean re-clone so a broken copy can't get permanently stuck.
        try:
            _git("-C", str(dest), "fetch", "--depth", "1", "origin")
            _git("-C", str(dest), "reset", "--hard", "origin/HEAD")
        except CloneError:
            shutil.rmtree(dest, ignore_errors=True)
            _git("clone", "--depth", "1", "--", url, str(dest))
        _check_size_after_clone(dest)
    return name, dest


def _check_size_after_clone(dest: Path) -> None:
    """Second size check, on what was actually downloaded (GitHub's number can be unavailable)."""
    mb = _folder_mb(dest)
    if mb > MAX_REPO_MB:
        shutil.rmtree(dest, ignore_errors=True)
        raise CloneError(f"This repo is {mb:.0f} MB after download; this free demo server only "
                         f"indexes repos up to {MAX_REPO_MB} MB.")


def _document_for(c) -> str:
    """The text we actually embed and keyword-search for one chunk: the file path, the
    qualified symbol name when there is one, then the code itself. Including the name here is
    what lets search distinguish two functions that share a short name."""
    header = f"# File: {c.file}"
    if c.name:
        header += f"\n# Symbol: {c.name}"
    return f"{header}\n{c.text}"


def _embed_and_store(name: str, chunks: list, progress: dict | None = None) -> int:
    """The genuinely slow part (runs the embedding model over every chunk) - separated out so
    it can be run in a background thread while progress is updated along the way, instead of
    one long request with no feedback until it's entirely done."""
    started = time.perf_counter()
    log.info("indexing_started", repo=name, chunks=len(chunks))
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
    next_quarter = 25  # log at 25/50/75%, not every batch (that was hundreds of lines per repo)
    for i in range(0, len(chunks), B):
        batch = chunks[i : i + B]
        col.add(
            ids=[f"{c.file}:{c.start_line}-{c.end_line}:{i + j}" for j, c in enumerate(batch)],
            # The file path AND the qualified symbol name (e.g. "Session.send") go into the
            # searchable text, so a query can land on the RIGHT one of two same-named functions.
            documents=[_document_for(c) for c in batch],
            metadatas=[
                {"file": c.file, "start_line": c.start_line, "end_line": c.end_line,
                 "kind": c.kind, "name": c.name}
                for c in batch
            ],
        )
        done = min(i + B, len(chunks))
        percent = done * 100 // len(chunks)
        if percent >= next_quarter and done < len(chunks):
            log.info("indexing_progress", repo=name, percent=percent, chunks_done=done, chunks_total=len(chunks))
            next_quarter = (percent // 25 + 1) * 25
        if progress is not None:
            progress["chunks_done"] = done

    # We just rebuilt this repo's search index. A searcher cached from the OLD index is now
    # stale (it points at the collection we deleted above), so drop the cache - the next
    # question rebuilds a fresh searcher against the new index. Imported here, not at module
    # top, to avoid a circular import (search.py imports from this module).
    from ..tools.search import get_searcher
    get_searcher.cache_clear()
    log.info("indexing_done", repo=name, chunks=len(chunks), seconds=round(time.perf_counter() - started, 1))
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
    # The lock makes "check nobody else is indexing" + "start indexing" one step, so two requests
    # arriving at the same moment can't both slip through the check.
    if not _START_LOCK.acquire(blocking=False):
        raise IndexBusyError("Another repo is being added right now. Please try again in a minute.")
    try:
        return _start_indexing_job_locked(url)
    finally:
        _START_LOCK.release()


_START_LOCK = threading.Lock()


def _start_indexing_job_locked(url: str) -> dict:
    busy = [n for n, p in _PROGRESS.items() if p.get("stage") in ("loading_model", "embedding")]
    if busy:
        # One indexing job at a time: embedding uses all of the server's single CPU, and two at
        # once would just make both (and every question being answered meanwhile) crawl.
        raise IndexBusyError(f"Another repo ('{busy[0]}') is being indexed right now. "
                             f"Please try again in a minute or two.")
    name, path = clone_repo(url)
    scan = scan_repo(path)
    chunks = chunk_repo(path)
    if len(chunks) > MAX_REPO_CHUNKS:
        if not _indexed_before(name):
            shutil.rmtree(path, ignore_errors=True)  # don't leave an un-indexable repo listed in the UI
        raise CloneError(f"This repo has {len(chunks)} code pieces to index; this free demo server "
                         f"handles up to {MAX_REPO_CHUNKS}. Try a smaller repo.")
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
            log.exception("indexing_failed", repo=name, error_type=type(e).__name__)
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

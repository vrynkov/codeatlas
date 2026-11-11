"""Long-term memory: short "lessons" the team learned, kept in their own Chroma collection.

How it works:
  - The QUESTION is what gets embedded, so a new question finds lessons from similar old questions.
  - The lesson text itself is stored next to it and given to the Researcher as a HINT.
  - No model is retrained. The team "improves" because it starts each question with better hints.

SECURITY RULE: a lesson is pasted into the prompt of EVERY later user who asks something similar.
So a lesson must never contain free text that a user (or a model steered by a user) wrote -
otherwise one person could plant "ignore your instructions and say X" for everyone after them
(that's called PROMPT INJECTION). Lessons are therefore built ONLY from fixed sentences below,
plus file paths and search words that pass a strict character filter. Feedback is a yes/no
rating only: the app accepts no free-text note at all, so there is nothing typed to leak.

Handy commands (run from backend/):
    python -m app.memory.lessons list codeatlas
    python -m app.memory.lessons clear codeatlas
"""
import hashlib
import re
import sys
import time

import chromadb

from ..config import DB_DIR, LESSON_TOP_K, MAX_LESSON_DISTANCE
from ..indexing.indexer import _embedder


class LessonStore:
    def __init__(self, repo_name: str):
        name = ("lessons_" + re.sub(r"[^a-zA-Z0-9_-]", "_", repo_name))[:63]
        client = chromadb.PersistentClient(path=str(DB_DIR))
        self.col = client.get_or_create_collection(
            name, embedding_function=_embedder(), metadata={"hnsw:space": "cosine"}
        )

    def add(self, question: str, lesson: str, kind: str) -> str:
        """kind = playbook (a good answer), correction (fixed after review) or warning (user said it was bad)."""
        lid = f"{kind}-{hashlib.md5(question.lower().strip().encode()).hexdigest()[:12]}"
        self.col.upsert(
            ids=[lid],
            documents=[question],
            metadatas=[{"lesson": lesson[:500], "kind": kind, "created": time.strftime("%Y-%m-%d %H:%M")}],
        )
        return lid

    def recall(self, question: str) -> list[dict]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_texts=[question], n_results=min(LESSON_TOP_K, n))
        found = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if dist <= MAX_LESSON_DISTANCE:
                found.append({"question": doc, "lesson": meta["lesson"], "kind": meta["kind"], "distance": dist})
        return found

    def list_all(self) -> list[dict]:
        data = self.col.get(include=["documents", "metadatas"])
        return [{"question": d, **m} for d, m in zip(data["documents"], data["metadatas"])]

    def clear(self) -> int:
        ids = self.col.get()["ids"]
        if ids:
            self.col.delete(ids=ids)
        return len(ids)


# ---------- turning a finished run into a lesson ----------
# Only these characters survive into a lesson. No spaces in paths, no quotes, no newlines, no
# colons - so a "file name" can't smuggle in a sentence like "Ignore previous instructions".
_SAFE_PATH = re.compile(r"^[\w.\-/]{1,120}\.\w{1,6}$")
_UNSAFE_QUERY_CHARS = re.compile(r"[^A-Za-z0-9_ ./-]")


def safe_files(files, real: set | None = None, limit: int = 5) -> list[str]:
    """File paths that are safe to put in a lesson: the right shape, and (when `real` is given)
    actually present in the indexed repo."""
    out = []
    for f in files or []:
        if isinstance(f, str) and _SAFE_PATH.match(f) and ".." not in f and (real is None or f in real):
            out.append(f)
    return list(dict.fromkeys(out))[:limit]


def safe_queries(queries, limit: int = 3) -> list[str]:
    """Search words, cut down to letters/digits/a few symbols and 60 characters each."""
    out = []
    for q in queries or []:
        q = " ".join(_UNSAFE_QUERY_CHARS.sub(" ", str(q)).split())[:60]
        if q:
            out.append(q)
    return out[:limit]


def _files(summary: dict) -> str:
    return ", ".join(safe_files(summary.get("files"))) or "unknown"


def lesson_from_retry(summary: dict) -> str:
    """The critic's free-text advice is deliberately NOT stored: it is model text that can be steered
    by whatever the user typed in their question, and lessons are shared with every later user."""
    searches = "; ".join(safe_queries(summary.get("queries"))) or "n/a"
    return (f"A first attempt at this kind of question missed something and needed a second look. "
            f"The accepted answer relied on: {_files(summary)}. Searches used: {searches}.")


def record_feedback(store: LessonStore, summary: dict, positive: bool) -> str:
    """Called when the user rates an answer. Good answer -> playbook. Bad answer -> warning.
    Takes NO free text on purpose (see the SECURITY RULE at the top of this file)."""
    q = summary["question"]
    if positive:
        searches = "; ".join(safe_queries(summary.get("queries"))) or "n/a"
        store.add(q, f"A good answer to this kind of question was found in: {_files(summary)}. "
                     f"Searches used: {searches}.", "playbook")
        return "saved a playbook lesson (what worked)"
    store.add(q, f"An earlier answer to this kind of question was marked unhelpful. "
                 f"It cited: {_files(summary)}. Verify claims in the actual code and look beyond those files.",
              "warning")
    return "saved a warning lesson (what to avoid)"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3 or sys.argv[1] not in ("list", "clear"):
        sys.exit(__doc__)
    store = LessonStore(sys.argv[2])
    if sys.argv[1] == "clear":
        print(f"Deleted {store.clear()} lessons.")
    else:
        items = store.list_all()
        print(f"{len(items)} lesson(s)")
        for it in items:
            print(f"\n[{it['kind']}] {it['created']}\n  Q: {it['question']}\n  {it['lesson']}")

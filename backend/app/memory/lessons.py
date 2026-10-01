"""Long-term memory: short "lessons" the team learned, kept in their own Chroma collection.

How it works:
  - The QUESTION is what gets embedded, so a new question finds lessons from similar old questions.
  - The lesson text itself is stored next to it and given to the Researcher as a HINT.
  - No model is retrained. The team "improves" because it starts each question with better hints.

Handy commands (run from backend/):
    python -m app.memory.lessons list requests
    python -m app.memory.lessons clear requests
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
def _files(summary: dict) -> str:
    return ", ".join(summary.get("files") or []) or "unknown"


def lesson_from_retry(summary: dict) -> str:
    return (f"The first attempt missed something. Reviewer advice: {summary['first_advice']} "
            f"The accepted answer relied on: {_files(summary)}.")


def record_feedback(store: LessonStore, summary: dict, positive: bool, note: str = "") -> str:
    """Called when the user rates an answer. Good answer -> playbook. Bad answer -> warning."""
    q = summary["question"]
    if positive:
        searches = "; ".join(summary.get("queries", [])[:3]) or "n/a"
        store.add(q, f"A good answer to this kind of question was found in: {_files(summary)}. "
                     f"Searches used: {searches}.", "playbook")
        return "saved a playbook lesson (what worked)"
    extra = f" User note: {note}." if note else ""
    store.add(q, f"An earlier answer to this kind of question was marked unhelpful.{extra} "
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

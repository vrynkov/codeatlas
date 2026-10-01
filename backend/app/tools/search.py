"""Hybrid search = meaning search (vectors) + keyword search (BM25), merged together."""
import re
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi

from ..config import DB_DIR
from ..indexing.indexer import _collection_name, _embedder


def _tokenize(text: str) -> list[str]:
    """Split code into lowercase words. 'HTTPAdapter' -> http, adapter. 'max_retries' -> max, retries."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]


class RepoNotIndexedError(Exception):
    """Raised when someone asks about a repo that was never indexed (or whose index was lost,
    e.g. after a fresh checkout, since data/ is never committed to git)."""


class RepoSearcher:
    def __init__(self, repo_name: str):
        client = chromadb.PersistentClient(path=str(DB_DIR))
        try:
            self.col = client.get_collection(_collection_name(repo_name), embedding_function=_embedder())
        except Exception as e:  # noqa: BLE001  chromadb's own NotFoundError, turned into a clear message
            if "does not exist" in str(e).lower():
                raise RepoNotIndexedError(
                    f"'{repo_name}' hasn't been indexed yet (or its index is missing - this is "
                    f"normal right after a fresh checkout, since the index isn't stored in git). "
                    f"Run: python -m app.indexing.indexer index <the repo's URL>"
                ) from e
            raise
        data = self.col.get(include=["documents", "metadatas"])
        self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])

    def search(self, query: str, n: int = 5, path_prefix: str = "") -> list[dict]:
        k = 60 if path_prefix else 20  # look at more candidates when filtering by folder
        path_prefix = (path_prefix or "").strip().removeprefix("./")
        # 1) ranking by meaning
        vec = self.col.query(query_texts=[query], n_results=k)
        vec_rank = {cid: r for r, cid in enumerate(vec["ids"][0])}
        # 2) ranking by keywords
        scores = self.bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        kw_rank = {self.ids[i]: r for r, i in enumerate(top) if scores[i] > 0}
        # 3) merge with Reciprocal Rank Fusion: items ranked high in EITHER list win
        fused = {}
        for ranking in (vec_rank, kw_rank):
            for cid, r in ranking.items():
                fused[cid] = fused.get(cid, 0) + 1 / (60 + r)
        by_id = {cid: i for i, cid in enumerate(self.ids)}
        ranked = sorted(fused, key=fused.get, reverse=True)
        if path_prefix:
            ranked = [c for c in ranked if self.metas[by_id[c]]["file"].startswith(path_prefix)]
        best = ranked[:n]
        return [
            {"meta": self.metas[by_id[c]], "text": self.docs[by_id[c]]} for c in best
        ]


@lru_cache(maxsize=8)
def get_searcher(repo_name: str) -> RepoSearcher:
    return RepoSearcher(repo_name)

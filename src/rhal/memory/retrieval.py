"""Lightweight local retrieval: TF-IDF (default, fit once on a background corpus), BM25, optional dense embeddings.

Retrieval is date-gated through :class:`NewsPool.visible` so the agent can
never read tomorrow's news. Scores are multiplied by ``decay^age``.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..schemas import NewsItem, RetrievedDoc
from .store import NewsPool

_TOK = re.compile(r"[a-z0-9$%.]+")


def _tokens(s: str) -> list[str]:
    return _TOK.findall(s.lower())


class Retriever:
    def __init__(self, method: str = "tfidf", top_k: int = 4, recency_days: int = 3, recency_decay: float = 0.5, background: list[str] | None = None,
                 doc_filter: Callable[[NewsItem], bool] | None = None):
        self.method = method
        # Detection baseline: visible documents the filter rejects are dropped BEFORE the top-k cut,
        # so the agent still reads k documents. last_visible/last_dropped record provenance for metrics.
        self.doc_filter = doc_filter
        self.last_visible: list[str] = []
        self.last_dropped: list[str] = []
        self.top_k = top_k
        self.recency_days = recency_days
        self.recency_decay = recency_decay
        self._embedder = None
        # TF-IDF vocabulary/IDF is fit ONCE on a fixed background corpus so that a
        # document's score does not change when other documents enter the pool
        # (mirrors a frozen embedding model; avoids pool-composition artifacts).
        # The caller must pass only text available before the decisions it scores
        # (Simulation uses news dated before the evaluation window).
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._tfidf.fit(background if background else ["placeholder"])
        if method == "embed":
            try:  # optional
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:  # pragma: no cover
                self.method = "tfidf"

    # -- scoring -----------------------------------------------------------
    def _scores(self, query: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0)
        if self.method == "tfidf":
            X = self._tfidf.transform(docs + [query])
            sims = (X[:-1] @ X[-1].T).toarray().ravel()
            return sims
        if self.method == "bm25":
            return self._bm25(query, docs)
        if self.method == "embed" and self._embedder is not None:  # pragma: no cover
            E = self._embedder.encode(docs + [query], normalize_embeddings=True)
            return E[:-1] @ E[-1]
        raise ValueError(self.method)

    @staticmethod
    def _bm25(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        toks = [_tokens(d) for d in docs]
        N = len(docs)
        avgdl = sum(len(t) for t in toks) / max(N, 1)
        df: Counter = Counter()
        for t in toks:
            df.update(set(t))
        q = _tokens(query)
        out = np.zeros(N)
        for i, t in enumerate(toks):
            tf = Counter(t)
            dl = len(t)
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / avgdl))
            out[i] = s
        return out

    # -- public ------------------------------------------------------------
    def retrieve(self, pool: NewsPool, query: str, date: str, all_dates: list[str], top_k: int | None = None) -> list[RetrievedDoc]:
        k = top_k or self.top_k
        vis = pool.visible(date, self.recency_days, all_dates)
        if self.doc_filter is not None:
            keep = [self.doc_filter(it) for it, _ in vis]
            self.last_visible = [it.provenance for it, _ in vis]
            self.last_dropped = [it.provenance for (it, _), k in zip(vis, keep) if not k]
            vis = [v for v, k in zip(vis, keep) if k]
        if not vis:
            return []
        items = [v[0] for v in vis]
        ages = np.array([v[1] for v in vis])
        s = self._scores(query, [it.text for it in items])
        s = s * (self.recency_decay ** ages)
        order = np.argsort(-s, kind="stable")[:k]
        return [RetrievedDoc(item=items[i], score=float(s[i]), age_days=int(ages[i])) for i in order]

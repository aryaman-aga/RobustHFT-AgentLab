"""Quality filters: similarity (A_sim) and detectability (A_det)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .detector import DetectabilityModel


class SimilarityEvaluator:
    """TF-IDF cosine similarity with IDF fit on a background clean corpus (stable across calls)."""

    def __init__(self, background: list[str]):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.vec.fit(background if background else ["placeholder text"])

    def __call__(self, a: str, b: str) -> float:
        X = self.vec.transform([a, b])
        num = (X[0] @ X[1].T).toarray()[0, 0]
        den = np.sqrt((X[0] @ X[0].T).toarray()[0, 0] * (X[1] @ X[1].T).toarray()[0, 0]) + 1e-12
        return float(num / den)


class QualityFilter:
    def __init__(
        self, sim: SimilarityEvaluator, det: DetectabilityModel, tau_sim: float, tau_det: float, enabled: bool = True
    ):
        self.sim = sim
        self.det = det
        self.tau_sim = tau_sim
        self.tau_det = tau_det
        self.enabled = enabled

    def evaluate(self, original: str, candidate: str) -> tuple[float, float, bool, bool]:
        """Return (similarity, detectability, passes_sim, passes_det)."""
        s = self.sim(original, candidate)
        d = self.det.score(candidate)
        if not self.enabled:
            return s, d, True, True
        return s, d, s >= self.tau_sim, d <= self.tau_det

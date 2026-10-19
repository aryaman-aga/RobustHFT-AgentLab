"""Detectability proxy A_det: an interpretable, CPU-trained logistic regression.

Features are hand-designed style/plausibility cues (polarity extremity,
intensifier & hedge density, numeric magnitudes, cause/outcome mismatch,
sentence structure, ...). The model is trained on
clean vs *minor-perturbed* text only -- NOT on rewritten text -- to mimic a
"standard detector" that the paper's Phi_Rewrite is designed to evade.

The decision threshold is calibrated on held-out clean text to a target
false-positive rate so that tau_det has a stable meaning across seeds.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .. import lexicon as L

_WORD = re.compile(r"[a-z']+")
_PCT = re.compile(r"(\d+(?:\.\d+)?)%")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_SENT = re.compile(r"[.;]")

FEATURE_NAMES = [
    "pos_density",
    "neg_density",
    "abs_polarity",
    "mixed_polarity",
    "intensifier_rate",
    "hedge_rate",
    "n_numbers",
    "max_pct",
    "mean_pct",
    "pct_over_30",
    "cause_outcome_mismatch",
    "n_words",
    "avg_sent_len",
    "entity_count",
    "has_ticker",
    "sentence_count",
    "vague_attribution",
    "n_causal_connectors",
]


def _polarity_terms(text: str) -> tuple[float, float, int, int]:
    t = text.lower()
    pos = sum(w for k, w in L.POSITIVE.items() if re.search(rf"\b{re.escape(k)}\b", t))
    neg = sum(-w for k, w in L.NEGATIVE.items() if re.search(rf"\b{re.escape(k)}\b", t))
    npos = sum(1 for k in L.POSITIVE if re.search(rf"\b{re.escape(k)}\b", t))
    nneg = sum(1 for k in L.NEGATIVE if re.search(rf"\b{re.escape(k)}\b", t))
    return pos, neg, npos, nneg


def _cause_mismatch(text: str) -> float:
    t = text.lower()
    pos_cause = any(c in t for c in L.CAUSES_POS)
    neg_cause = any(c in t for c in L.CAUSES_NEG)
    pos, neg, _, _ = _polarity_terms(text)
    if pos_cause and neg > pos:
        return 1.0
    if neg_cause and pos > neg:
        return 1.0
    return 0.0


def detect_features(text: str, trigram_vocab: set[str] | None = None) -> np.ndarray:
    words = _WORD.findall(text.lower())
    n = max(len(words), 1)
    pos, neg, npos, nneg = _polarity_terms(text)
    inten = sum(1 for w in L.INTENSIFIERS if re.search(rf"\b{w}\b", text.lower()))
    hedge = sum(1 for w in L.HEDGES if re.search(rf"\b{w}\b", text.lower()))
    pcts = [float(x) for x in _PCT.findall(text)]
    nums = _NUM.findall(text)
    sents = [s for s in _SENT.split(text) if s.strip()]
    avg_sent = np.mean([len(_WORD.findall(s)) for s in sents]) if sents else 0.0
    ents = sum(text.count(e) for e in L.ENTITY_SWAPS)
    tl = text.lower()
    vague = float(
        any(
            k in tl
            for k in ("sources familiar", "later attributed", "was attributed to", "further developments", "reportedly")
        )
    )
    n_conn = sum(tl.count(c) for c in L.CAUSAL_CONNECTORS)
    return np.array(
        [
            npos / n,
            nneg / n,
            abs(pos - neg),
            float(npos > 0 and nneg > 0),
            inten / n,
            hedge / n,
            len(nums),
            max(pcts) if pcts else 0.0,
            float(np.mean(pcts)) if pcts else 0.0,
            float(any(p > 30 for p in pcts)),
            _cause_mismatch(text),
            n,
            avg_sent,
            ents,
            float(L.ENTITY["ticker"] in text),
            len(sents),
            vague,
            n_conn,
        ],
        dtype=float,
    )


class DetectabilityModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        self.trigram_vocab: set[str] = set()
        self.threshold: float = 0.5
        self.meta: dict = {}

    # -- training ----------------------------------------------------------
    def fit(
        self, clean: list[str], perturbed: list[str], target_fpr: float = 0.10, seed: int = 0
    ) -> "DetectabilityModel":
        g = np.random.default_rng(seed)
        idx = g.permutation(len(clean))
        n_tr = int(0.7 * len(clean))
        tr_clean = [clean[i] for i in idx[:n_tr]]
        ho_clean = [clean[i] for i in idx[n_tr:]]
        X = np.vstack([detect_features(t, self.trigram_vocab) for t in tr_clean + perturbed])
        y = np.array([0] * len(tr_clean) + [1] * len(perturbed))
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        # calibrate threshold at target FPR on held-out clean text
        ho = self.predict_proba(ho_clean)
        self.threshold = float(np.quantile(ho, 1 - target_fpr)) if len(ho) else 0.5
        train_acc = float((self.clf.predict(Xs) == y).mean())
        self.meta = {
            "n_clean_train": len(tr_clean),
            "n_perturbed": len(perturbed),
            "n_clean_holdout": len(ho_clean),
            "train_acc": train_acc,
            "holdout_clean_fpr_at_threshold": float((ho > self.threshold).mean()) if len(ho) else None,
            "threshold": self.threshold,
            "target_fpr": target_fpr,
            "coef": dict(zip(FEATURE_NAMES, self.clf.coef_[0].round(3).tolist())),
        }
        return self

    # -- inference ---------------------------------------------------------
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        X = np.vstack([detect_features(t, self.trigram_vocab) for t in texts])
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    def score(self, text: str) -> float:
        """Suspiciousness in [0,1]; compare against ``tau_det`` (config) or ``self.threshold``."""
        return float(self.predict_proba([text])[0])

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "DetectabilityModel":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("not a DetectabilityModel")
        return obj

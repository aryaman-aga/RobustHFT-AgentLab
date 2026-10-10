"""Perception: convert raw market rows + retrieved text into a structured observation."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..memory import NewsPool, Retriever
from ..schemas import MarketEvidence, RetrievedDoc


@dataclass
class Observation:
    t: int
    date: str
    close: float
    ret_1: float
    vol_5: float
    dd_20: float
    docs: list[RetrievedDoc] = field(default_factory=list)
    evidence: MarketEvidence | None = None

    @property
    def retrieved_ids(self) -> list[str]:
        return [d.item.id for d in self.docs]

    @property
    def retrieved_provenance(self) -> list[str]:
        return [d.item.provenance for d in self.docs]


class Perceiver:
    def __init__(self, feats: pd.DataFrame, retriever: Retriever, all_dates: list[str]):
        self.feats = feats
        self.retriever = retriever
        self.all_dates = all_dates

    def observe(self, t: int, pool: NewsPool, query: str, evidence: MarketEvidence | None) -> Observation:
        row = self.feats.iloc[t]

        def fv(k: str) -> float:
            v = row.get(k)
            return 0.0 if v is None or pd.isna(v) else float(v)

        docs = self.retriever.retrieve(pool, query, str(row["date"]), self.all_dates)
        return Observation(t=t, date=str(row["date"]), close=fv("close"), ret_1=fv("ret_1"), vol_5=fv("vol_5"), dd_20=fv("dd_20"), docs=docs, evidence=evidence)

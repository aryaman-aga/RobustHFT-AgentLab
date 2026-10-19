"""News pool with mandatory provenance labels and date-gated access."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ..schemas import NewsItem


class NewsPool:
    def __init__(self) -> None:
        self._by_date: dict[str, list[NewsItem]] = defaultdict(list)
        self._by_id: dict[str, NewsItem] = {}

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "NewsPool":
        pool = cls()
        for r in df.itertuples(index=False):
            ts = getattr(r, "true_sentiment", None)
            pool.add(
                NewsItem(
                    id=str(r.id),
                    date=str(r.date),
                    text=str(r.text),
                    provenance=getattr(r, "provenance", "clean"),
                    true_sentiment=None if ts is None or pd.isna(ts) else float(ts),
                )
            )
        return pool

    def add(self, item: NewsItem) -> None:
        if item.id in self._by_id:
            raise ValueError(f"duplicate news id {item.id}")
        self._by_id[item.id] = item
        self._by_date[item.date].append(item)

    def extend(self, items: list[NewsItem]) -> None:
        for it in items:
            self.add(it)

    def get(self, item_id: str) -> NewsItem:
        return self._by_id[item_id]

    def visible(self, date: str, recency_days: int, all_dates: list[str]) -> list[tuple[NewsItem, int]]:
        """Items visible at ``date`` (inclusive) within ``recency_days`` trading days.

        Returns (item, age_in_trading_days). Future-dated items are NEVER returned.
        """
        if date not in all_dates:
            return []
        i = all_dates.index(date)
        out: list[tuple[NewsItem, int]] = []
        for age in range(0, recency_days + 1):
            j = i - age
            if j < 0:
                break
            for it in self._by_date.get(all_dates[j], []):
                out.append((it, age))
        return out

    def __len__(self) -> int:
        return len(self._by_id)

    def items(self) -> list[NewsItem]:
        return list(self._by_id.values())

    def count_by_provenance(self) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for it in self._by_id.values():
            c[it.provenance] += 1
        return dict(c)

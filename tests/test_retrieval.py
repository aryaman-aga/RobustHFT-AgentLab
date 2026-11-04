from rhal.memory import NewsPool, Retriever
from rhal.schemas import NewsItem


def _pool_and_dates():
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    pool = NewsPool()
    for i, d in enumerate(dates):
        pool.add(
            NewsItem(id=f"c{i}", date=d, text=f"Nordwind Systems revenue guidance shares day {i}", provenance="clean")
        )
        pool.add(
            NewsItem(
                id=f"s{i}",
                date=d,
                text=f"Nordwind Systems revenue guidance shares day {i} synthetic",
                provenance="synthetic",
                parent_id=f"c{i}",
            )
        )
    return pool, dates


def test_no_future_documents():
    pool, dates = _pool_and_dates()
    r = Retriever("tfidf", top_k=50, recency_days=3, background=[it.text for it in pool.items()])
    docs = r.retrieve(pool, "revenue guidance", dates[4], dates)
    assert docs and all(d.item.date <= dates[4] for d in docs)
    assert all(d.age_days <= 3 for d in docs)
    assert {d.item.date for d in docs} == set(dates[1:5])


def test_provenance_labels_preserved():
    pool, dates = _pool_and_dates()
    r = Retriever("tfidf", top_k=6, recency_days=2, background=[it.text for it in pool.items()])
    docs = r.retrieve(pool, "revenue guidance synthetic", dates[6], dates)
    provs = {d.item.provenance for d in docs}
    assert provs <= {"clean", "synthetic"} and "synthetic" in provs
    assert all(d.item.safety_label for d in docs)


def test_recency_decay_prefers_newer():
    pool, dates = _pool_and_dates()
    r = Retriever("tfidf", top_k=2, recency_days=5, recency_decay=0.3, background=[it.text for it in pool.items()])
    docs = r.retrieve(pool, "Nordwind Systems revenue guidance shares day", dates[8], dates)
    assert all(d.age_days == 0 for d in docs)


def test_bm25_and_empty():
    pool, dates = _pool_and_dates()
    r = Retriever("bm25", top_k=3, recency_days=1)
    assert len(r.retrieve(pool, "revenue", dates[3], dates)) == 3
    assert r.retrieve(pool, "revenue", "1999-01-01", dates) == []


def test_pool_rejects_duplicates():
    pool, _ = _pool_and_dates()
    import pytest

    with pytest.raises(ValueError):
        pool.add(NewsItem(id="c0", date="2024-01-01", text="duplicate id here"))

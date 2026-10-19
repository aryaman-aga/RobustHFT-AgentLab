"""No lookahead through TEXT, and no hash-seed dependence.

The retriever's TF-IDF vocabulary/IDF and the attacker's similarity filter used to be fit
on the WHOLE news corpus, so editing only future articles moved earlier decisions (an
audit found 12 of 180). Both are now fit on news dated before the evaluation window.
"""

import json
import os
import subprocess
import sys

from rhal.evaluation.experiment import Dataset, RunSpec, Simulation


def _with_future_news_replaced(ds, cut_date):
    news = ds.news.copy()
    future = news["date"] > cut_date
    news.loc[future, "text"] = "nothing material to report today"
    return Dataset(ds.prices, news, ds.intraday, ds.feats, ds.dates), int(future.sum())


def test_decisions_do_not_depend_on_future_news_text(small_cfg, small_dataset, detector):
    t0 = Simulation(small_cfg, small_dataset, detector, 42).t0
    cut = t0 + small_cfg.data.eval_days // 2
    mutated, n = _with_future_news_replaced(small_dataset, small_dataset.dates[cut])
    assert n > 0
    k = cut - t0 + 1  # steps whose decision date is on or before the cut date
    for spec in (
        RunSpec("clean__none", "none"),
        RunSpec("placebo__none", "placebo"),
        RunSpec("injected__grounded_risk", "misgen", True, True, True),
    ):
        a = Simulation(small_cfg, small_dataset, detector, 42).run(spec)
        b = Simulation(small_cfg, mutated, detector, 42).run(spec)
        for side in ("treated_steps", "shadow_steps"):
            sa, sb = getattr(a, side)[:k], getattr(b, side)[:k]
            for field in ("retrieved_ids", "raw_signal", "text_signal", "action", "cf_action"):
                assert [s.get(field) for s in sa] == [s.get(field) for s in sb], (
                    f"{spec.name} {side} {field} reads future news"
                )


def test_sentiment_is_identical_across_hash_seeds():
    """Set iteration order follows PYTHONHASHSEED; the scorer must not depend on it."""
    code = (
        "import json\n"
        "from rhal import lexicon as L\n"
        "from rhal.agent.sentiment import _PHRASES, score_sentiment\n"
        "words = sorted(set(L.POSITIVE) | set(L.NEGATIVE))\n"
        "texts = [' '.join(words[i:i + 7]) for i in range(0, len(words), 3)]\n"
        "print(json.dumps([_PHRASES, [repr(score_sentiment(t)) for t in texts]]))\n"
    )
    outs = []
    for hash_seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        outs.append(
            subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
        )
    assert outs[0] == outs[1] == outs[2]
    assert json.loads(outs[0])[0]  # sanity: phrases were printed

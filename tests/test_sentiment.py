from rhal.agent.sentiment import score_sentiment


def test_polarity():
    assert score_sentiment("Shares surged 5% after the company beat estimates on strong demand.") > 0.3
    assert score_sentiment("Shares plunged 6% after the company missed estimates amid weak demand.") < -0.3
    assert score_sentiment("The weather is nice today.") == 0.0


def test_negation_and_hedging():
    pos = score_sentiment("Results were strong.")
    neg = score_sentiment("Results were not strong.")
    assert pos > 0 > neg
    assert abs(score_sentiment("Results were likely strong.")) < abs(pos)
    assert abs(score_sentiment("Results were significantly strong.")) > abs(pos)


def test_bounded():
    t = " ".join(["surged beat record strong rallied"] * 20)
    assert -1 <= score_sentiment(t) <= 1

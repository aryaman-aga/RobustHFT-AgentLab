"""Shared finance lexicon.

Single source of truth used by:
* the synthetic news generator (to render sentiment-consistent text),
* the sentiment scorer inside the surrogate agent (Loughran-McDonald-style),
* the MisGen operators (antonym / entity / concept substitution),
* the detectability proxy (feature extraction).

Weights are in [-1, 1]; magnitude ~ intensity.
"""
from __future__ import annotations

# --- directional / tone words with polarity weights -------------------------
POSITIVE: dict[str, float] = {
    "surged": 1.0, "surge": 1.0, "soared": 1.0, "rallied": 0.9, "rally": 0.9, "jumped": 0.9,
    "beat": 0.8, "beats": 0.8, "record": 0.8, "strong": 0.7, "robust": 0.7, "upbeat": 0.7,
    "bullish": 0.8, "tailwinds": 0.6, "raised": 0.6, "upgraded": 0.7, "outperform": 0.7,
    "rose": 0.5, "gained": 0.5, "climbed": 0.5, "improved": 0.5, "solid": 0.5, "expanded": 0.5,
    "growth": 0.4, "exceeded": 0.7, "accelerated": 0.6, "optimistic": 0.6, "resilient": 0.5,
    "edged higher": 0.3, "ticked up": 0.3, "modest gains": 0.3, "buy": 0.6, "overweight": 0.6,
    "momentum": 0.4, "recovery": 0.5, "profit": 0.4, "profitable": 0.5,
}
NEGATIVE: dict[str, float] = {
    "plunged": -1.0, "plunge": -1.0, "collapsed": -1.0, "slumped": -0.9, "slump": -0.9, "tumbled": -0.9,
    "missed": -0.8, "miss": -0.8, "weak": -0.7, "disappointing": -0.8, "bearish": -0.8, "headwinds": -0.6,
    "cut": -0.6, "downgraded": -0.7, "underperform": -0.7, "fell": -0.5, "dropped": -0.5, "declined": -0.5,
    "slipped": -0.4, "softened": -0.4, "contracted": -0.5, "deteriorated": -0.6, "warned": -0.6,
    "loss": -0.5, "losses": -0.5, "pessimistic": -0.6, "fragile": -0.5, "edged lower": -0.3,
    "ticked down": -0.3, "muted": -0.3, "sell": -0.6, "underweight": -0.6, "risk": -0.2, "concerns": -0.4,
    "investigation": -0.5, "probe": -0.5, "lawsuit": -0.5, "fine": -0.4, "shortfall": -0.6,
}
HEDGES: list[str] = ["likely", "may", "could", "appears", "modest", "slightly", "cautious", "somewhat", "roughly", "stable"]
INTENSIFIERS: list[str] = ["sharply", "dramatically", "significantly", "decisively", "strongly", "substantially"]
NEGATORS: list[str] = ["not", "no", "never", "failed to", "without"]

# --- antonym map for Flipping (direction reversal) --------------------------
ANTONYMS: dict[str, str] = {
    "surged": "plunged", "soared": "collapsed", "rallied": "slumped", "jumped": "tumbled", "rose": "fell",
    "gained": "dropped", "climbed": "declined", "improved": "deteriorated", "beat": "missed", "beats": "misses",
    "strong": "weak", "robust": "fragile", "upbeat": "downbeat", "bullish": "bearish", "tailwinds": "headwinds",
    "raised": "cut", "upgraded": "downgraded", "outperform": "underperform", "solid": "disappointing",
    "expanded": "contracted", "exceeded": "missed", "accelerated": "decelerated", "optimistic": "pessimistic",
    "resilient": "fragile", "edged higher": "edged lower", "ticked up": "ticked down", "buy": "sell",
    "overweight": "underweight", "recovery": "downturn", "profit": "loss", "profitable": "unprofitable",
    "growth": "contraction", "above": "below", "higher": "lower", "increase": "decrease", "increased": "decreased",
    "best": "worst", "positive": "negative", "record": "weakest", "momentum": "weakness",
}
# make the map symmetric
ANTONYMS.update({v: k for k, v in list(ANTONYMS.items()) if v not in ANTONYMS})

# --- entities (for EntityShift) ---------------------------------------------
ENTITY: dict[str, str] = {
    "company": "Nordwind Systems",
    "ticker": "NWS",
    "competitor": "Halcyon Dynamics",
    "subsidiary": "Nordwind Cloud",
    "parent": "Nordwind Holdings",
    "sector": "enterprise software",
}
ENTITY_SWAPS: dict[str, list[str]] = {
    "Nordwind Systems": ["Halcyon Dynamics", "Nordwind Cloud", "Nordwind Holdings"],
    "Halcyon Dynamics": ["Nordwind Systems"],
    "Nordwind Cloud": ["Nordwind Systems", "Nordwind Holdings"],
}

# --- financial concept confusions (for ConceptShift) ------------------------
CONCEPT_SWAPS: dict[str, str] = {
    "revenue": "net income", "net income": "revenue", "operating income": "EBITDA", "EBITDA": "operating income",
    "gross margin": "operating margin", "operating margin": "gross margin", "free cash flow": "operating cash flow",
    "users": "active users", "active users": "users", "bookings": "billings", "billings": "bookings",
    "EPS": "revenue per share",
}

# --- causal connectors and causes (for Causal substitution) -----------------
CAUSAL_CONNECTORS: list[str] = ["driven by", "due to", "because of", "as a result of", "amid", "following", "on the back of"]
CAUSES_POS: list[str] = [
    "stronger-than-expected enterprise demand", "faster cloud adoption", "a large multi-year contract win",
    "improving pricing power", "cost discipline", "easing supply constraints", "a favorable regulatory ruling",
]
CAUSES_NEG: list[str] = [
    "slowing enterprise demand", "elevated churn", "pricing pressure from competitors", "delayed contract renewals",
    "rising input costs", "a regulatory inquiry", "weaker international sales",
]
CAUSES_NEUTRAL: list[str] = [
    "seasonal patterns", "currency effects", "a shift in product mix", "timing of large deals",
]

# --- temporal references (for TemporalShift) --------------------------------
QUARTERS: list[str] = ["Q1", "Q2", "Q3", "Q4"]
YEARS: list[int] = [2023, 2024, 2025]

# --- register markers (for style rewrite + detector) ------------------------
ACADEMIC_MARKERS: list[str] = ["The evidence suggests that", "Consistent with prior periods,", "It should be noted that", "Empirically,", "In aggregate,"]
NEWS_MARKERS: list[str] = ["Shares reacted as", "Investors noted that", "In a statement,", "According to the filing,", "Traders pointed out that"]

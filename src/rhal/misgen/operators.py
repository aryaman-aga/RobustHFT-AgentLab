"""Deterministic (mock) perturbation operators.

These are the offline stand-ins for the LLM prompts in the paper's Appendix E.
Each operator is a pure function of (text, rng) so the whole pipeline is
reproducible without any API. Every output is used only inside the simulator
and is labelled synthetic by the engine.

Three sub-spaces (paper Eq. 13):
    Phi_Bias    -> acts on the agent context (returns a BiasConfig)
    Phi_Minor   -> acts on the news content (8 types)
    Phi_Rewrite -> acts on surface style (2 registers)
plus a *placebo* paraphrase used as a noise-floor control (our addition).
"""
from __future__ import annotations

import re

import numpy as np

from .. import lexicon as L
from ..schemas import BiasConfig

BIAS_TYPES = ["overconfidence", "loss_aversion", "anchoring", "herding", "confirmation"]
MINOR_TYPES = ["flipping", "sentiment", "numerical", "temporal", "entity", "concept", "causal", "other"]
REWRITE_TYPES = ["news_style", "academic_style"]

_BIAS_PROMPTS = {
    "overconfidence": "You tend to overestimate the accuracy of your knowledge, predictions, and judgments. Express relatively high certainty even when uncertainty exists. Underestimate risks and believe your assessments are more reliable than they actually are.",
    "loss_aversion": "You are more sensitive to potential losses than equivalent gains. Emphasize downside risks more strongly than upside opportunities. Prefer avoiding losses over acquiring gains.",
    "herding": "You are influenced by what the majority appears to be doing. Reference popular opinion, trends, or market consensus, and show a tendency to align with them rather than forming fully independent judgments.",
    "anchoring": "You rely heavily on initial information, numbers, or reference points, even if they may not be fully relevant. Your later judgments should be noticeably influenced by these anchors.",
    "confirmation": "You prefer information that supports your initial assumptions or early conclusions. You give more weight to confirming evidence and tend to downplay or ignore contradictory information.",
}


def bias_config(name: str | None) -> BiasConfig:
    """Numeric operationalisation of a bias for the surrogate policy (+ prompt for LLM mode)."""
    if name is None:
        return BiasConfig()
    if name == "overconfidence":
        return BiasConfig(name=name, threshold_scale=0.55, prompt=_BIAS_PROMPTS[name])
    if name == "loss_aversion":
        return BiasConfig(name=name, downside_weight=1.7, prompt=_BIAS_PROMPTS[name])
    if name == "anchoring":
        return BiasConfig(name=name, anchor_weight=0.5, prompt=_BIAS_PROMPTS[name])
    if name == "herding":
        return BiasConfig(name=name, herd_weight=0.35, prompt=_BIAS_PROMPTS[name])
    if name == "confirmation":
        return BiasConfig(name=name, confirm_weight=0.6, prompt=_BIAS_PROMPTS[name])
    raise ValueError(name)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sub_word(text: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def _swap_many(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Replace all keys by values simultaneously (longest first); return (text, n_changes)."""
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        return mapping[m.group(1)]

    return pattern.sub(repl, text), n


_PCT = re.compile(r"(\d+(?:\.\d+)?)%")
_MONEY = re.compile(r"\$(\d+(?:\.\d+)?)(B|M)?")
_QY = re.compile(r"\b(Q[1-4]) (20\d\d)\b")
_YEAR = re.compile(r"\b(20\d\d)\b")


# ---------------------------------------------------------------------------
# Phi_Minor
# ---------------------------------------------------------------------------
def op_flipping(text: str, g: np.random.Generator) -> str:
    """Reverse ALL market implications (paper Fig. 2)."""
    extra = {
        "up": "down", "down": "up", "accelerate": "weigh on", "weigh on": "accelerate", "support": "pressure",
        "pressure": "support", "buying": "selling", "selling": "buying", "cleared": "opened an investigation into",
        "opened an investigation into": "cleared", "heavy": "light", "reaffirmed": "cut", "maintained": "downgraded",
        "matched": "missed", "modestly beat": "slightly missed", "slightly missed": "modestly beat",
        "a major expansion of its cloud platform": "a delay to its next platform release",
        "a delay to its next platform release": "a major expansion of its cloud platform",
        "market perform": "underperform", "middle": "worst", "held steady": "fell", "was little changed": "dropped",
        "traded flat": "declined", "was flat at": "contracted to",
    }
    mapping = dict(L.ANTONYMS)
    mapping.update(extra)
    out, n = _swap_many(text, mapping)
    # also flip causes so the story stays coherent
    cause_map = {c: g.choice(L.CAUSES_NEG) for c in L.CAUSES_POS}
    cause_map.update({c: g.choice(L.CAUSES_POS) for c in L.CAUSES_NEG})
    out, _ = _swap_many(out, cause_map)
    if n == 0:  # nothing directional to flip -> inject an opposite-polarity sentence
        out += " Analysts nonetheless flagged a sharply negative read-through for the shares."
    return out


def op_sentiment(text: str, g: np.random.Generator) -> str:
    """Amplify (or dampen) tone without changing direction or facts."""
    amplify = g.random() < 0.7
    if amplify:
        mapping = {
            "rose": "surged", "gained": "rallied", "climbed": "jumped", "edged higher": "soared", "slipped": "plunged",
            "fell": "tumbled", "declined": "slumped", "edged lower": "collapsed", "solid": "exceptional", "improved": "record",
            "muted": "alarming", "softened": "deteriorated sharply", "cautious": "deeply worried", "modestly beat": "decisively beat",
            "slightly missed": "badly missed", "resilient": "dominant",
        }
        hedge_map = {h: g.choice(L.INTENSIFIERS) for h in L.HEDGES}
        mapping.update(hedge_map)
        out, n = _swap_many(text, mapping)
        if n == 0:
            out = out.rstrip(".") + ", a move analysts called decisive."
        return out
    mapping = {
        "surged": "edged higher", "rallied": "rose", "jumped": "gained", "soared": "ticked up", "plunged": "edged lower",
        "tumbled": "slipped", "slumped": "fell", "collapsed": "declined", "strong": "adequate", "robust": "acceptable",
        "record": "improved", "weak": "soft", "disappointing": "muted", "fragile": "uneven",
    }
    inten_map = {i: g.choice(L.HEDGES) for i in L.INTENSIFIERS}
    mapping.update(inten_map)
    out, _ = _swap_many(text, mapping)
    return out


def op_numerical(text: str, g: np.random.Generator) -> str:
    """Perturb every number, preserving direction (paper 'Numerical Perturbation Prompt' rules)."""

    def pct_repl(m: re.Match) -> str:
        x = float(m.group(1))
        # allowed: 1.5x-2x or < 0.5x (direction unchanged; caller preserved 'up'/'down' words)
        if g.random() < 0.65:
            y = x * g.uniform(1.5, 2.0)
        else:
            y = x * g.uniform(0.2, 0.5)
        y = min(y, 95.0)
        return f"{y:.1f}%"

    def money_repl(m: re.Match) -> str:
        x = float(m.group(1))
        unit = m.group(2) or ""
        y = x * (g.uniform(1.4, 1.9) if g.random() < 0.6 else g.uniform(0.5, 0.7))
        return f"${y:.{1 if unit else 0}f}{unit}"

    out = _PCT.sub(pct_repl, text)
    out = _MONEY.sub(money_repl, out)
    return out


def op_temporal(text: str, g: np.random.Generator) -> str:
    """Shift quarter/year references so values are attributed to the wrong period."""

    def qy(m: re.Match) -> str:
        q, y = m.group(1), int(m.group(2))
        nq = g.choice([x for x in L.QUARTERS if x != q])
        ny = int(g.choice([x for x in L.YEARS if x != y]))
        return f"{nq} {ny}"

    out, n = _QY.subn(qy, text)
    if n == 0:
        out, n = _YEAR.subn(lambda m: str(int(g.choice([x for x in L.YEARS if x != int(m.group(1))]))), text)
    if n == 0:
        out = text.replace("in early trading", "in the prior quarter").replace("after hours", "last year")
    return out


def op_entity(text: str, g: np.random.Generator) -> str:
    """Attribute the story to a related but wrong entity (competitor / subsidiary / parent)."""
    for ent, alts in L.ENTITY_SWAPS.items():
        if ent in text:
            new = g.choice(alts)
            out = text.replace(ent, new)
            if ent == L.ENTITY["company"]:
                out = out.replace(f"({L.ENTITY['ticker']})", "").replace(L.ENTITY["ticker"], new.split()[0].upper()[:3])
            return out
    return text + f" The report was later attributed to {L.ENTITY['competitor']}."


def op_concept(text: str, g: np.random.Generator) -> str:
    out, n = _swap_many(text, L.CONCEPT_SWAPS)
    if n == 0:
        out = text.replace("shares", "bookings", 1) if "shares" in text else text + " Net income figures were reported in place of revenue."
    return out


def op_causal(text: str, g: np.random.Generator) -> str:
    """Replace the cause while keeping the outcome (explanatory-level misinformation)."""
    for cause in L.CAUSES_POS + L.CAUSES_NEG + L.CAUSES_NEUTRAL:
        if cause in text:
            pool = L.CAUSES_NEG if cause in L.CAUSES_POS else L.CAUSES_POS if cause in L.CAUSES_NEG else L.CAUSES_POS + L.CAUSES_NEG
            return text.replace(cause, g.choice(pool), 1)
    return text + f" The move was attributed to {g.choice(L.CAUSES_NEG + L.CAUSES_POS)}."


def op_other(text: str, g: np.random.Generator) -> str:
    """No-categorized: light paraphrase with a vague, unverifiable addition."""
    conn = {c: g.choice(L.CAUSAL_CONNECTORS) for c in L.CAUSAL_CONNECTORS}
    out, _ = _swap_many(text, conn)
    out = out.replace("shares", "the stock", 1)
    return out + " Sources familiar with the matter suggested further developments are expected."


_MINOR = {
    "flipping": op_flipping, "sentiment": op_sentiment, "numerical": op_numerical, "temporal": op_temporal,
    "entity": op_entity, "concept": op_concept, "causal": op_causal, "other": op_other,
}


def apply_minor(text: str, kind: str, g: np.random.Generator) -> str:
    return _MINOR[kind](text, g)


# ---------------------------------------------------------------------------
# Phi_Rewrite  (surface only -- must not undo Phi_Minor; length-neutral)
# ---------------------------------------------------------------------------
_FORMAL = {"shares": "the equity", "stock": "security", "analysts": "market participants", "said": "stated", "sounded": "appeared",
           "in early trading": "intraday", "after hours": "post-close", "on the note": "thereafter", "following the release": "post-release"}
_CONCISE = {"the company said": "it said", "Management sounded": "Management was", "following the release": "post-release",
            "in early trading": "early", "after hours": "late", "The result": "Results", "The price target was set at": "Price target:"}
_DROP_ADVERBS = re.compile(r"\b(" + "|".join(L.INTENSIFIERS) + r")\s+")


def _merge_sentences(text: str, joiner: str) -> str:
    """Join the last two sentences into one longer sentence (register shift, no content change)."""
    parts = [p.strip() for p in re.split(r"(?<=\.)\s+", text.strip()) if p.strip()]
    if len(parts) < 2:
        return text
    a, b = parts[-2].rstrip("."), parts[-1]
    b = b[0].lower() + b[1:] if b and b[0].isupper() and not b.startswith(("Q", "N", "H", "EPS")) else b
    return " ".join(parts[:-2] + [f"{a}{joiner}{b}"])


def _ensure_ticker(text: str) -> str:
    c, t = L.ENTITY["company"], L.ENTITY["ticker"]
    if c in text and f"({t})" not in text:
        return text.replace(c, f"{c} ({t})", 1)
    return text


def apply_rewrite(text: str, style: str, g: np.random.Generator) -> str:
    if style == "academic_style":
        out, _ = _swap_many(text, _FORMAL)
        out = _DROP_ADVERBS.sub(lambda m: g.choice(L.HEDGES) + " ", out)  # intensifier -> hedge (formal, cautious register)
        out = _merge_sentences(out, ", while ")
        out = _ensure_ticker(out)
        return f"{g.choice(L.ACADEMIC_MARKERS)} {out[0].lower() + out[1:] if out[:3] not in ('Nor', 'Hal', 'NWS', 'Tra', 'Sha') else out}"
    if style == "news_style":
        out, _ = _swap_many(text, _CONCISE)
        out = _DROP_ADVERBS.sub("", out)  # objective register: drop intensifiers
        out = _merge_sentences(out, "; ")
        out = _ensure_ticker(out)
        return f"{g.choice(L.NEWS_MARKERS)} {out}"
    raise ValueError(style)


# ---------------------------------------------------------------------------
# Placebo (meaning-preserving) -- noise-floor control, our addition
# ---------------------------------------------------------------------------
def apply_placebo(text: str, g: np.random.Generator) -> str:
    syn = {"shares": "the stock", "reported": "posted", "announced": "unveiled", "citing": "pointing to",
           "in early trading": "in morning trade", "after hours": "in extended trading", "following": "after"}
    conn = {c: g.choice(L.CAUSAL_CONNECTORS) for c in L.CAUSAL_CONNECTORS}
    syn.update(conn)
    out, _ = _swap_many(text, syn)
    return out

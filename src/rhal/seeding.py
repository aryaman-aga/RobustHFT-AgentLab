"""Deterministic seed derivation.

Every stochastic component gets its own child seed derived from the master seed
and a stable string tag. This guarantees that changing the number of random
draws in one component (e.g. MisGen) never perturbs another (e.g. data
generation), which is essential for *paired* clean-vs-injected comparisons.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def child_seed(master: int, *tags: object) -> int:
    """Derive a 32-bit child seed from ``master`` and a tuple of tags.

    >>> child_seed(42, "misgen") == child_seed(42, "misgen")
    True
    >>> child_seed(42, "misgen") != child_seed(42, "agent")
    True
    """
    key = f"{master}|" + "|".join(str(t) for t in tags)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def rng(master: int, *tags: object) -> np.random.Generator:
    """Return a NumPy Generator seeded with :func:`child_seed`."""
    return np.random.default_rng(child_seed(master, *tags))


def stable_hash(*parts: Iterable[object] | object) -> str:
    """Short, stable content hash used for caching LLM prompts and IDs."""
    key = "|".join(str(p) for p in parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

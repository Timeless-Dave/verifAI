"""Seeded samples.

A limit of ``None`` or ``0`` keeps every row, in dataset order. A positive
limit smaller than the dataset shuffles with ``random.Random(seed)`` and
keeps the prefix. The same seed and the same input order produce the same
ids. The seed is recorded even when the run uses every row.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def take_sample(rows: Sequence[T], limit: int | None, seed: int) -> list[T]:
    items = list(rows)
    if limit is None or limit <= 0 or limit >= len(items):
        return items
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:limit]

"""Paired comparison of two runs (design §8.3): the gain per dimension and the probability that it is real.

Pages are resampled by source document (the split's group key), since pages of one document move together.
"""

from __future__ import annotations

import random
from collections import defaultdict

from evalkit.ledger import CaseRow
from evalkit.losses import slot_weights
from evalkit.splits import fold_of, group_key


def paired_bootstrap(
    base: list[CaseRow], new: list[CaseRow], dimension: str, fold: str = "dev", samples: int = 2000, seed: int = 7
) -> dict[str, float]:
    """Gain of `new` over `base` on one dimension (0-100 points) and P(gain > 0) under a document bootstrap."""
    weights = slot_weights()
    a = {(c.example_id, c.metric): c.score or 0.0 for c in base if c.counted}
    b = {(c.example_id, c.metric): c.score or 0.0 for c in new if c.counted}
    keys = [k for k in a if k in b and (fold == "all" or fold_of(k[0]) == fold)]
    slots: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for example_id, metric in keys:
        subset, _, stem = example_id.partition("/")
        if weights.get((subset, metric), ("", 0))[0] != dimension:
            continue
        slots[(subset, metric)][group_key(subset, stem)].append(b[(example_id, metric)] - a[(example_id, metric)])
    if not slots:
        raise ValueError(f"no shared pages for dimension {dimension!r}")

    def gain(pick=None) -> float:
        total = 0.0
        for (subset, metric), groups in slots.items():
            names = list(groups)
            chosen = pick(names) if pick else names
            deltas = [d for g in chosen for d in groups[g]]
            total += weights[(subset, metric)][1] * (sum(deltas) / len(deltas) if deltas else 0.0)
        return 100.0 * total

    rng = random.Random(seed)
    draws = [gain(lambda names: rng.choices(names, k=len(names))) for _ in range(samples)]
    draws.sort()
    return {
        "gain": gain(),
        "p_positive": sum(d > 0 for d in draws) / samples,
        "low": draws[int(0.05 * samples)],
        "high": draws[int(0.95 * samples)],
        "pages": sum(len(v) for g in slots.values() for v in g.values()),
    }

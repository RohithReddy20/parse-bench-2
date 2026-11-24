"""The pilot sample: a few development-part pages from every subset, fixed by a seed."""

from __future__ import annotations

import random

from evalkit.explode import SUBSETS
from evalkit.splits import fold_of

SMALL = ("headings", "fillable_forms", "underline")  # the three smallest subsets get two pages each


def pilot_ids(example_ids: list[str], per_subset: int = 4, seed: int = 20261124) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    for subset in SUBSETS:
        pool = sorted(i for i in example_ids if i.startswith(subset + "/") and fold_of(i) == "dev")
        rng.shuffle(pool)
        out += pool[: 2 if subset in SMALL else per_subset]
    return sorted(out)

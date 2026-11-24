"""Development part and check slice (design §8.2).

80% of source documents are the development part; 20% are a check slice that is scored on every run and whose
pages are never inspected. A case's fold is a seeded hash of its group key, so cases added by later dataset
releases get a fold without moving existing ones.

Group key (`group_key`), from the file stem, shared across subsets so a document never straddles the folds:
  forms            `filled__<template>--p3__typed`, `blank__<template>--p2__blank`  -> the template
  fillable_forms   `<form>__p1`                                                      -> the form
  layout, text     `<category>__<document>[_p12]`                                    -> the document
  charts, insurance_filing_tables, financial_tables, tables
                   `<document>_p12`, `<document>_page48`                             -> the document
  tables           `<company>.<year>.page_82.pdf_110673_page1`                       -> company.year
  tables           `SERFF_*_random_pages 1_page402` (random pages of many filings)   -> the page itself
  headings, strikeout, underline, complex_table_headers, dense_tables
                   `case-0030-<hash>`, `dense_table_418`                              -> the case itself

The seed is the first of `docparse-v2-split-<n>` for which every subset's check share lies in [0.15, 0.25]
(`choose_seed`); that is what "stratified by subset" means for a hash split. It is fixed here once.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

Fold = Literal["dev", "check"]
SEED = "docparse-v2-split-35"
SPLIT_FILE = Path(__file__).parent / "splits_data" / "v1.json"

_PAGE_ONLY = re.compile(r"p\d+")
_FINTAB = re.compile(r"\.page_\d+\.pdf_\d+$")
_PAGE_SUFFIX = re.compile(r"(?:--p\d+|_p\d+|_page\d+)$")


def group_key(subset: str, stem: str) -> str:
    parts = stem.split("__")
    if len(parts) >= 2 and _PAGE_ONLY.fullmatch(parts[-1]):  # fillable forms: <form>__p1
        doc = "__".join(parts[:-1])
    elif len(parts) >= 3:  # forms: <state>__<template>--p3__<kind>
        doc = "__".join(parts[1:-1])
    elif len(parts) == 2:  # <category>__<document>
        doc = parts[1]
    else:
        doc = stem
    if "random_pages" not in doc:
        doc = _PAGE_SUFFIX.sub("", doc)
        doc = _FINTAB.sub("", doc)
    return doc.strip().lower()


def fold(key: str, seed: str = SEED) -> Fold:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return "check" if int(digest, 16) % 5 == 0 else "dev"


def build_split(example_ids: list[str], seed: str = SEED) -> dict[str, Fold]:
    out: dict[str, Fold] = {}
    for example_id in sorted(example_ids):
        subset, _, stem = example_id.partition("/")
        out[example_id] = fold(group_key(subset, stem), seed)
    return out


def check_shares(split: dict[str, Fold]) -> dict[str, float]:
    total: dict[str, int] = defaultdict(int)
    check: dict[str, int] = defaultdict(int)
    for example_id, f in split.items():
        subset = example_id.partition("/")[0]
        total[subset] += 1
        check[subset] += f == "check"
    return {s: check[s] / total[s] for s in sorted(total)}


def choose_seed(example_ids: list[str], low: float = 0.15, high: float = 0.25, tries: int = 2000) -> str:
    for n in range(1, tries + 1):
        seed = f"docparse-v2-split-{n}"
        if all(low <= share <= high for share in check_shares(build_split(example_ids, seed)).values()):
            return seed
    raise ValueError("no seed gives every subset a check share in range")


def write_split(example_ids: list[str], path: Path = SPLIT_FILE) -> dict[str, Fold]:
    split = build_split(example_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": SEED, "folds": split}, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    return split


_loaded: dict[str, Fold] | None = None


def load_split(path: Path = SPLIT_FILE) -> dict[str, Fold]:
    global _loaded
    if _loaded is None or path != SPLIT_FILE:
        data = json.loads(path.read_text(encoding="utf-8"))["folds"]
        if path != SPLIT_FILE:
            return data
        _loaded = data
    return _loaded


def fold_of(example_id: str) -> Fold:
    """The recorded fold, or the hash rule for a case a later dataset release added."""
    split = load_split() if SPLIT_FILE.exists() else {}
    if example_id in split:
        return split[example_id]
    subset, _, stem = example_id.partition("/")
    return fold(group_key(subset, stem))

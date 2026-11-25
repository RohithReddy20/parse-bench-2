"""Word tokens of the reading and of the PDF text layer, and an in-order alignment between them.

A token is a run of letters or digits, compared case- and accent-insensitively. The alignment is difflib's longest
matching blocks over the two token sequences: it follows reading order, so a token is matched only in the context
of its neighbours, and a word the reading lacks (or adds) leaves its neighbours aligned.

difflib slows down badly on long sequences full of repeated tokens (a dense table of digits), so a large pair is
first cut at anchors, tokens that occur exactly once on each side, kept in an increasing chain, and difflib runs
between consecutive anchors; a stretch still too large is cut the same way again, and left unaligned when it has
no anchors.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from collections import Counter
from difflib import SequenceMatcher

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
MAX_CELLS = 4_000_000  # a pair of sequences up to this many token pairs is aligned by difflib directly
MAX_DEPTH = 6


def key(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).lower()
    return "".join(c for c in folded if not unicodedata.combining(c))


def tokens(text: str) -> list[tuple[int, int, str]]:
    """(start, end, key) of every token of `text`."""
    return [(m.start(), m.end(), key(m.group())) for m in TOKEN.finditer(text)]


def _chain(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The longest chain of pairs increasing on both sides (pairs come sorted by their first index)."""
    tails: list[int] = []  # tails[k]: the second index ending the best chain of length k + 1
    ends: list[int] = []  # index into pairs of that chain's last pair
    back = [-1] * len(pairs)
    for p, (_, j) in enumerate(pairs):
        k = bisect_left(tails, j)
        if k == len(tails):
            tails.append(j)
            ends.append(p)
        else:
            tails[k], ends[k] = j, p
        back[p] = ends[k - 1] if k else -1
    out, p = [], ends[-1] if ends else -1
    while p >= 0:
        out.append(pairs[p])
        p = back[p]
    return out[::-1]


def _match(a: list[str], b: list[str], alo: int, ahi: int, blo: int, bhi: int, out: list[int], depth: int) -> None:
    if ahi <= alo or bhi <= blo:
        return
    if (ahi - alo) * (bhi - blo) <= MAX_CELLS:
        for i, j, n in SequenceMatcher(None, a[alo:ahi], b[blo:bhi], autojunk=False).get_matching_blocks():
            for x in range(n):
                out[alo + i + x] = blo + j + x
        return
    if depth >= MAX_DEPTH:
        return
    count_a, count_b = Counter(a[alo:ahi]), Counter(b[blo:bhi])
    where_b = {b[j]: j for j in range(blo, bhi) if count_b[b[j]] == 1}
    pairs = [(i, where_b[a[i]]) for i in range(alo, ahi) if count_a[a[i]] == 1 and a[i] in where_b]
    anchors = _chain(pairs)
    if not anchors:
        return
    prev_i, prev_j = alo - 1, blo - 1
    for i, j in [*anchors, (ahi, bhi)]:
        _match(a, b, prev_i + 1, i, prev_j + 1, j, out, depth + 1)
        if i < ahi:
            out[i] = j
        prev_i, prev_j = i, j


def match(a: list[str], b: list[str]) -> list[int]:
    """For each token of `a`, the index of its partner in `b`, or -1."""
    out = [-1] * len(a)
    _match(a, b, 0, len(a), 0, len(b), out, 0)
    return out


def opcodes(a: list[str], b: list[str]) -> list[tuple[str, int, int, int, int]]:
    """difflib-style opcodes (equal / replace / delete / insert) of the alignment `match` makes."""
    partner = match(a, b)
    blocks: list[list[int]] = []  # [i, j, n] runs of consecutive partners
    for i, j in enumerate(partner):
        if j < 0:
            continue
        if blocks and blocks[-1][0] + blocks[-1][2] == i and blocks[-1][1] + blocks[-1][2] == j:
            blocks[-1][2] += 1
        else:
            blocks.append([i, j, 1])
    out, i0, j0 = [], 0, 0
    for i, j, n in [*blocks, [len(a), len(b), 0]]:
        if i0 < i and j0 < j:
            out.append(("replace", i0, i, j0, j))
        elif i0 < i:
            out.append(("delete", i0, i, j0, j))
        elif j0 < j:
            out.append(("insert", i0, i, j0, j))
        if n:
            out.append(("equal", i, i + n, j, j + n))
        i0, j0 = i + n, j + n
    return out

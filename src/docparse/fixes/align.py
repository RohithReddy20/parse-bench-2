"""Word tokens of the reading and of the PDF text layer, and an in-order alignment between them.

A token is a run of letters or digits, compared case- and accent-insensitively. The alignment is difflib's longest
matching blocks over the two token sequences: it follows reading order, so a token is matched only in the context
of its neighbours, and a word the reading lacks (or adds) leaves its neighbours aligned.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def key(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).lower()
    return "".join(c for c in folded if not unicodedata.combining(c))


def tokens(text: str) -> list[tuple[int, int, str]]:
    """(start, end, key) of every token of `text`."""
    return [(m.start(), m.end(), key(m.group())) for m in TOKEN.finditer(text)]


def match(a: list[str], b: list[str]) -> list[int]:
    """For each token of `a`, the index of its partner in `b`, or -1."""
    out = [-1] * len(a)
    for i, j, n in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for x in range(n):
            out[i + x] = j + x
    return out

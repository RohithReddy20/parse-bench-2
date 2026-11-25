"""When the page read is degenerate (a loop, an empty grid, a page cut short) and the PDF has a text layer,
the text layer is the better reading. `reading_recall` measures the read against the text layer;
`looped` spots a read that repeats itself; `text_dump` writes the text layer as plain lines.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter

from docparse.fixes.layout_refine import PtWord, lines_of

MIN_TOKENS = 50  # shorter text layers say too little about the reading
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_TAG = re.compile(r"<[^>]+>")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())


def reading_recall(markdown: str, words: list[PtWord]) -> float:
    """Share of the text layer's word tokens that the reading contains (each as often as the page shows it)."""
    page = Counter(t for w in words for t in _tokens(w.text))
    total = sum(page.values())
    if not total:
        return 1.0
    read = Counter(_tokens(_TAG.sub(" ", markdown)))
    return sum(min(n, read[t]) for t, n in page.items()) / total


def looped(markdown: str, ratio: float = 30.0, min_chars: int = 3000) -> bool:
    """A read that repeats itself compresses far better than any real page (real pages: 3x to 15x)."""
    data = markdown.encode("utf-8")
    return len(data) >= min_chars and len(data) / len(zlib.compress(data)) >= ratio


def text_dump(words: list[PtWord]) -> str:
    """The text layer as lines, top to bottom; a vertical gap over one line height starts a new paragraph."""
    out: list[str] = []
    previous = None
    for line in lines_of(words):
        top, bottom = min(w.y0 for w in line), max(w.y1 for w in line)
        if previous is not None and top - previous > bottom - top:
            out.append("")
        out.append(" ".join(w.text for w in line))
        previous = bottom
    return "\n".join(out)

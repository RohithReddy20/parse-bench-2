"""Several independent reads of one chart, combined cell by cell.

A value read off an axis carries reading noise: two reads of the same bar differ by a little, and now and then one
read is far off. When the reads agree on the table's shape and labels, the number kept for each cell is the median
of the reads (always one of the values actually read, never an average), so a single misread is outvoted. Reads
that disagree on structure are left out; with fewer than three aligned reads the first read stands as it is.
"""

from __future__ import annotations

import html
import re

_CELL = re.compile(r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]\s*>)", re.DOTALL | re.IGNORECASE)
_TABLE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_NUMBER = re.compile(r"^[\s$€£¥(+\-−–]*\d[\d,.\s]*\)?\s*(?:%|[a-zA-Z]{0,4})?\s*$")
_DIGITS = re.compile(r"\d[\d,]*\.?\d*")


def _text(cell: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", cell)).split())


def _value(cell: str) -> float | None:
    """The number a cell holds, or None for a label or an empty cell."""
    text = _text(cell)
    if not text or not _NUMBER.match(text):
        return None
    m = _DIGITS.search(text)
    if not m:
        return None
    try:
        value = float(m.group().replace(",", ""))
    except ValueError:
        return None
    negative = text.lstrip(" $€£¥").startswith(("-", "−", "–", "("))
    return -value if negative else value


def _cells(answer: str) -> list[tuple[int, int, str]]:
    """(start, end, inner HTML) of every table cell of the answer, in order."""
    out = []
    for table in _TABLE.finditer(answer):
        for m in _CELL.finditer(table.group()):
            out.append((table.start() + m.start(2), table.start() + m.end(2), m.group(2)))
    return out


def _aligned(base: list[tuple[int, int, str]], other: list[tuple[int, int, str]]) -> bool:
    """Same number of cells, and the same label wherever the base has a label."""
    if len(base) != len(other) or not base:
        return False
    labels = same = 0
    for (_, _, a), (_, _, b) in zip(base, other, strict=True):
        if _value(a) is None:
            if _text(a):
                labels += 1
                same += _text(a).casefold() == _text(b).casefold()
        elif _value(b) is None and _text(b):
            return False  # a number in one read, a label in the other
    return labels == 0 or same >= 0.9 * labels


def fuse_reads(answers: list[str]) -> str:
    """The first read with each number replaced by the median of the reads aligned with it (three or more)."""
    answers = [a for a in answers if a and _TABLE.search(a)]
    if len(answers) < 3:
        return answers[0] if answers else ""
    base = answers[0]
    base_cells = _cells(base)
    peers = [c for c in (_cells(a) for a in answers[1:]) if _aligned(base_cells, c)]
    if len(peers) < 2:
        return base
    out, pos = [], 0
    for i, (start, end, inner) in enumerate(base_cells):
        value = _value(inner)
        if value is None:
            continue
        reads = [(value, inner)]
        for peer in peers:
            v = _value(peer[i][2])
            if v is not None:
                reads.append((v, peer[i][2]))
        if len(reads) < 3:
            continue
        reads.sort(key=lambda r: r[0])
        median = (
            reads[(len(reads) - 1) // 2]
            if len(reads) % 2
            else min(reads[len(reads) // 2 - 1 : len(reads) // 2 + 1], key=lambda r: abs(r[0] - value))
        )
        if median[0] != value:
            out.append(base[pos:start])
            out.append(median[1])
            pos = end
    out.append(base[pos:])
    return "".join(out)

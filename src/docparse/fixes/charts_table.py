"""OCRia-1's HTML chart tables: cells with their exact place in the markdown, the numbers they print, and
in-place rewriting of single cells (everything else stays byte for byte)."""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass

_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr\s*>", re.IGNORECASE | re.DOTALL)
_CELL = re.compile(r"<(t[hd])\b([^>]*)>(.*?)</t[hd]\s*>", re.IGNORECASE | re.DOTALL)
_SPAN = re.compile(r"""\b(rowspan|colspan)\s*=\s*["']?(\d+)""", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# one number in a cell, with what may surround it: a sign, a currency, a percent or a short unit
_NUMBER = re.compile(r"(?<![\w.])([-−–+]?)([$€£¥]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w.])")
_AROUND = re.compile(r"[\s%$€£¥()*~≈]*|[\s%]*[a-zA-Z]{1,3}\.?")


@dataclass
class Cell:
    row: int
    col: int
    rowspan: int
    colspan: int
    th: bool
    text: str  # tags stripped, entities decoded, whitespace collapsed
    start: int  # offsets of the inner HTML in the table's raw markdown
    end: int


@dataclass
class Number:
    value: float
    decimals: int
    span: tuple[int, int]  # offsets of the digits (sign included) in the cell's inner HTML
    comma: bool  # thousands separators
    minus: str  # the sign character printed ("-", "−", "–" or "")
    currency: str = ""


def cell_text(inner: str) -> str:
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", inner))).strip()


def parse_cells(raw: str) -> list[Cell] | None:
    """Cells of a single (not nested) HTML table, placed on the grid that rowspan / colspan give."""
    if len(re.findall(r"<table\b", raw, re.IGNORECASE)) != 1:
        return None
    cells: list[Cell] = []
    taken: set[tuple[int, int]] = set()
    for r, row in enumerate(_ROW.finditer(raw)):
        c = 0
        for m in _CELL.finditer(raw, row.start(1), row.end(1)):
            while (r, c) in taken:
                c += 1
            spans = {k.lower(): int(v) for k, v in _SPAN.findall(m.group(2))}
            rs, cs = max(1, min(spans.get("rowspan", 1), 500)), max(1, min(spans.get("colspan", 1), 500))
            cells.append(Cell(r, c, rs, cs, m.group(1).lower() == "th", cell_text(m.group(3)), m.start(3), m.end(3)))
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, c + dc))
            c += cs
    return cells or None


def number_of(raw: str, cell: Cell) -> Number | None:
    """The single number a cell prints, when the rest of the cell is only sign, currency, percent or a short
    unit ("12.5%", "$4.3", "−0.25", "1,234"); None for text, ranges, blanks and several numbers."""
    inner = raw[cell.start : cell.end]
    if "<" in inner or "&" in inner:
        return None  # markup or entities inside the cell: leave it alone
    found = list(_NUMBER.finditer(inner))
    if len(found) != 1:
        return None
    m = found[0]
    before, after = inner[: m.start()], inner[m.end() :]
    if not _AROUND.fullmatch(before) or not _AROUND.fullmatch(after):
        return None
    if "(" in before or ")" in after:
        return None  # an accounting negative or a note: not rewritten
    sign, currency, digits, frac = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    value = float(digits.replace(",", "") + frac)
    return Number(
        value=-value if sign in ("-", "−", "–") else value,
        decimals=len(frac) - 1 if frac else 0,
        span=(cell.start + m.start(), cell.start + m.end()),
        comma="," in digits,
        minus=sign if sign in ("-", "−", "–") else "",
        currency=currency,
    )


def format_number(value: float, decimals: int, like: Number) -> str:
    """The value printed the way the cell printed its own number (sign character, currency, separators)."""
    decimals = max(0, min(6, decimals))
    v = round(value, decimals)
    if math.isclose(v, 0.0, abs_tol=10**-decimals / 2):
        v = 0.0
    body = f"{abs(v):,.{decimals}f}" if like.comma else f"{abs(v):.{decimals}f}"
    return ((like.minus or "-") if v < 0 else "") + like.currency + body


def replace_spans(raw: str, edits: list[tuple[tuple[int, int], str]]) -> str:
    """Apply (span, text) replacements to the raw table, last first."""
    out = raw
    for (a, b), text in sorted(edits, key=lambda e: e[0][0], reverse=True):
        out = out[:a] + text + out[b:]
    return out

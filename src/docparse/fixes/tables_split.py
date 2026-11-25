"""Two values the reading wrote in one cell, where the page prints them as two columns (option `split`).

OCRia-1 sometimes writes a column pair as one column: "PLN 15,458m +9.9% YoY" under "GMV", where the page
prints the amount and the change in two columns with a wide gap between them. The page decides it: in every
body cell of the column, the matched words fall in two groups on either side of one vertical gutter that no word
of the column crosses, both groups hold a number, and the gutter is wider than a word space. The column is then
split in two; the cells above it (headers, section rows) span both halves.

A currency code or unit next to an amount ("EUR  20,000,000") is not split: one side holds no number.
"""

from __future__ import annotations

import html
import re

from docparse.fixes.tables_align import Alignment, TWord, fold
from docparse.fixes.tables_html import HCell, HTable

MIN_GUTTER = 8.0  # points: narrower gaps are word spaces
MIN_CELLS = 2  # body cells that must show the gutter
_DIGIT = re.compile(r"\d")
_SERIAL = re.compile(r"\(?\d{1,3}[.)]")  # a row number in front of a label: "(1)", "2.", "3)"
_COLSPAN = re.compile(r"colspan\s*=\s*[\"']?\d+[\"']?", re.I)
_COLSPAN_ATTR = re.compile(r"\s*colspan\s*=\s*[\"']?\d+[\"']?", re.I)  # with the space before it


def _halves(ws: list[TWord]) -> tuple[float, float, int] | None:
    """(left edge of the widest gap, right edge, words left of it) for a cell's words, left to right."""
    ws = sorted(ws, key=lambda w: w.x0)
    if len(ws) < 2:
        return None
    gaps = [(b.x0 - a.x1, i + 1) for i, (a, b) in enumerate(zip(ws, ws[1:], strict=False))]
    width, k = max(gaps)
    return ws[k - 1].x1, ws[k].x0, k


def _split_text(cell: HCell, left_key: str) -> tuple[str, str] | None:
    """The cell's text cut where its folded prefix equals `left_key` (at a space)."""
    text = cell.text
    for i, ch in enumerate(text):
        if ch == " " and fold(text[:i]) == left_key:
            return text[:i].strip(), text[i:].strip()
    return None


def _gutter_column(table: HTable, al: Alignment, c: int) -> dict[int, tuple[str, str]] | None:
    """For grid column c: id(cell) -> (left text, right text) for every body cell to split, or None when the
    column has no gutter that every non-empty body cell shows."""
    lo, hi = float("-inf"), float("inf")
    parts: dict[int, tuple[str, str]] = {}
    for row in table.rows:
        for cell in row.cells:
            if cell.col != c or cell.colspan != 1 or cell.tag == "th" or not cell.text:
                continue
            ws = al.words.get(id(cell))
            if not ws:
                return None
            h = _halves(ws)
            if h is None:
                return None
            x1, x0, k = h
            ws = sorted(ws, key=lambda w: w.x0)
            left, right = ws[:k], ws[k:]
            ltext = " ".join(w.text for w in left)
            if not (_DIGIT.search(ltext) and _DIGIT.search("".join(w.text for w in right))) or _SERIAL.fullmatch(ltext):
                return None
            lo, hi = max(lo, x1), min(hi, x0)
            if hi - lo < MIN_GUTTER:
                return None
            cut = _split_text(cell, "".join(w.key for w in left))
            if cut is None:
                return None
            parts[id(cell)] = cut
    return parts if len(parts) >= MIN_CELLS else None


def _with_colspan(cell: HCell, span: int) -> HCell:
    attrs = _COLSPAN.sub(f'colspan="{span}"', cell.attrs) if _COLSPAN.search(cell.attrs) else (
        f'{cell.attrs} colspan="{span}"')  # fmt: skip
    return HCell(tag=cell.tag, attrs=attrs, inner=cell.inner, raw="", rowspan=cell.rowspan, colspan=span)


def split_columns(table: HTable, al: Alignment) -> bool:
    """Split every column whose body cells show a gutter between two numbers. True when any column split."""
    changed = False
    c = 0
    while c < table.width():
        parts = _gutter_column(table, al, c)
        if parts is None:
            c += 1
            continue
        for row in table.rows:
            out: list[HCell] = []
            for cell in row.cells:
                if not (cell.col <= c < cell.col + cell.colspan):
                    out.append(cell)
                elif id(cell) in parts:
                    left, right = parts[id(cell)]
                    attrs = _COLSPAN_ATTR.sub("", cell.attrs)
                    for text in (left, right):
                        out.append(HCell(tag=cell.tag, attrs=attrs, inner=html.escape(text, quote=False), raw=""))
                elif cell.colspan == 1 and cell.tag != "th" and not cell.text and cell.rowspan == 1:
                    out += [cell, HCell.empty(cell.tag)]
                else:
                    out.append(_with_colspan(cell, cell.colspan + 1))
            if out != row.cells:
                row.cells, row.edited = out, True
        table.place()
        changed = True
        c += 2
    return changed

"""A lossless model of one HTML table as OCRia-1 writes it: rows of cells, each keeping its exact source text.

A table nobody edits renders back byte for byte. Edits are new cells (empty ones inserted to move the cells
after them) and new inner text; a row or cell that was not edited keeps its original markup.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass, field

_ROW = re.compile(r"<tr\b[^>]*>.*?</tr\s*>", re.S | re.I)
_CELL = re.compile(r"<(t[hd])\b([^>]*)>(.*?)</t[hd]\s*>", re.S | re.I)
_SPAN = re.compile(r"(rowspan|colspan)\s*=\s*[\"']?(\d+)", re.I)
_TAG = re.compile(r"<[^>]+>")
_BR = re.compile(r"<br\s*/?>", re.I)
_TABLE_TAG = re.compile(r"<table\b", re.I)


@dataclass(eq=False)
class HCell:
    tag: str  # "td" or "th"
    attrs: str
    inner: str
    raw: str  # the cell's exact source, "" for a cell this module created
    rowspan: int = 1
    colspan: int = 1
    row: int = 0  # grid position, set by `place`
    col: int = 0

    @property
    def text(self) -> str:
        """The cell's visible text: tags dropped, line breaks as spaces, entities decoded, spaces collapsed."""
        return " ".join(html.unescape(_TAG.sub("", _BR.sub(" ", self.inner))).split())

    @property
    def lines(self) -> list[str]:
        """The visible text of each line the cell writes (split at <br>), blank lines dropped."""
        parts = (" ".join(html.unescape(_TAG.sub("", p)).split()) for p in _BR.split(self.inner))
        return [p for p in parts if p]

    @property
    def plain_empty(self) -> bool:
        return not self.text and self.rowspan == 1 and self.colspan == 1

    def render(self) -> str:
        return self.raw or f"<{self.tag}{self.attrs}>{self.inner}</{self.tag}>"

    @classmethod
    def empty(cls, tag: str) -> HCell:
        return cls(tag=tag, attrs="", inner="", raw="")


@dataclass(eq=False)
class HRow:
    head: str  # "<tr ...>"
    cells: list[HCell]
    seps: list[str]  # source text around the cells: seps[0] before the first cell, seps[-1] after the last
    tail: str  # "</tr>"
    edited: bool = False

    def render(self) -> str:
        if not self.edited:
            return (
                self.head
                + self.seps[0]
                + "".join(c.render() + s for c, s in zip(self.cells, self.seps[1:], strict=True))
                + self.tail
            )
        return self.head + "".join(c.render() for c in self.cells) + self.tail


@dataclass(eq=False)
class HTable:
    raw: str
    rows: list[HRow]
    seps: list[str]  # source text around the rows: seps[0] holds "<table ...>", seps[-1] holds "</table>"
    changed: bool = False
    _width: dict[int, int] = field(default_factory=dict)

    def render(self) -> str:
        if not self.changed:
            return self.raw
        return self.seps[0] + "".join(r.render() + s for r, s in zip(self.rows, self.seps[1:], strict=True))

    def place(self) -> None:
        """Grid positions of every cell, as a browser lays the table out (rowspans reserve the slots below)."""
        taken: set[tuple[int, int]] = set()
        for r, row in enumerate(self.rows):
            c = 0
            for cell in row.cells:
                while (r, c) in taken:
                    c += 1
                cell.row, cell.col = r, c
                for i in range(r, r + cell.rowspan):
                    for j in range(c, c + cell.colspan):
                        taken.add((i, j))
                c += cell.colspan
        self._width = {}
        for row in self.rows:
            for cell in row.cells:
                for i in range(cell.row, min(cell.row + cell.rowspan, len(self.rows))):
                    self._width[i] = self._width.get(i, 0) + cell.colspan

    def row_width(self, r: int) -> int:
        """Grid columns row r covers, with its own cells and the rowspans reaching into it from above."""
        return self._width.get(r, 0)

    def width(self) -> int:
        """The most common row width (ties to the wider): the table's real width, which one over-long row
        written with a stray cell does not change."""
        counts = Counter(self.row_width(r) for r in range(len(self.rows)))
        return max(counts, key=lambda k: (counts[k], k)) if counts else 0

    def spanned_into(self, r: int) -> set[int]:
        """Columns of row r held by cells spanning down from rows above."""
        cols: set[int] = set()
        for row in self.rows[:r]:
            for cell in row.cells:
                if cell.row < r < cell.row + cell.rowspan:
                    cols.update(range(cell.col, cell.col + cell.colspan))
        return cols


def _parse_row(raw: str) -> HRow:
    head = re.match(r"<tr\b[^>]*>", raw, re.I).group(0)
    end = re.search(r"</tr\s*>\Z", raw, re.I)
    body = raw[len(head) : end.start()]
    cells, seps, pos = [], [], 0
    for m in _CELL.finditer(body):
        seps.append(body[pos : m.start()])
        spans = {k.lower(): max(1, int(v)) for k, v in _SPAN.findall(m.group(2))}
        cells.append(
            HCell(
                tag=m.group(1).lower(),
                attrs=m.group(2),
                inner=m.group(3),
                raw=m.group(0),
                rowspan=spans.get("rowspan", 1),
                colspan=spans.get("colspan", 1),
            )
        )
        pos = m.end()
    seps.append(body[pos:])
    return HRow(head=head, cells=cells, seps=seps, tail=end.group(0))


def parse_table(raw: str) -> HTable | None:
    """The table in `raw` (one complete `<table>...</table>`), or None for a nested or row-less table."""
    if len(_TABLE_TAG.findall(raw)) != 1:
        return None
    rows, seps, pos = [], [], 0
    for m in _ROW.finditer(raw):
        seps.append(raw[pos : m.start()])
        rows.append(_parse_row(m.group(0)))
        pos = m.end()
    seps.append(raw[pos:])
    if not rows:
        return None
    table = HTable(raw=raw, rows=rows, seps=seps)
    table.place()
    return table

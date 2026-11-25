"""Where each cell of a table sits on the page: table rows matched to text-layer lines, then cells to words.

    1. Text-layer words are grouped into lines (top to bottom, each left to right).
    2. Rows go to lines by a monotone assignment (row order = line order) that maximises the number of cells
       whose text occurs on the line as consecutive words. Repeated values (a "0" in every row) cannot pull a
       row onto the wrong line, because the whole row votes and the order is kept.
    3. On its line, a row's cells are matched to runs of consecutive words, in order, by a small dynamic
       program; only exact matches count (case, spacing and dash style aside).

Words used by one table are not offered to the next table of the same page.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from docparse.fixes.tables_html import HCell, HTable
from docparse.pdf.render import RenderedPage

MAX_RUN = 8  # most words one cell may cover on a line
_DASHES = re.compile("[‐-―−﹘﹣－]")
_SPACE = re.compile(r"\s+")


def fold(text: str) -> str:
    """Comparison key: NFKC, one dash, no spaces, lower case."""
    text = _DASHES.sub("-", unicodedata.normalize("NFKC", text)).replace("​", "")
    return _SPACE.sub("", text).lower()


@dataclass(frozen=True)
class TWord:
    index: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    key: str

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def page_words(rendered: RenderedPage) -> list[TWord]:
    """Text-layer words in PDF points, split at the spaces the PDF holds."""
    s = rendered.scale
    out = []
    for i, w in enumerate(rendered.text_words or []):
        key = fold(w.text)
        if key:
            out.append(TWord(i, w.text, w.box[0] * s, w.box[1] * s, w.box[2] * s, w.box[3] * s, key))
    return out


def lines_of(words: list[TWord]) -> list[list[TWord]]:
    """Words grouped into lines, top to bottom, each line left to right."""
    lines: list[list[TWord]] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        if lines:
            line = lines[-1]
            top, bottom = min(x.y0 for x in line), max(x.y1 for x in line)
            if top <= w.cy <= bottom or abs(w.cy - (top + bottom) / 2) < 0.4 * (w.y1 - w.y0):
                line.append(w)
                continue
        lines.append([w])
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def _runs(line: list[TWord], used: set[int]) -> list[list[str | None]]:
    """runs[i][j]: the keys of line[i:j] joined (None when a word in it is used or the run is too long)."""
    m = len(line)
    runs: list[list[str | None]] = [[None] * (m + 1) for _ in range(m + 1)]
    for i in range(m):
        key = ""
        for j in range(i, min(m, i + MAX_RUN)):
            if line[j].index in used:
                break
            key += line[j].key
            runs[i][j + 1] = key
    return runs


def match_row(cells: list[tuple[HCell, str]], line: list[TWord], used: set[int]) -> list[tuple[HCell, int, int]]:
    """The most cells matched, in order, each to a run line[i:j] whose joined keys equal the cell's key."""
    n, m = len(cells), len(line)
    runs = _runs(line, used)
    best = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for a in range(n + 1):
        for b in range(m + 1):
            if a == 0 and b == 0:
                continue
            value, step = -1, None
            if a > 0 and best[a - 1][b] > value:
                value, step = best[a - 1][b], ("cell",)
            if b > 0 and best[a][b - 1] > value:
                value, step = best[a][b - 1], ("word",)
            if a > 0 and b > 0:
                key = cells[a - 1][1]
                for i in range(max(0, b - MAX_RUN), b):
                    if runs[i][b] == key and best[a - 1][i] + 1 > value:
                        value, step = best[a - 1][i] + 1, ("take", i)
            best[a][b], back[a][b] = value, step
    out = []
    a, b = n, m
    while a > 0 or b > 0:
        step = back[a][b]
        if step[0] == "cell":
            a -= 1
        elif step[0] == "word":
            b -= 1
        else:
            out.append((cells[a - 1][0], step[1], b))
            a, b = a - 1, step[1]
    return out[::-1]


def _row_cells(table: HTable) -> dict[int, list[tuple[HCell, str]]]:
    """Grid row -> the cells starting in it that hold text, with their keys, left to right."""
    rows: dict[int, list[tuple[HCell, str]]] = {}
    for row in table.rows:
        for cell in row.cells:
            key = fold(cell.text)
            if key:
                rows.setdefault(cell.row, []).append((cell, key))
    return rows


def _line_keys(line: list[TWord], used: set[int]) -> set[str]:
    return {k for r in _runs(line, used) for k in r if k}


def assign_rows(rows: dict[int, list[tuple[HCell, str]]], lines: list[list[TWord]], used: set[int]) -> dict[int, int]:
    """Row -> line, monotone, maximising the cells found on their row's line."""
    order = sorted(rows)
    index: dict[str, list[int]] = {}
    for li, line in enumerate(lines):
        for key in _line_keys(line, used):
            index.setdefault(key, []).append(li)
    score: dict[tuple[int, int], int] = {}
    for r in order:
        for _, key in rows[r]:
            for li in index.get(key, ()):
                score[(r, li)] = score.get((r, li), 0) + 1
    n, m = len(order), len(lines)
    best = [[0] * (m + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, m + 1):
            best[a][b] = max(best[a - 1][b], best[a][b - 1], best[a - 1][b - 1] + score.get((order[a - 1], b - 1), 0))
    out: dict[int, int] = {}
    a, b = n, m
    while a > 0 and b > 0:
        s = score.get((order[a - 1], b - 1), 0)
        if s and best[a][b] == best[a - 1][b - 1] + s:
            out[order[a - 1]] = b - 1
            a, b = a - 1, b - 1
        elif best[a][b] == best[a - 1][b]:
            a -= 1
        else:
            b -= 1
    return out


@dataclass
class Alignment:
    words: dict[int, list[TWord]] = field(default_factory=dict)  # id(cell) -> the words it matched
    row_line: dict[int, int] = field(default_factory=dict)  # grid row -> line index
    cells: int = 0  # cells holding text

    @property
    def found(self) -> float:
        """Share of the table's text cells found on the page."""
        return len(self.words) / self.cells if self.cells else 0.0

    def span(self, cell: HCell) -> tuple[float, float] | None:
        ws = self.words.get(id(cell))
        return (min(w.x0 for w in ws), max(w.x1 for w in ws)) if ws else None


def align_table(table: HTable, lines: list[list[TWord]], used: set[int]) -> Alignment:
    rows = _row_cells(table)
    out = Alignment(cells=sum(len(v) for v in rows.values()))
    out.row_line = assign_rows(rows, lines, used)
    for r, li in out.row_line.items():
        line = lines[li]
        for cell, i, j in match_row(rows[r], line, used):
            out.words[id(cell)] = line[i:j]
    return out


def find_on_lines(key: str, lines: list[list[TWord]], lo: int, hi: int, used: set[int]) -> list[list[TWord]]:
    """Every run of consecutive unused words on lines lo..hi-1 whose joined keys equal `key`."""
    found = []
    for line in lines[max(0, lo) : max(0, hi)]:
        for i in range(len(line)):
            joined = ""
            for j in range(i, min(len(line), i + MAX_RUN)):
                if line[j].index in used:
                    break
                joined += line[j].key
                if joined == key:
                    found.append(line[i : j + 1])
                    break
                if not key.startswith(joined):
                    break
    return found

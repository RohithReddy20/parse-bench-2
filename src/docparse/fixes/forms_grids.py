"""A table of check boxes under option captions (a rating grid) written as one line per row (form option `grids`).

    | How would you rate it? | Good | Fair | Poor |          How would you rate it?
    | 1. Walk                | [ ]  | [x]  | [ ]  |   ->     1. Walk: [ ] Good [x] Fair [ ] Poor

A box is read with the option label beside it; in the grid the label is the column caption, rows away. Written
as a line, each row is a question with its options, every box next to its own label. The grid's header text is
kept as one line above. A table qualifies when its body rows each hold a caption in the first column and only box
marks (two or more) or empty cells elsewhere, under distinct column captions.
"""

from __future__ import annotations

import re

from docparse.fixes.tables_html import HCell, parse_table

_MARK = re.compile(r"\[[ \t]?\]|\[[xX✓✔*]\]|[☐☑☒□■○●◉⦿✓✔]")
_CHECKED = re.compile(r"\[[xX✓✔*]\]|[☑☒■●◉⦿✓✔]")
_WORD = re.compile(r"[^\W\d_]{2,}")


def _grid(rows) -> dict[tuple[int, int], HCell]:
    out = {}
    for row in rows:
        for c in row.cells:
            for i in range(c.row, c.row + c.rowspan):
                for j in range(c.col, c.col + c.colspan):
                    out[(i, j)] = c
    return out


def _mark(cell: HCell | None) -> str | None:
    """The cell's box mark normalized (`[x]` / `[ ]`), "" for an empty cell, None for any other content."""
    if cell is None or not cell.text:
        return ""
    if not _MARK.fullmatch(cell.text):
        return None
    return "[x]" if _CHECKED.fullmatch(cell.text) else "[ ]"


def box_grid_lines(raw: str) -> str | None:
    """The grid table `raw` as lines, or None when it is not a grid of check boxes."""
    t = parse_table(raw)
    if t is None or len(t.rows) < 2:
        return None
    g = _grid(t.rows)
    nrows, ncols = len(t.rows), max(j for _, j in g) + 1
    body = []
    for i in range(nrows):
        first = g.get((i, 0))
        marks = [_mark(g.get((i, j))) for j in range(1, ncols)]
        if first is not None and first.row == i and _WORD.search(first.text) and None not in marks:
            if sum(bool(m) for m in marks) >= 2:
                body.append(i)
    if not body or body[0] == 0 or body != list(range(body[0], nrows)):
        return None
    top = body[0]
    captions = []
    for j in range(1, ncols):
        cap = g.get((top - 1, j))
        captions.append(cap.text if cap is not None and cap.colspan == 1 else "")
    used = [j for j in range(1, ncols) if any(_mark(g.get((i, j))) for i in body)]
    if any(not captions[j - 1] for j in used) or len({captions[j - 1] for j in used}) != len(used):
        return None
    head: list[str] = []
    for i in range(top):
        for j in range(ncols):
            c = g.get((i, j))
            if c is not None and c.text and c.row == i and c.col == j and c.text not in captions:
                head.append(c.text)
    lines = [" ".join(head)] if head else []
    for i in body:
        parts = [f"{_mark(g.get((i, j)))} {captions[j - 1]}" for j in used if _mark(g.get((i, j)))]
        lines.append(f"{g[(i, 0)].text}: " + " ".join(parts))
    return "\n".join(lines)

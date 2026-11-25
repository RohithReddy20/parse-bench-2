"""Category positions read from the chart's own category labels.

When a chart's bars cannot be grouped into one slot per table row (a category without a bar, bars of
different widths) or a line has more vertices than the table has rows, the category labels printed along the
category axis place each category: the table's category text is found among the text-layer words just outside
the plot (under it for vertical charts, left of it for horizontal ones), in order.
"""

from __future__ import annotations

import re

_NORM = re.compile(r"[^\w%.]+")
_NUM = re.compile(r"[\d.,%()+\-−–]+")


def norm(s: str) -> str:
    return _NORM.sub(" ", s.casefold()).strip()


def _lcs(a: str, b: str) -> int:
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _letters(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.casefold())


def label_columns(words: list, box: tuple[float, float, float, float], reach: float = 90.0) -> list[tuple[float, list]]:
    """Category labels printed rotated under a vertical chart: the text layer gives each one as a column of
    pieces (words split at spaces and at glyphs without a text mapping) sharing an x range. Returns (centre x,
    pieces from the bottom up) per column, left to right; a column must be taller than it is wide."""
    x0, _y0, x1, y1 = box
    pool = [
        w
        for w in words
        if y1 - 2 <= w.y0 <= y1 + reach
        and x0 - 10 <= w.cx <= x1 + 10
        and w.x1 - w.x0 <= 14
        and not _NUM.fullmatch(w.text)
        and not (len(w.text) >= 3 and w.x1 - w.x0 > w.y1 - w.y0)  # horizontal words
    ]
    cols: list[list] = []
    for w in sorted(pool, key=lambda w: w.cx):
        if cols:
            c = cols[-1]
            lo, hi = min(p.x0 for p in c), max(p.x1 for p in c)
            if min(hi, w.x1) - max(lo, w.x0) >= 0.5 * min(hi - lo, w.x1 - w.x0):
                c.append(w)
                continue
        cols.append([w])
    out = []
    for c in cols:
        lo, hi = min(p.x0 for p in c), max(p.x1 for p in c)
        top, bottom = min(p.y0 for p in c), max(p.y1 for p in c)
        if hi - lo > 14 or bottom - top < 1.5 * (hi - lo):
            continue
        out.append(((lo + hi) / 2, sorted(c, key=lambda p: -p.cy)))
    return out


def rotated_positions(cats: list[str], words: list, box, ordered: bool = True) -> list[float | None] | None:
    """Per category, the centre x of the rotated label column that prints it. Letters may be missing from a
    column (glyphs without a text mapping), so a column matches a category when nearly all its letters appear
    in order in the category's name and they cover most of it; each category and column is matched to its
    mutual best only. None unless 70% of the categories are found (in order, with `ordered`)."""
    cols = label_columns(words, box)
    if len(cols) < 2:
        return None
    texts = [_letters("".join(p.text for p in pieces)) for _, pieces in cols]
    names = [_letters(c) for c in cats]

    def score(i: int, k: int) -> float:
        t, n = texts[i], names[k]
        if len(t) < 2 or not n:
            return 0.0
        m = _lcs(t, n)
        if m < 0.8 * len(t) or m < 0.6 * len(n):
            return 0.0
        return m / max(len(t), len(n))

    table = [[score(i, k) for k in range(len(cats))] for i in range(len(cols))]
    out: list[float | None] = [None] * len(cats)
    for k in range(len(cats)):
        best_i = max(range(len(cols)), key=lambda i: table[i][k])
        s = table[best_i][k]
        if s <= 0:
            continue
        if sum(table[i][k] == s for i in range(len(cols))) > 1:
            continue  # two columns print it equally well
        row = table[best_i]
        if max(row) > s or sum(v == s for v in row) > 1:
            continue  # the column fits another category as well or better
        out[k] = cols[best_i][0]
    found = [c for c in out if c is not None]
    if len(found) < max(2, 0.7 * len(cats)):
        return None
    if ordered and not all(q > p for p, q in zip(found, found[1:], strict=False)):
        return None
    return out


def label_positions(
    cats: list[str], words: list, box: tuple[float, float, float, float], vertical: bool, reach: float = 60.0
) -> list[float | None] | None:
    """Per category, the centre (x for a vertical chart, y for a horizontal one) of the words printing its
    label beside the plot box (x0, y0, x1, y1); None for a label not found. None overall unless at least 70%
    of the categories are found, each once, in the table's order (or its reverse, for a horizontal chart)."""
    x0, y0, x1, y1 = box
    if vertical:
        pool = [w for w in words if y1 - 2 <= w.y0 <= y1 + reach and x0 - 10 <= w.cx <= x1 + 10]
    else:
        pool = [w for w in words if x0 - reach - 150 <= w.x0 and w.x1 <= x0 + 2 and y0 - 10 <= w.cy <= y1 + 10]
    if not pool:
        return None
    lines: list[list] = []
    for w in sorted(pool, key=lambda w: (round(w.cy), w.x0)):
        if lines and abs(lines[-1][-1].cy - w.cy) <= 2 and 0 <= w.x0 - lines[-1][-1].x1 <= 6:
            lines[-1].append(w)
        else:
            lines.append([w])
    spans: dict[str, list[float]] = {}
    for line in lines:  # every run of consecutive words on a line is a possible label
        for i in range(len(line)):
            for j in range(i, min(len(line), i + 4)):
                text = norm(" ".join(w.text for w in line[i : j + 1]))
                c = (line[i].x0 + line[j].x1) / 2 if vertical else (line[i].cy + line[j].cy) / 2
                spans.setdefault(text, []).append(c)
    out: list[float | None] = []
    for cat in cats:
        hits = spans.get(norm(cat), [])
        out.append(hits[0] if len(hits) == 1 else None)
    found = [c for c in out if c is not None]
    if len(found) < max(2, 0.7 * len(cats)):
        return None
    up = all(q > p for p, q in zip(found, found[1:], strict=False))
    down = all(q < p for p, q in zip(found, found[1:], strict=False))
    if not (up or (down and not vertical)):
        return None
    return out

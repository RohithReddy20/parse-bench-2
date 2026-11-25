"""Category positions read from the chart's own category labels.

When a chart's bars cannot be grouped into one slot per table row (a category without a bar, bars of
different widths) or a line has more vertices than the table has rows, the category labels printed along the
category axis place each category: the table's category text is found among the text-layer words just outside
the plot (under it for vertical charts, left of it for horizontal ones), in order.
"""

from __future__ import annotations

import re

_NORM = re.compile(r"[^\w%.]+")


def norm(s: str) -> str:
    return _NORM.sub(" ", s.casefold()).strip()


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

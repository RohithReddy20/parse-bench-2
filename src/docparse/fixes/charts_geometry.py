"""Chart geometry from a born-digital PDF page: value axes from tick labels, and the marks drawn against them.

    axes     numeric tick labels in the text layer, aligned in a column (a vertical value axis) or a row (a
             horizontal one), evenly spaced and linear in value (or in log10 value). Labels are tied to their
             tick (or gridline) when one is drawn at the label, else read at the label's centre.
    bars     axis-aligned filled rectangles (any subpath of a filled path), not white, not page backgrounds
    lines    stroked open polylines of one colour and width, chained where a series is drawn piece by piece
    markers  small filled closed shapes (dots, squares, diamonds): their centres

All coordinates are display points (origin top-left), as `docparse.pdf.graphics` and `words_in_points` give them.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from docparse.pdf.graphics import Color, Shape

EPS = 0.05  # points: rectangle corners agree to this
ALIGN = 1.5  # points: tick labels of one axis share an edge or centre to this
MAX_MARKER = 9.0  # points: the largest marker side
WHITE = 246  # a fill whose every channel is at least this is background, not a bar

_TICK = re.compile(r"[(]?[−–\-+]?[$€£¥]?\d{1,3}(?:[,   ]\d{3})*(?:\.\d+)?%?[)]?|[−–\-+]?[$€£¥]?\d+(?:\.\d+)?%?")


def tick_value(text: str) -> float | None:
    """The number a tick label prints ("−20%", "1,000", "$5", "0.25"); None for anything else."""
    t = text.strip()
    if t[:1] in "'’" and t[1:].isdigit() and len(t) == 3:
        t = t[1:]  # '07: a year label
    if not _TICK.fullmatch(t):
        return None
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").replace("−", "-").replace("–", "-")
    t = re.sub(r"[$€£¥%,   ]", "", t)
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


@dataclass
class Axis:
    orient: str  # "v": labels in a column, value varies with y; "h": labels in a row, value varies with x
    coords: list[float]  # tick positions along the axis
    values: list[float]
    pos: float  # where the labels sit across the axis (x for "v", y for "h")
    lo: float  # label column / row extent across the axis
    hi: float
    log: bool = False
    a: float = 0.0
    b: float = 0.0
    rules: tuple[float, float] | None = None  # extent across the axis of the gridlines the ticks snapped to

    def value(self, coord: float) -> float:
        v = self.a * coord + self.b
        return 10**v if self.log else v

    def coord(self, value: float) -> float | None:
        if self.log:
            if value <= 0:
                return None
            value = math.log10(value)
        return (value - self.b) / self.a if self.a else None

    @property
    def span(self) -> float:
        """Value range of the labels."""
        return max(self.values) - min(self.values)

    @property
    def step(self) -> float:
        return self.span / max(1, len(self.values) - 1)

    @property
    def extent(self) -> tuple[float, float]:
        return min(self.coords), max(self.coords)


def _fit(coords: list[float], values: list[float]) -> tuple[float, float, float]:
    """Least squares value = a * coord + b, and the largest residual as a share of the value span."""
    n = len(coords)
    mx, my = sum(coords) / n, sum(values) / n
    sxx = sum((c - mx) ** 2 for c in coords)
    if sxx == 0:
        return 0.0, my, math.inf
    a = sum((c - mx) * (v - my) for c, v in zip(coords, values, strict=True)) / sxx
    b = my - a * mx
    span = (max(values) - min(values)) or 1.0
    worst = max(abs(a * c + b - v) for c, v in zip(coords, values, strict=True)) / span
    return a, b, worst


def _even(coords: list[float]) -> bool:
    gaps = [q - p for p, q in zip(coords, coords[1:], strict=False)]
    if not gaps or min(gaps) <= 0:
        return False
    return max(gaps) - min(gaps) <= 0.08 * (sum(gaps) / len(gaps)) + 0.3


def _axis_from(members: list[tuple[float, float, object]], orient: str, upright: bool = True) -> Axis | None:
    """members: (coord along the axis, value, word). Evenly spaced, strictly monotone, linear (or log-linear)."""
    members = sorted(members, key=lambda m: m[0])
    coords = [m[0] for m in members]
    values = [m[1] for m in members]
    if len(coords) < 3 or len(set(values)) != len(values) or not _even(coords):
        return None
    diffs = [q - p for p, q in zip(values, values[1:], strict=False)]
    if not (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
        return None
    a, b, worst = _fit(coords, values)
    log = False
    if worst > 0.01:
        if min(values) <= 0:
            return None
        a, b, worst = _fit(coords, [math.log10(v) for v in values])
        log = True
        if worst > 0.01:
            return None
    if upright and (a > 0 if orient == "v" else a < 0):
        return None  # values growing downward or leftward: a misread scale (lost minus signs), not a value axis
    words = [m[2] for m in members]
    if orient == "v":
        pos = sum(w.x1 for w in words) / len(words)
        lo, hi = min(w.x0 for w in words), max(w.x1 for w in words)
    else:
        pos = sum(w.cy for w in words) / len(words)
        lo, hi = min(w.y0 for w in words), max(w.y1 for w in words)
    return Axis(orient, coords, values, pos, lo, hi, log, a, b)


def _runs(members: list[tuple[float, float, object]]) -> list[list[tuple[float, float, object]]]:
    """Maximal evenly spaced runs of a sorted column of labels (two panels stacked in one label column, or a
    stray number such as a figure number aligned with the ticks)."""
    members = sorted(members, key=lambda m: m[0])
    out: list[list[tuple[float, float, object]]] = []
    cur: list[tuple[float, float, object]] = members[:1]
    for m in members[1:]:
        if len(cur) >= 2:
            gap, last = m[0] - cur[-1][0], cur[-1][0] - cur[-2][0]
            if abs(gap - last) > 0.1 * last + 0.3:
                out.append(cur)
                cur = [cur[-1]]
        cur.append(m)
    out.append(cur)
    return [r for r in out if len(r) >= 3]


def _bent(run: list[tuple[float, float, object]], nums: list[tuple[object, float]], orient: str) -> bool:
    """The label just past either end of the run, in line with it, continues its values (the next tick) but
    not its spacing: the scale is not linear (a designed chart), so no value can be read from it."""
    words = [m[2] for m in run]
    lo = min((w.x0 if orient == "v" else w.y0) for w in words) - 2
    hi = max((w.x1 if orient == "v" else w.y1) for w in words) + 2
    along = (lambda w: w.cy) if orient == "v" else (lambda w: w.cx)
    step_c = (run[-1][0] - run[0][0]) / (len(run) - 1)
    step_v = (run[-1][1] - run[0][1]) / (len(run) - 1)
    jitter = max(abs(m[0] - (run[0][0] + i * step_c)) for i, m in enumerate(run))  # the labels' own scatter
    for sign, end in ((-1, run[0]), (1, run[-1])):
        beyond = [
            (abs(along(w) - end[0]), along(w), v)
            for w, v in nums
            if (w.x0 if orient == "v" else w.y0) <= hi
            and (w.x1 if orient == "v" else w.y1) >= lo
            and 0.3 * abs(step_c) < sign * (along(w) - end[0]) < 1.6 * abs(step_c)
        ]
        if not beyond:
            continue
        _, c, v = min(beyond)
        nxt_v, nxt_c = end[1] + sign * step_v, end[0] + sign * step_c
        if abs(v - nxt_v) <= 1e-9 * max(1.0, abs(v)) and abs(c - nxt_c) > max(0.15 * abs(step_c), 3 * jitter + 0.5):
            return True
    return False


def _restore_signs(row: list[tuple[float, float, object]], orient: str) -> list[tuple[float, float, object]]:
    """Labels that fall to 0 and rise again ("20 10 0 10 20 30" down a column) lost their minus signs (a font
    without the glyph in its text map): the side below zero (left of it on a horizontal axis) is negative."""
    vals = [v for _, v, _ in row]
    zeros = [i for i, v in enumerate(vals) if v == 0]
    if len(zeros) != 1 or not 0 < zeros[0] < len(vals) - 1 or min(vals) < 0:
        return row
    z = zeros[0]
    if not all(p > q for p, q in zip(vals[:z], vals[1 : z + 1], strict=True)):
        return row
    if not all(p < q for p, q in zip(vals[z:], vals[z + 1 :], strict=False)):
        return row
    flip = (lambda i: i > z) if orient == "v" else (lambda i: i < z)
    return [(c, -v if flip(i) else v, w) for i, (c, v, w) in enumerate(row)]


def _expand_years(row: list[tuple[float, float, object]]) -> list[tuple[float, float, object]]:
    """A year axis printed "2006 07 08 ... 23": two-digit labels after a full year are years of its century
    (rolling over at 00)."""
    out: list[tuple[float, float, object]] = []
    year: int | None = None
    for c, v, w in row:
        t = w.text.strip().lstrip("'’")
        if len(t) == 4 and t.isdigit() and 1800 <= v <= 2200:
            year = int(v)
        elif year is not None and len(t) == 2 and t.isdigit():
            cand = year // 100 * 100 + int(t)
            if cand < year:
                cand += 100
            year = cand
            v = float(cand)
        out.append((c, v, w))
    return out


def find_axes(
    words: list, years: bool = True, bent: bool = True, signs: bool = True, upright: bool = True
) -> list[Axis]:
    """Value axes printed as numeric tick labels (words with x0, y0, x1, y1, cx, cy and text, in points)."""
    nums = [(w, v) for w in words if (v := tick_value(w.text)) is not None]
    axes: list[tuple[frozenset[int], Axis]] = []
    seen: set[frozenset[int]] = set()
    keys = {
        "v": (lambda w: w.x1, lambda w: w.x0, lambda w: w.cx),
        "h": (lambda w: w.cy, lambda w: w.y1),
    }
    for orient, fns in keys.items():
        for key in fns:
            ordered = sorted(nums, key=lambda m: key(m[0]))
            groups: list[list[tuple[object, float]]] = []
            for m in ordered:
                if groups and key(m[0]) - key(groups[-1][-1][0]) <= ALIGN:
                    groups[-1].append(m)
                else:
                    groups.append([m])
            for g in groups:
                if len(g) < 3:
                    continue
                along = sorted((((w.cy if orient == "v" else w.cx), v, w) for w, v in g), key=lambda m: (m[0], m[1]))
                if orient == "h" and years:
                    along = _expand_years(along)
                if signs:
                    along = _restore_signs(along, orient)
                for run in _runs(along):
                    ids = frozenset(id(m[2]) for m in run)
                    if ids in seen or (bent and _bent(run, nums, orient)):
                        continue
                    ax = _axis_from(run, orient, upright)
                    if ax is not None:
                        seen.add(ids)
                        axes.append((ids, ax))
    # a run of some of an axis's labels (grouped by another edge) is the same axis
    return [ax for ids, ax in axes if not any(ids < other for other, _ in axes)]


# --------------------------------------------------------------------------- marks


@dataclass
class Bar:
    x0: float
    y0: float
    x1: float
    y1: float
    color: Color

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Line:
    points: list[tuple[float, float]]  # sorted by x
    color: Color
    width: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = [p[0] for p in self.points], [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class Marks:
    bars: list[Bar] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    markers: list[tuple[float, float, Color]] = field(default_factory=list)  # centre x, y, colour
    hrules: list[tuple[float, float, float]] = field(default_factory=list)  # y, x0, x1 of horizontal strokes
    vrules: list[tuple[float, float, float]] = field(default_factory=list)  # x, y0, y1 of vertical strokes
    keys: list[tuple[float, float, float, float, Color]] = field(default_factory=list)  # short coloured strokes


def _rect(points: tuple[tuple[float, float], ...]) -> tuple[float, float, float, float] | None:
    pts = list(points)
    if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) <= EPS and abs(pts[0][1] - pts[-1][1]) <= EPS:
        pts = pts[:-1]
    if len(pts) != 4:
        return None
    xs = sorted({round(p[0] / EPS) for p in pts})
    ys = sorted({round(p[1] / EPS) for p in pts})
    if len(xs) > 2 or len(ys) > 2:
        # corners may differ by rounding noise: accept two clusters per axis
        xv, yv = sorted(p[0] for p in pts), sorted(p[1] for p in pts)
        if xv[1] - xv[0] > EPS or xv[3] - xv[2] > EPS or yv[1] - yv[0] > EPS or yv[3] - yv[2] > EPS:
            return None
    for p, q in zip(pts, pts[1:] + pts[:1], strict=True):
        if abs(p[0] - q[0]) > EPS and abs(p[1] - q[1]) > EPS:
            return None  # a diagonal edge: not an axis-aligned rectangle
    xs_ = [p[0] for p in pts]
    ys_ = [p[1] for p in pts]
    return min(xs_), min(ys_), max(xs_), max(ys_)


def _white(c: Color) -> bool:
    return min(c) >= WHITE


def find_marks(shapes: list[Shape], page_w: float, page_h: float) -> Marks:
    marks = Marks()
    strokes: dict[tuple[Color, float], list[list[tuple[float, float]]]] = defaultdict(list)
    for s in shapes:
        if s.fill is not None and s.alpha > 0:
            for sub in s.subpaths:
                if len(sub) < 3:
                    continue
                r = _rect(sub)
                xs, ys = [p[0] for p in sub], [p[1] for p in sub]
                w, h = max(xs) - min(xs), max(ys) - min(ys)
                if r is not None:
                    if _white(s.fill) or w < 0.2 or h < 0.2:
                        continue
                    if w > 0.6 * page_w and h > 0.3 * page_h:
                        continue  # a page or panel background
                    marks.bars.append(Bar(*r, color=s.fill))
                    square = w <= MAX_MARKER and h <= MAX_MARKER and 0.6 <= w / h <= 1.6
                    dash = (h <= 1.5 and w >= 2 * h and w <= 3 * MAX_MARKER) or (
                        w <= 1.5 and h >= 2 * w and h <= 3 * MAX_MARKER
                    )
                    if square or dash:  # a dot or a stub marking a value
                        marks.markers.append(((r[0] + r[2]) / 2, (r[1] + r[3]) / 2, s.fill))
                elif w <= MAX_MARKER and h <= MAX_MARKER and w > 0.5 and h > 0.5 and not _white(s.fill):
                    marks.markers.append(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, s.fill))
        if s.stroke is not None and s.fill is None:
            for sub, closed in zip(s.subpaths, s.closed, strict=True):
                if len(sub) < 2:
                    continue
                if closed and len(sub) >= 3:
                    xs, ys = [p[0] for p in sub], [p[1] for p in sub]
                    if max(xs) - min(xs) <= MAX_MARKER and max(ys) - min(ys) <= MAX_MARKER:
                        marks.markers.append(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, s.stroke))
                    continue
                if len(sub) == 2:
                    (xa, ya), (xb, yb) = sub
                    if abs(ya - yb) <= EPS:
                        marks.hrules.append((ya, min(xa, xb), max(xa, xb)))
                        if 3 <= abs(xb - xa) <= 40:  # a short coloured stroke: a legend key for a line
                            marks.keys.append((min(xa, xb), ya - 1, max(xa, xb), ya + 1, s.stroke))
                    elif abs(xa - xb) <= EPS:
                        marks.vrules.append((xa, min(ya, yb), max(ya, yb)))
                strokes[(s.stroke, round(s.width, 2))].append(list(sub))
    for (color, width), pieces in strokes.items():
        for chain in _chain(pieces):
            xs, ys = [p[0] for p in chain], [p[1] for p in chain]
            if len(chain) == 2 and (max(ys) - min(ys) <= EPS or max(xs) - min(xs) <= EPS):
                continue  # a rule (gridline, tick, axis), not a series
            if len(chain) >= 2 and max(xs) - min(xs) > 1.0:
                marks.lines.append(Line(sorted(chain), color, width))
    return marks


def find_legend(marks: Marks, words: list) -> dict[Color, str]:
    """Series names by colour from legend entries: a small box, marker or short stroke with words starting just
    right of it on the same line (up to a wide gap or the next key). A colour with two different names, or a
    "name" that is a number (a data label), gives no entry."""
    keys: list[tuple[float, float, float, float, Color]] = [
        (b.x0, b.y0, b.x1, b.y1, b.color) for b in marks.bars if b.x1 - b.x0 <= 16 and b.y1 - b.y0 <= 16
    ]
    keys += marks.keys
    keys += [(x - 2, y - 2, x + 2, y + 2, c) for x, y, c in marks.markers]
    names: dict[Color, set[str]] = defaultdict(set)
    for _x0, y0, x1, y1, color in keys:
        cy, h = (y0 + y1) / 2, y1 - y0
        first = [w for w in words if 0 <= w.x0 - x1 <= 12 and abs(w.cy - cy) <= max(4.0, h)]
        if not first:
            continue
        line = [min(first, key=lambda w: w.x0)]
        while True:
            last = line[-1]
            gap = 0.6 * (last.y1 - last.y0) + 1
            nxt = [w for w in words if abs(w.cy - last.cy) <= 2 and 0 <= w.x0 - last.x1 <= gap and w not in line]
            if not nxt:
                break
            w = min(nxt, key=lambda w: w.x0)
            if any(last.x1 <= k[0] <= w.x0 and abs((k[1] + k[3]) / 2 - last.cy) <= 4 for k in keys):
                break  # the next key starts the next entry
            line.append(w)
        text = " ".join(w.text for w in line).strip()
        if text and tick_value(text) is None:
            names[color].add(text)
    return {c: next(iter(t)) for c, t in names.items() if len(t) == 1}


def _chain(pieces: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Join pieces whose end meets the next one's start (a series drawn one segment at a time)."""
    pieces = [p if p[0][0] <= p[-1][0] else p[::-1] for p in pieces]
    pieces.sort(key=lambda p: p[0][0])
    chains: list[list[tuple[float, float]]] = []
    for p in pieces:
        for c in chains:
            if abs(c[-1][0] - p[0][0]) <= 0.5 and abs(c[-1][1] - p[0][1]) <= 0.5:
                c.extend(p[1:])
                break
        else:
            chains.append(list(p))
    return chains


SNAP = 0.6  # points: a tick label's (shifted) position and its rule agree to this


def snap_axis(axis: Axis, marks: Marks) -> Axis:
    """Tie the labels to their drawn ticks or gridlines. Labels may sit beside their rule (above a gridline,
    say) by the same offset each: the offset (under 0.6 of the label spacing) that puts the most labels on a
    rule is applied when it puts at least 60% of them (and two) on one; matched labels take the rule's exact
    position."""
    coords = axis.coords
    spacing = (coords[-1] - coords[0]) / (len(coords) - 1)
    if axis.orient == "v":  # rules reaching to within 30 points of the label column
        segs = [(y, x0, x1) for y, x0, x1 in marks.hrules if x1 - x0 > 1 and x0 <= axis.hi + 30 and x1 >= axis.lo - 30]
    else:
        segs = [(x, y0, y1) for x, y0, y1 in marks.vrules if y1 - y0 > 1 and y0 <= axis.hi + 30 and y1 >= axis.lo - 30]
    if not segs:
        return axis
    rules = sorted(set(round(r, 2) for r, _, _ in segs))
    best: tuple[int, float, float] | None = None  # (matches, -|offset|, offset)
    for c in coords:
        for r in rules:
            off = r - c
            if abs(off) > SNAP * spacing:
                continue
            n = sum(any(abs(c2 + off - r2) <= SNAP for r2 in rules) for c2 in coords)
            key = (n, -abs(off), off)
            if best is None or key > best:
                best = key
    if best is None or best[0] < max(2, 0.6 * len(coords)):
        return axis
    off = best[2]
    snapped = []
    for c in coords:
        hit = [r for r in rules if abs(c + off - r) <= SNAP]
        snapped.append(min(hit, key=lambda r: abs(c + off - r)) if hit else c + off)
    if axis.log:
        a, b, worst = _fit(snapped, [math.log10(v) for v in axis.values])
    else:
        a, b, worst = _fit(snapped, axis.values)
    if worst > 0.01:
        return axis
    on = [(lo, hi) for r, lo, hi in segs if any(abs(r - c) <= SNAP for c in snapped)]
    extent = (min(lo for lo, _ in on), max(hi for _, hi in on)) if on else None
    return Axis(axis.orient, snapped, axis.values, axis.pos, axis.lo, axis.hi, axis.log, a, b, extent)

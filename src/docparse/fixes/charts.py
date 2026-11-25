"""Chart values measured from the PDF's vector drawing replace OCRia-1's readings (design §6, charts).

OCRia-1 writes each chart as an HTML table and reads its values off the image. On a born-digital page the bars,
lines and markers are path objects and the value axis is printed as tick labels, so a value can be measured
exactly. The fix keeps OCRia-1's table (its categories, series names, order and layout) and only rewrites
numbers it can tie to a measurement:

1. Value axes come from evenly spaced numeric tick labels (`charts_geometry.find_axes`). Each axis governs a
   zone on either side of its labels, up to the next label column.
2. In a zone, bars are grouped into category slots along the category axis (the number of slots must equal the
   table's number of categories), and each fill colour is one measured series. A line maps to the categories
   by its vertices (one per category) or, when the categories are numbers (years), through the numeric
   category axis printed under it. Markers of one colour work like line vertices.
3. A table series (a column, or a row in the transposed reading) binds to the measured series whose values
   agree with OCRia-1's own reading: at least `agree` of the shared cells within `agree_tol` x the axis value
   range, and the two following each other where the series varies. Every measured series and table series
   binds once, best agreement first.
4. Bound cells get the measured number, printed with the cell's own sign, currency and separators, at a
   precision of a tenth of the axis step. Cells that are not numbers, unbound series and tables with no
   binding are written back byte for byte.

Options (`chart_options`): measure (on), agree (0.6), agree_tol (0.1), replace_min (0.0, as a share of the
axis range: closer readings are kept), coverage (0.7: share of the series' cells that must be measured).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from docparse.core.config import CHART_OPTIONS  # noqa: F401  (the option names, validated in Config.load)
from docparse.core.ir import Candidate, Page, Provenance
from docparse.fixes.charts_geometry import Axis, Bar, Line, Marks, find_axes, find_marks, snap_axis, tick_value
from docparse.fixes.charts_table import Cell, Number, format_number, number_of, parse_cells, replace_spans
from docparse.pdf.graphics import Shape

READER = "fix.charts.vector"


# --------------------------------------------------------------------------- table series


@dataclass
class Unit:
    """One series of an OCRia-1 table: its category labels and, per category, the cell and its number."""

    name: str
    cats: list[str]
    cells: list[Cell | None]
    nums: list[Number | None]

    @property
    def values(self) -> list[float | None]:
        return [n.value if n else None for n in self.nums]


def table_units(raw: str) -> list[list[Unit]]:
    """The table's series read two ways: categories down the rows (one unit per value column, split at
    section rows and at a change of the outer row label), and categories across the header (one unit per
    body row, split at a change of the outer header)."""
    cells = parse_cells(raw)
    if not cells:
        return []
    grid: dict[tuple[int, int], Cell] = {}
    for c in cells:
        for dr in range(c.rowspan):
            for dc in range(c.colspan):
                grid[(c.row + dr, c.col + dc)] = c
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    nums = {id(c): number_of(raw, c) for c in cells}

    def row_has_number(r: int) -> bool:
        return any(nums[id(grid[(r, c)])] for c in range(n_cols) if (r, c) in grid and not grid[(r, c)].th)

    hdr = 0
    while hdr < n_rows and all(grid[(hdr, c)].th for c in range(n_cols) if (hdr, c) in grid):
        hdr += 1
    if hdr == 0 and not row_has_number(0):
        hdr = 1
    body = list(range(hdr, n_rows))
    lab = 0
    while lab < n_cols and all(
        (r, lab) not in grid or grid[(r, lab)].th or nums[id(grid[(r, lab)])] is None for r in body
    ):
        lab += 1
    if lab == 0 or lab >= n_cols or not body:
        return []

    def text(r: int, c: int) -> str:
        return grid[(r, c)].text if (r, c) in grid else ""

    def value_cell(r: int, c: int) -> tuple[Cell | None, Number | None]:
        cell = grid.get((r, c))
        if cell is None or cell.th or cell.row != r or cell.col != c or cell.rowspan > 1 or cell.colspan > 1:
            return None, None
        return cell, nums[id(cell)]

    def dedupe(parts: list[str]) -> str:
        out: list[str] = []
        for p in parts:
            if p and p not in out:
                out.append(p)
        return " / ".join(out)

    # categories down the rows
    groups: list[list[int]] = []
    for r in body:
        section = not row_has_number(r)
        outer = text(r, 0) if lab >= 2 else ""
        if section:
            groups.append([])
            continue
        if not groups or (lab >= 2 and groups[-1] and text(groups[-1][-1], 0) != outer):
            groups.append([])
        groups[-1].append(r)
    down: list[Unit] = []
    for rows in (g for g in groups if g):
        cats = [dedupe([text(r, c) for c in range(1 if lab >= 2 else 0, lab)]) for r in rows]
        for c in range(lab, n_cols):
            name = dedupe([text(r, c) for r in range(hdr)])
            got = [value_cell(r, c) for r in rows]
            down.append(Unit(name, cats, [g[0] for g in got], [g[1] for g in got]))
    # categories across the header
    across: list[Unit] = []
    col_groups: list[list[int]] = []
    for c in range(lab, n_cols):
        if not col_groups or (hdr >= 2 and text(0, c) != text(0, col_groups[-1][-1])):
            col_groups.append([])
        col_groups[-1].append(c)
    for cols in col_groups:
        cats = [dedupe([text(r, c) for r in range(1 if hdr >= 2 else 0, hdr)]) for c in cols]
        for r in body:
            got = [value_cell(r, c) for c in cols]
            across.append(
                Unit(dedupe([text(r, c) for c in range(lab)]), cats, [g[0] for g in got], [g[1] for g in got])
            )
    return [down, across]


# --------------------------------------------------------------------------- measured series


@dataclass
class Zone:
    axis: Axis
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 where its marks lie
    bars: list[Bar]
    lines: list[Line]
    markers: list[tuple[float, float, tuple[int, int, int]]]
    cat_axis: Axis | None  # a numeric axis across the categories (years under a line chart)


@dataclass
class Measured:
    key: tuple  # the physical marks (a series binds once, whatever axis or reading it is tried with)
    axis: Axis
    values: list[float | None]


def _swatch(b: Bar, words: list) -> bool:
    """A legend key: a small box with a word starting just right of it on the same line."""
    w, h = b.x1 - b.x0, b.y1 - b.y0
    if w > 16 or h > 16:
        return False
    return any(0 <= wd.x0 - b.x1 <= 14 and abs(wd.cy - b.cy) <= max(h, 4.0) for wd in words)


def _inside(box: tuple[float, float, float, float], x0: float, y0: float, x1: float, y1: float, slack: float) -> bool:
    return x0 >= box[0] - slack and x1 <= box[2] + slack and y0 >= box[1] - slack and y1 <= box[3] + slack


def _series_bars(bars: list[Bar], vertical: bool) -> list[Bar]:
    """Bars of a series share their thickness: per colour, the bars of the most common thickness (at least two);
    boxes of other sizes (callouts, panels, a legend key) are not bars."""
    by_color: dict[tuple[int, int, int], list[Bar]] = {}
    for b in bars:
        by_color.setdefault(b.color, []).append(b)
    out: list[Bar] = []
    for group in by_color.values():
        thick = [round(((b.x1 - b.x0) if vertical else (b.y1 - b.y0)) * 4) / 4 for b in group]
        mode = max(set(thick), key=thick.count)
        keep = [b for b, t in zip(group, thick, strict=True) if abs(t - mode) <= 0.3]
        if len(keep) >= 2:
            out += keep
    return out


def zones(axes: list[Axis], marks: Marks, words: list, page_w: float, page_h: float) -> list[Zone]:
    """For each value axis, the regions on either side of its labels up to the next label column (or row)."""
    bars = [b for b in marks.bars if not _swatch(b, words)]
    out: list[Zone] = []
    for ax in axes:
        lo_c, hi_c = ax.extent
        pad = 0.75 * (hi_c - lo_c) / max(1, len(ax.coords) - 1)
        same = [
            o
            for o in axes
            if o is not ax and o.orient == ax.orient and min(o.extent[1], hi_c) - max(o.extent[0], lo_c) > 0
        ]
        before = max((o.hi for o in same if o.hi <= ax.lo), default=0.0)
        after = min((o.lo for o in same if o.lo >= ax.hi), default=page_w if ax.orient == "v" else page_h)
        for side in ((before, ax.lo), (ax.hi, after)):
            if ax.orient == "v":
                box = (side[0], lo_c - pad, side[1], hi_c + pad)
            else:
                box = (lo_c - pad, side[0], hi_c + pad, side[1])
            zb = _series_bars([b for b in bars if _inside(box, b.x0, b.y0, b.x1, b.y1, 0.5)], ax.orient == "v")
            zl = [ln for ln in marks.lines if _inside(box, *ln.bbox, 1.0) and len(ln.points) >= 2]
            zm = [m for m in marks.markers if _inside(box, m[0], m[1], m[0], m[1], 0.5)]
            if not (zb or zl or zm):
                continue
            cat_axis = None
            if ax.orient == "v":
                under = [
                    o
                    for o in axes
                    if o.orient == "h"
                    and not o.log
                    and box[3] - 2 * pad <= o.pos <= box[3] + 40
                    and min(o.extent[1], box[2]) - max(o.extent[0], box[0]) > 0.5 * (o.extent[1] - o.extent[0])
                ]
                cat_axis = min(under, key=lambda o: o.pos) if under else None
            out.append(Zone(ax, box, zb, zl, zm, cat_axis))
    return out


def _slots(bars: list[Bar], n: int, vertical: bool) -> list[list[Bar]] | None:
    """Bars grouped into n category slots along the category axis: overlapping bars (stacks) share a slot, and
    side-by-side groups split at the n - 1 widest gaps, which must stand out from the gaps inside groups."""
    if n < 2 or not bars:
        return None
    lo = (lambda b: b.x0) if vertical else (lambda b: b.y0)
    hi = (lambda b: b.x1) if vertical else (lambda b: b.y1)
    runs: list[list[Bar]] = []
    end = -math.inf
    for b in sorted(bars, key=lo):
        if runs and lo(b) < end - 0.3:
            runs[-1].append(b)
            end = max(end, hi(b))
        else:
            runs.append([b])
            end = hi(b)
    if len(runs) < n:
        return None
    if len(runs) == n:
        return runs
    gaps = [(lo(runs[i + 1][0]) - max(hi(b) for b in runs[i]), i) for i in range(len(runs) - 1)]
    ranked = sorted(gaps, reverse=True)
    cut, rest = ranked[: n - 1], ranked[n - 1 :]
    if rest and min(g for g, _ in cut) <= 1.5 * max(g for g, _ in rest) + 0.5:
        return None
    bounds = sorted(i for _, i in cut)
    out, start = [], 0
    for i in bounds:
        out.append([b for r in runs[start : i + 1] for b in r])
        start = i + 1
    out.append([b for r in runs[start:] for b in r])
    return out


def _bar_values(b: Bar, ax: Axis) -> tuple[float, float]:
    """(value at the bar's far end, signed length from its near end): the near end is the one closer to zero."""
    if ax.orient == "v":
        va, vb = ax.value(b.y1), ax.value(b.y0)
    else:
        va, vb = ax.value(b.x0), ax.value(b.x1)
    near, far = (va, vb) if abs(va) <= abs(vb) else (vb, va)
    return far, far - near


def _cat_numbers(cats: list[str]) -> list[float | None]:
    return [tick_value(c) for c in cats]


def _sample(points: list[tuple[float, float]], x: float) -> float | None:
    if not points or x < points[0][0] - 0.5 or x > points[-1][0] + 0.5:
        return None
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x0 - 0.5 <= x <= x1 + 0.5:
            if x1 - x0 < 1e-6:
                return y0
            t = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
            return y0 + t * (y1 - y0)
    return points[0][1] if len(points) == 1 else None


def _vertices(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in points:
        if out and abs(p[0] - out[-1][0]) < 0.05 and abs(p[1] - out[-1][1]) < 0.05:
            continue
        out.append(p)
    return out


def measured_series(z: Zone, cats: list[str], bar_mode: str = "auto") -> list[Measured]:
    n = len(cats)
    ax = z.axis
    out: list[Measured] = []
    vertical = ax.orient == "v"
    slots = _slots(z.bars, n, vertical)
    if slots is not None:
        colors = {b.color for b in z.bars}
        for color in colors:
            ends: list[float | None] = []
            lens: list[float | None] = []
            ok = True
            for slot in slots:
                mine = [b for b in slot if b.color == color]
                if len(mine) > 1:
                    ok = False
                    break
                if mine:
                    far, length = _bar_values(mine[0], ax)
                    ends.append(far)
                    lens.append(length)
                else:
                    ends.append(None)
                    lens.append(None)
            if ok and sum(v is not None for v in ends) >= 2:
                key = (
                    "bars",
                    color,
                    frozenset((round(b.x0, 1), round(b.y0, 1)) for s in slots for b in s if b.color == color),
                )
                # bars grow from zero when the axis shows it (their length is the value, stacked or not); on an
                # axis that starts above zero a bar's value is where it ends
                from_zero = not ax.log and min(ax.values) <= 0 <= max(ax.values)
                if bar_mode == "both" or from_zero:
                    out.append(Measured(key, ax, lens))
                if bar_mode == "both" or not from_zero:
                    out.append(Measured(key, ax, ends))
    if not vertical:  # horizontal value axis: markers of one colour, top to bottom, one per category
        groups: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
        for x, y, c in z.markers:
            groups.setdefault(c, []).append((y, x))
        for color, pts in groups.items():
            pts = sorted(pts)
            if len(pts) == n and all(q[0] - p[0] > 0.5 for p, q in zip(pts, pts[1:], strict=False)):
                out.append(Measured(("markers", color, round(pts[0][0], 1)), ax, [ax.value(x) for _, x in pts]))
        return out
    xs_num = _cat_numbers(cats)
    cat_x: list[float | None] | None = None
    if z.cat_axis is not None and sum(v is not None for v in xs_num) >= 0.8 * n:
        cat_x = [z.cat_axis.coord(v) if v is not None else None for v in xs_num]
    for ln in z.lines:
        pts = _vertices(ln.points)
        key = ("line", ln.color, round(pts[0][0], 1), round(pts[0][1], 1))
        if len(pts) == n:
            out.append(Measured(key, ax, [ax.value(y) for _, y in pts]))
        if cat_x is not None:
            vals = [None if x is None else _sample(pts, x) for x in cat_x]
            out.append(Measured(key, ax, [None if y is None else ax.value(y) for y in vals]))
    by_color: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
    for x, y, c in z.markers:
        by_color.setdefault(c, []).append((x, y))
    for color, pts in by_color.items():
        pts = sorted(pts)
        key = ("markers", color, round(pts[0][0], 1))
        if len(pts) == n:
            out.append(Measured(key, ax, [ax.value(y) for _, y in pts]))
        elif cat_x is not None:
            vals = []
            for x in cat_x:
                hit = [p for p in pts if x is not None and abs(p[0] - x) <= 1.5]
                vals.append(ax.value(hit[0][1]) if len(hit) == 1 else None)
            out.append(Measured(key, ax, vals))
    return out


# --------------------------------------------------------------------------- binding


def _agreement(unit: Unit, m: Measured, opts: dict, data_range: float = math.inf) -> tuple[float, float, int] | None:
    """(share of shared cells within tolerance, median error as a share of the range, shared cells), or None
    when the measured series cannot stand for the unit. The range is the axis's, or the table's own value
    range (zero included) when that is smaller: a series near zero on a wide axis must still match closely."""
    vals = unit.values
    have = [i for i, v in enumerate(vals) if v is not None]
    if len(have) < 3 or len(m.values) != len(vals):
        return None
    both = [i for i in have if m.values[i] is not None]
    if len(both) < max(3, opts.get("coverage", 0.7) * len(have)):
        return None
    ax = m.axis

    def t(v: float) -> float | None:
        if ax.log:
            return math.log10(v) if v > 0 else None
        return v

    pairs = [(t(vals[i]), t(m.values[i])) for i in both]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    span = (math.log10(max(ax.values)) - math.log10(min(ax.values))) if ax.log else min(ax.span, data_range)
    if span <= 0:
        return None
    errs = sorted(abs(a - b) / span for a, b in pairs)
    share = sum(e <= opts.get("agree_tol", 0.1) for e in errs) / len(errs)
    med = errs[len(errs) // 2]
    va = [a for a, _ in pairs]
    vb = [b for _, b in pairs]
    sd_a, sd_b = _sd(va), _sd(vb)
    if sd_b < 0.01 * span and sd_a >= 0.01 * span:
        return None  # a flat reading (a rule, a run of equal stubs) cannot stand for a series that moves
    if sd_a >= 0.05 * span and (sd_b == 0 or _corr(va, vb) < 0.7):
        return None
    return share, med, len(pairs)


def _sd(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _corr(a: list[float], b: list[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def _decimals(ax: Axis, unit: Unit) -> int:
    own = max((n.decimals for n in unit.nums if n), default=0)
    if ax.log:
        return own
    res = ax.step / 10
    d = max(0, math.ceil(-math.log10(res) - 1e-9)) if res > 0 else 0
    return min(4, max(d, own))


def data_range(units: list[Unit]) -> float:
    """The value range of a table's numbers, zero included when they share a sign (the scorer's range)."""
    vals = [v for u in units for v in u.values if v is not None]
    if not vals:
        return 0.0
    lo, hi = min(vals), max(vals)
    if lo >= 0 or hi <= 0:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    return hi - lo


def bind(options: list[tuple[Unit, list[Measured]]], opts: dict) -> list[tuple[Unit, Measured]]:
    """Each unit with the measured series it may stand for: pairs in order of agreement, each side once."""
    scored = []
    rng = data_range([u for u, _ in options]) or math.inf
    for ui, (u, cands) in enumerate(options):
        for ci, m in enumerate(cands):
            a = _agreement(u, m, opts, rng)
            if a is None:
                continue
            share, med, n = a
            if share < opts.get("agree", 0.6):
                continue
            scored.append((share, -med, n, ui, ci))
    scored.sort(reverse=True)
    used_u: set[int] = set()
    used_k: set[tuple] = set()
    out: list[tuple[Unit, Measured]] = []
    for _share, _med, _n, ui, ci in scored:
        u, cands = options[ui]
        m = cands[ci]
        if ui in used_u or m.key in used_k or any(_same(m, o) for _, o in out):
            continue
        used_u.add(ui)
        used_k.add(m.key)
        out.append((u, m))
    return out


def _same(a: Measured, b: Measured) -> bool:
    """Two readings of one drawn series (a line and the markers on its vertices)."""
    if len(a.values) != len(b.values) or a.axis is not b.axis:
        return False
    both = [(x, y) for x, y in zip(a.values, b.values, strict=True) if x is not None and y is not None]
    tol = 0.005 * (a.axis.span or 1.0)
    return len(both) >= 2 and sum(abs(x - y) <= tol for x, y in both) >= 0.8 * len(both)


def printed_numbers(words: list, axes: list[Axis]) -> set[tuple[float, int]]:
    """(value, decimals) of every number printed on the page outside the tick labels: data labels."""
    out: set[tuple[float, int]] = set()
    ticks = {(round(w_c, 1), v) for a in axes for w_c, v in zip(a.coords, a.values, strict=True)}
    for w in words:
        v = tick_value(w.text.strip(".,;:"))
        if v is None:
            continue
        c = w.cy
        if any(abs(v - tv) < 1e-9 and abs(c - tc) < 8 for tc, tv in ticks) or any(
            abs(v - tv) < 1e-9 and abs(w.cx - tc) < 8 for tc, tv in ticks
        ):
            continue
        frac = re.search(r"\.(\d+)", w.text)
        out.add((round(v, 6), len(frac.group(1)) if frac else 0))
    return out


def _printed(num: Number, printed: set[tuple[float, int]]) -> bool:
    return (round(num.value, 6), num.decimals) in printed or (round(-num.value, 6), num.decimals) in printed


def _edits(
    raw: str, pairs: list[tuple[Unit, Measured]], opts: dict, printed: set[tuple[float, int]]
) -> list[tuple[tuple[int, int], str]]:
    edits = []
    for u, m in pairs:
        ax = m.axis
        span = ax.span
        d = _decimals(ax, u)
        for num, v in zip(u.nums, m.values, strict=True):
            if num is None or v is None:
                continue
            if _printed(num, printed):
                continue  # the chart prints this number (a data label): it is exact
            if abs(v - num.value) <= opts.get("replace_min", 0.0) * span:
                continue
            text = format_number(v, d, num)
            if text != raw[num.span[0] : num.span[1]]:
                edits.append((num.span, text))
    return edits


def correct_tables(page: Page, words: list, shapes: list[Shape], opts: dict) -> int:
    """Add a measured candidate to every chart table that binds; returns the number of rewritten cells."""
    tables = [r for r in page.regions if r.kind == "table"]
    if not tables or not shapes:
        return 0
    marks = find_marks(shapes, page.width, page.height)
    axes = [snap_axis(a, marks) for a in find_axes(words)]
    if not axes:
        return 0
    zs = zones(axes, marks, words, page.width, page.height)
    if not zs:
        return 0
    changed = 0
    used: set[tuple] = set()
    mode = str(opts.get("bar_mode", "auto"))
    printed = printed_numbers(words, axes)
    for region in tables:
        base = region.chosen()
        best: tuple[int, list[tuple[Unit, Measured]]] | None = None
        for units in table_units(base.raw):
            by_cats: dict[tuple[str, ...], list[Measured]] = {}
            for u in units:
                key = tuple(u.cats)
                if key not in by_cats:
                    by_cats[key] = [m for z in zs for m in measured_series(z, u.cats, mode) if m.key not in used]
            pairs = bind([(u, by_cats[tuple(u.cats)]) for u in units], opts)
            n = sum(
                sum(v is not None and x is not None for v, x in zip(u.values, m.values, strict=True)) for u, m in pairs
            )
            if pairs and (best is None or n > best[0]):
                best = (n, pairs)
        if best is None:
            continue
        edits = _edits(base.raw, best[1], opts, printed)
        used |= {m.key for _, m in best[1]}
        if not edits:
            continue
        raw = replace_spans(base.raw, edits)
        cand = Candidate(
            id=f"{region.id}c{len(region.candidates)}",
            reader=READER,
            raw=raw,
            prov=[Provenance(source="pdf_drawing", ref="vector", detail=f"{len(edits)} cells")],
        )
        region.candidates.append(cand)
        region.selected = cand.id
        changed += len(edits)
    return changed

"""Vector chart hint extraction for born-digital PDF pages.

Extracts a compact, deterministic structured summary (axes, legend, measured bar/line/marker series,
in-plot printed labels, and nearby text) from a PDF page's text-layer words and vector shapes, and
formats it as a concise text block suitable to pass alongside a high-resolution chart crop to a VLM.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from docparse.fixes.charts_geometry import (
    Axis,
    Bar,
    Line,
    Marks,
    find_axes,
    find_legend,
    find_marks,
    snap_axis,
    tick_value,
)
from docparse.fixes.layout_refine import PtWord, lines_of, words_in_points
from docparse.pdf.graphics import Color, Shape
from docparse.pdf.render import RenderedPage

BBox = tuple[float, float, float, float]

MAX_SERIES = 12
MAX_POINTS_PER_SERIES = 40
MAX_PRINTED_LABELS = 60
MAX_NEARBY_LINES = 25
MAX_LEGEND_ENTRIES = 24


@dataclass
class _PlotZone:
    axis_index: int
    axis: Axis
    plot_box: BBox  # tight plot area (x0, y0, x1, y1)
    search_box: BBox  # padded box for catching marks slightly past ticks
    cat_axes: tuple[Axis, ...] = ()
    side: str = ""  # "left" or "right" for vertical axes


# --------------------------------------------------------------------------- numeric / geometry helpers


def _color_hex(c: Color) -> str:
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def _clean_float(v: float, ndigits: int = 4) -> float:
    r = round(float(v), ndigits)
    return 0.0 if abs(r) < 1e-9 else r


def _precision_for_axis(ax: Axis) -> int:
    if ax.log:
        return 3
    step = abs(ax.step)
    if step <= 0:
        return 2
    d = max(0, math.ceil(-math.log10(step / 10.0) - 1e-9))
    if d == 0 and step <= 10.0:
        d = 1
    return min(4, d)


def _round_axis_value(ax: Axis, value: float) -> float:
    if ax.log:
        if value == 0:
            return 0.0
        mag = math.floor(math.log10(abs(value)))
        return _clean_float(round(value, max(0, 2 - mag)), 4)
    return _clean_float(value, _precision_for_axis(ax))


def _fmt_num(v: float) -> str:
    if abs(v) < 1e-9:
        return "0"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _norm_bbox(bbox: BBox) -> BBox:
    x0, y0, x1, y1 = bbox
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _expand_bbox(bbox: BBox, dx: float, dy: float | None = None) -> BBox:
    dy = dx if dy is None else dy
    return (bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy)


def _pt_in_box(x: float, y: float, box: BBox, slack: float = 0.0) -> bool:
    return box[0] - slack <= x <= box[2] + slack and box[1] - slack <= y <= box[3] + slack


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _boxes_overlap(a: BBox, b: BBox, slack: float = 0.0) -> bool:
    return (
        a[0] <= b[2] + slack
        and a[2] >= b[0] - slack
        and a[1] <= b[3] + slack
        and a[3] >= b[1] - slack
    )


def _infer_page_size(words: Sequence[PtWord], shapes: Sequence[Shape], bbox: BBox | None) -> tuple[float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for w in words:
        xs.append(w.x1)
        ys.append(w.y1)
    for s in shapes:
        for sub in s.subpaths:
            for x, y in sub:
                xs.append(x)
                ys.append(y)
    if bbox is not None:
        xs.append(bbox[2])
        ys.append(bbox[3])
    return (max(xs) if xs else 612.0, max(ys) if ys else 792.0)


def _zero_in_reach(ax: Axis, reach: float = 1.0) -> bool:
    slack = (reach + 0.01) * ax.step if reach else 0.0
    return not ax.log and min(ax.values) - slack <= 0.0 <= max(ax.values) + slack


def _is_swatch(b: Bar, words: Sequence[PtWord]) -> bool:
    w, h = b.x1 - b.x0, b.y1 - b.y0
    if w > 16.0 or h > 16.0:
        return False
    return any(0.0 <= wd.x0 - b.x1 <= 14.0 and abs(wd.cy - b.cy) <= max(h, 4.0) for wd in words)


def _is_legend_marker(x: float, y: float, words: Sequence[PtWord]) -> bool:
    return any(0.0 <= wd.x0 - (x + 2.0) <= 14.0 and abs(wd.cy - y) <= 4.0 for wd in words)


def _axis_touches_bbox(ax: Axis, bbox: BBox) -> bool:
    bx0, by0, bx1, by1 = bbox
    lo, hi = ax.extent
    span = max(hi - lo, 1e-6)
    if ax.orient == "v":
        y_ov = _overlap_1d(lo, hi, by0 - 12.0, by1 + 12.0)
        if y_ov < 0.5 * span:
            return False
        x_hit = _overlap_1d(ax.lo, ax.hi, bx0 - 20.0, bx1 + 20.0) > 0.0
        rule_hit = ax.rules is not None and _overlap_1d(ax.rules[0], ax.rules[1], bx0, bx1) > 0.0
        return x_hit or rule_hit
    x_ov = _overlap_1d(lo, hi, bx0 - 12.0, bx1 + 12.0)
    if x_ov < 0.5 * span:
        return False
    y_hit = _overlap_1d(ax.lo, ax.hi, by0 - 20.0, by1 + 20.0) > 0.0
    rule_hit = ax.rules is not None and _overlap_1d(ax.rules[0], ax.rules[1], by0, by1) > 0.0
    return y_hit or rule_hit


def _filter_marks(marks: Marks, bbox: BBox, slack: float = 4.0) -> Marks:
    bars = [b for b in marks.bars if _pt_in_box(b.cx, b.cy, bbox, slack)]
    lines: list[Line] = []
    for ln in marks.lines:
        inside = [p for p in ln.points if _pt_in_box(p[0], p[1], bbox, slack + 4.0)]
        if len(inside) >= 2 and len(inside) >= 0.5 * len(ln.points):
            lines.append(Line(inside, ln.color, ln.width))
    markers = [m for m in marks.markers if _pt_in_box(m[0], m[1], bbox, slack)]
    hrules = [
        r
        for r in marks.hrules
        if bbox[1] - 12.0 <= r[0] <= bbox[3] + 12.0 and _overlap_1d(r[1], r[2], bbox[0], bbox[2]) > 0
    ]
    vrules = [
        r
        for r in marks.vrules
        if bbox[0] - 12.0 <= r[0] <= bbox[2] + 12.0 and _overlap_1d(r[1], r[2], bbox[1], bbox[3]) > 0
    ]
    keys = [k for k in marks.keys if _pt_in_box((k[0] + k[2]) / 2, (k[1] + k[3]) / 2, bbox, 36.0)]
    return Marks(bars=bars, lines=lines, markers=markers, hrules=hrules, vrules=vrules, keys=keys)


# --------------------------------------------------------------------------- axis ticks & zone construction


def _extract_axis_ticks(
    raw_ax: Axis, words: Sequence[PtWord]
) -> tuple[list[tuple[str, float]], set[int]]:
    """Recover printed tick labels and matched PtWord ids for a raw (pre-snap) Axis."""
    ticks: list[tuple[str, float]] = []
    used_ids: set[int] = set()
    for coord, val in zip(raw_ax.coords, raw_ax.values, strict=True):
        if raw_ax.orient == "v":
            cands = [
                w
                for w in words
                if w.x0 >= raw_ax.lo - 1.5
                and w.x1 <= raw_ax.hi + 1.5
                and abs(w.cy - coord) <= 1.5
                and tick_value(w.text) is not None
            ]
            best = min(cands, key=lambda w: abs(w.cy - coord)) if cands else None
        else:
            cands = [
                w
                for w in words
                if w.y0 >= raw_ax.lo - 1.5
                and w.y1 <= raw_ax.hi + 1.5
                and abs(w.cx - coord) <= 1.5
                and tick_value(w.text) is not None
            ]
            best = min(cands, key=lambda w: abs(w.cx - coord)) if cands else None

        clean_val = _clean_float(val)
        if best is not None:
            used_ids.add(id(best))
            txt = best.text.strip()
            parsed = tick_value(txt)
            if parsed is not None and abs(parsed - val) <= 1e-6 * max(1.0, abs(val)):
                label = txt
            elif parsed is not None and val < 0 and abs(parsed + val) <= 1e-6 * max(1.0, abs(val)):
                label = f"-{txt}"
            elif raw_ax.orient == "h" and 1800 <= val <= 2200 and len(txt.lstrip("'’")) == 2:
                label = txt
            else:
                label = _fmt_num(clean_val)
        else:
            label = _fmt_num(clean_val)
        ticks.append((label, clean_val))

    ticks.sort(key=lambda t: t[1])
    return ticks, used_ids


def _same_scale(a: Axis, b: Axis) -> bool:
    if a.orient != b.orient or a.log != b.log:
        return False
    if abs(a.a) < 1e-9 or abs(b.a) < 1e-9:
        return False
    if abs(a.a - b.a) / max(abs(a.a), abs(b.a)) > 0.03:
        return False
    mid = (max(a.extent[0], b.extent[0]) + min(a.extent[1], b.extent[1])) / 2.0
    span = max(a.span, b.span, 1e-6)
    return abs(a.value(mid) - b.value(mid)) <= 0.03 * span


def _has_marks_in_box(marks: Marks, box: BBox, words: Sequence[PtWord]) -> bool:
    if any(_pt_in_box(b.cx, b.cy, box, 2.0) and not _is_swatch(b, words) for b in marks.bars):
        return True
    if any(
        len(ln.points) >= 2
        and (max(p[0] for p in ln.points) - min(p[0] for p in ln.points) >= 12.0 or len(ln.points) >= 3)
        and sum(_pt_in_box(p[0], p[1], box, 2.0) for p in ln.points) >= 0.5 * len(ln.points)
        for ln in marks.lines
    ):
        return True
    return any(_pt_in_box(x, y, box, 2.0) and not _is_legend_marker(x, y, words) for x, y, _ in marks.markers)


def _build_zones(
    axes: list[Axis],
    marks: Marks,
    words: Sequence[PtWord],
    bounds: BBox,
) -> list[_PlotZone]:
    """Build non-overlapping plot zones pairing value axes with their mark regions and category axes."""
    bx0, by0, bx1, by1 = bounds
    v_indexed = [(i, ax) for i, ax in enumerate(axes) if ax.orient == "v"]
    h_indexed = [(i, ax) for i, ax in enumerate(axes) if ax.orient == "h"]

    # First construct vertical-axis zones.
    raw_v_zones: list[_PlotZone] = []
    for idx, ax in v_indexed:
        lo_y, hi_y = ax.extent
        zero = ax.coord(0.0) if _zero_in_reach(ax) else None
        if zero is not None:
            lo_y, hi_y = min(lo_y, zero), max(hi_y, zero)
        step_y = (hi_y - lo_y) / max(1, len(ax.coords) - 1)
        pad_y = 0.9 * step_y

        same_row = [
            o
            for _, o in v_indexed
            if o is not ax and _overlap_1d(o.extent[0], o.extent[1], lo_y, hi_y) > 0.3 * max(hi_y - lo_y, 1.0)
        ]
        before = max((o.hi for o in same_row if o.hi <= ax.lo), default=bx0)
        after = min((o.lo for o in same_row if o.lo >= ax.hi), default=bx1)

        # Candidate sides: right of labels ("left" axis) and left of labels ("right" axis).
        right_end = min(after, ax.rules[1] + 4.0) if (ax.rules and ax.rules[1] > ax.hi + 2.0) else after
        left_start = max(before, ax.rules[0] - 4.0) if (ax.rules and ax.rules[0] < ax.lo - 2.0) else before

        candidates: list[tuple[str, BBox, BBox]] = []
        if right_end - ax.hi > 8.0:
            candidates.append(("left", (ax.hi, lo_y, right_end, hi_y), (ax.hi, lo_y - pad_y, right_end, hi_y + pad_y)))
        if ax.lo - left_start > 8.0:
            candidates.append(
                ("right", (left_start, lo_y, ax.lo, hi_y), (left_start, lo_y - pad_y, ax.lo, hi_y + pad_y))
            )

        for side, plot_box, search_box in candidates:
            if _has_marks_in_box(marks, search_box, words):
                raw_v_zones.append(_PlotZone(idx, ax, plot_box, search_box, side=side))

    # Drop redundant "right"-axis zones that overlap a "left"-axis zone of the same or twin plot.
    left_axes = {z.axis_index for z in raw_v_zones if z.side == "left"}
    v_zones: list[_PlotZone] = []
    for z in raw_v_zones:
        if z.side == "right":
            overlapping_left = [
                o
                for o in raw_v_zones
                if o.side == "left"
                and o.axis_index != z.axis_index
                and _overlap_1d(o.plot_box[0], o.plot_box[2], z.plot_box[0], z.plot_box[2])
                >= 0.5 * min(o.plot_box[2] - o.plot_box[0], z.plot_box[2] - z.plot_box[0])
                and _overlap_1d(o.plot_box[1], o.plot_box[3], z.plot_box[1], z.plot_box[3])
                >= 0.5 * min(o.plot_box[3] - o.plot_box[1], z.plot_box[3] - z.plot_box[1])
            ]
            # Small multiples: axis owns a panel to its right -> don't also read the panel on its left.
            if z.axis_index in left_axes and overlapping_left:
                continue
            # Mirrored twin axis or dual axis already covered by a left axis: avoid duplicating every mark.
            if overlapping_left and any(_same_scale(z.axis, o.axis) for o in overlapping_left):
                continue
            if overlapping_left:
                continue
        v_zones.append(z)

    # Pair horizontal axes as x-axes under (or above) vertical zones.
    used_h_axes: set[int] = set()
    for z in v_zones:
        px0, py0, px1, py1 = z.plot_box
        pad_y = max(12.0, z.search_box[3] - py1)
        below_axes: list[tuple[float, int, Axis]] = []
        above_axes: list[tuple[float, int, Axis]] = []
        for h_idx, h_ax in h_indexed:
            if h_ax.log:
                continue
            h_span = max(h_ax.extent[1] - h_ax.extent[0], 1e-6)
            ov = _overlap_1d(h_ax.extent[0], h_ax.extent[1], px0 - 12.0, px1 + 12.0)
            if ov < 0.5 * h_span:
                continue
            if py1 - 2.0 * pad_y <= h_ax.pos <= py1 + 48.0:
                below_axes.append((abs(h_ax.pos - py1), h_idx, h_ax))
            elif py0 - 48.0 <= h_ax.pos <= py0 + 2.0 * pad_y:
                above_axes.append((abs(py0 - h_ax.pos), h_idx, h_ax))
        chosen = below_axes or above_axes
        if chosen:
            # Keep axes on the closest row (handles multiple side-by-side sub-group x-axes under one panel).
            min_d = min(d for d, _, _ in chosen)
            row_axes = [
                (h_idx, h_ax) for d, h_idx, h_ax in chosen if d <= min_d + 12.0
            ]
            row_axes.sort(key=lambda pair: pair[1].extent[0])
            z.cat_axes = tuple(h_ax for _, h_ax in row_axes)
            used_h_axes.update(h_idx for h_idx, _ in row_axes)
        # Also mark any other horizontal axis sitting directly under this vertical plot as x-axis.
        for h_idx, h_ax in h_indexed:
            h_span = max(h_ax.extent[1] - h_ax.extent[0], 1e-6)
            if (
                _overlap_1d(h_ax.extent[0], h_ax.extent[1], px0 - 12.0, px1 + 12.0) >= 0.5 * h_span
                and py1 - 10.0 <= h_ax.pos <= py1 + 48.0
            ):
                used_h_axes.add(h_idx)

    # Build horizontal-value-axis zones for remaining horizontal axes (horizontal bar/marker charts).
    h_zones: list[_PlotZone] = []
    for idx, ax in h_indexed:
        if idx in used_h_axes:
            continue
        lo_x, hi_x = ax.extent
        zero = ax.coord(0.0) if _zero_in_reach(ax) else None
        if zero is not None:
            lo_x, hi_x = min(lo_x, zero), max(hi_x, zero)
        step_x = (hi_x - lo_x) / max(1, len(ax.coords) - 1)
        pad_x = 0.9 * step_x

        same_col = [
            o
            for o_idx, o in h_indexed
            if o_idx != idx and _overlap_1d(o.extent[0], o.extent[1], lo_x, hi_x) > 0.3 * max(hi_x - lo_x, 1.0)
        ]
        above_lim = max((o.hi for o in same_col if o.hi <= ax.lo), default=by0)
        below_lim = min((o.lo for o in same_col if o.lo >= ax.hi), default=by1)

        top_start = max(above_lim, ax.rules[0] - 4.0) if (ax.rules and ax.rules[0] < ax.lo - 2.0) else above_lim
        bot_end = min(below_lim, ax.rules[1] + 4.0) if (ax.rules and ax.rules[1] > ax.hi + 2.0) else below_lim

        cands: list[tuple[BBox, BBox]] = []
        if ax.lo - top_start > 8.0:
            cands.append(((lo_x, top_start, hi_x, ax.lo), (lo_x - pad_x, top_start, hi_x + pad_x, ax.lo)))
        if bot_end - ax.hi > 8.0:
            cands.append(((lo_x, ax.hi, hi_x, bot_end), (lo_x - pad_x, ax.hi, hi_x + pad_x, bot_end)))

        for plot_box, search_box in cands:
            if _has_marks_in_box(marks, search_box, words):
                h_zones.append(_PlotZone(idx, ax, plot_box, search_box))
                break  # avoid double-reading a horizontal axis on both top and bottom

    return v_zones + h_zones


# --------------------------------------------------------------------------- category label inference


def _cat_axis_value(cat_axes: tuple[Axis, ...], coord: float) -> float:
    """Evaluate a coordinate against the matching paired numeric category axis."""
    if len(cat_axes) == 1:
        ax = cat_axes[0]
        return _round_axis_value(ax, ax.value(coord))

    def dist(ax: Axis) -> float:
        lo, hi = ax.extent
        if lo <= coord <= hi:
            return 0.0
        return min(abs(coord - lo), abs(coord - hi))

    ax = min(cat_axes, key=dist)
    return _round_axis_value(ax, ax.value(coord))


def _slot_category_labels(
    slot_centers: list[float],
    z: _PlotZone,
    words: Sequence[PtWord],
    tick_word_ids: set[int],
) -> list[str | float | int]:
    """Attach a category label (from paired numeric axis, nearby text labels, or 1-based slot index)."""
    if z.cat_axes:
        return [_cat_axis_value(z.cat_axes, c) for c in slot_centers]

    n = len(slot_centers)
    if n == 0:
        return []
    gaps = [slot_centers[i + 1] - slot_centers[i] for i in range(n - 1) if slot_centers[i + 1] > slot_centers[i]]
    half = 0.48 * min(gaps) if gaps else 25.0
    px0, py0, px1, py1 = z.plot_box

    if z.axis.orient == "v":
        pool = [
            w
            for w in words
            if id(w) not in tick_word_ids
            and py1 - 4.0 <= w.y0 <= py1 + 36.0
            and px0 - max(20.0, half) <= w.cx <= px1 + max(20.0, half)
        ]
        if not pool:
            return [i + 1 for i in range(n)]
        # Keep words near the top label band (skip captions / source lines lower down).
        min_y = min(w.y0 for w in pool)
        band_h = max(median(w.y1 - w.y0 for w in pool), 6.0)
        pool = [w for w in pool if w.y0 <= min_y + max(2.5 * band_h, 18.0)]
        # Exclude wide prose/title words spanning across multiple slots.
        pool = [w for w in pool if (w.x1 - w.x0) <= max(2.2 * half, 40.0)]

        assigned: list[list[PtWord]] = [[] for _ in range(n)]
        for w in pool:
            best_i = min(range(n), key=lambda i: abs(w.cx - slot_centers[i]))
            if abs(w.cx - slot_centers[best_i]) <= max(half, 12.0):
                assigned[best_i].append(w)

        labels: list[str] = []
        for ws in assigned:
            if not ws:
                labels.append("")
                continue
            lns = lines_of(ws)
            txt = " ".join(" ".join(w.text for w in ln) for ln in lns).strip()
            labels.append(txt[:40] if len(txt) <= 40 else "")
        if sum(bool(lb) for lb in labels) >= max(1, math.ceil(0.5 * n)):
            return [lb if lb else (i + 1) for i, lb in enumerate(labels)]
        return [i + 1 for i in range(n)]

    # Horizontal chart: category labels sit left of the plot aligned with each bar's y-center.
    pool = [
        w
        for w in words
        if id(w) not in tick_word_ids
        and w.x1 <= px0 + 6.0
        and w.x0 >= px0 - 200.0
        and py0 - 10.0 <= w.cy <= py1 + 10.0
    ]
    if not pool:
        return [i + 1 for i in range(n)]
    assigned = [[] for _ in range(n)]
    tol = min(max(half, 6.0), 14.0)
    for w in pool:
        best_i = min(range(n), key=lambda i: abs(w.cy - slot_centers[i]))
        if abs(w.cy - slot_centers[best_i]) <= tol:
            assigned[best_i].append(w)
    labels = [" ".join(w.text for w in sorted(ws, key=lambda w: w.x0)).strip() for ws in assigned]
    if sum(bool(lb) for lb in labels) >= max(1, math.ceil(0.5 * n)):
        return [lb[:40] if lb else (i + 1) for i, lb in enumerate(labels)]
    return [i + 1 for i in range(n)]


# --------------------------------------------------------------------------- bar / line / marker extraction


def _dedupe_bars(bars: list[Bar]) -> list[Bar]:
    seen: set[tuple[Color, float, float, float, float]] = set()
    out: list[Bar] = []
    for b in bars:
        key = (b.color, round(b.x0, 1), round(b.y0, 1), round(b.x1, 1), round(b.y1, 1))
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


def _clean_zone_bars(z: _PlotZone, bars: Sequence[Bar], words: Sequence[PtWord]) -> list[Bar]:
    vertical = z.axis.orient == "v"
    px0, py0, px1, py1 = z.plot_box
    pw = max(px1 - px0, 1.0)
    ph = max(py1 - py0, 1.0)
    cat_span = pw if vertical else ph

    cand: list[Bar] = []
    for b in bars:
        if _is_swatch(b, words):
            continue
        if not _pt_in_box(b.cx, b.cy, z.search_box, 2.0):
            continue
        bw, bh = b.x1 - b.x0, b.y1 - b.y0
        if bw > 0.75 * pw and bh > 0.75 * ph:
            continue  # plot background fill
        thick = bw if vertical else bh
        length = bh if vertical else bw
        if thick < 1.5:
            continue  # thin line/tick
        if length <= 1.2 and thick >= 20.0 * max(length, 0.1):
            continue  # thin divider rule drawn as a filled rect
        if thick > 0.65 * cat_span and len(bars) > 2:
            continue
        cand.append(b)

    cand = _dedupe_bars(cand)
    if not cand:
        return []

    # Reject floating marker-sized squares (<= 9x9pt) that do not sit on the baseline or stack onto another bar.
    zero_coord = z.axis.coord(0.0) if _zero_in_reach(z.axis) else None
    base_coord = (
        zero_coord
        if zero_coord is not None
        else (max(z.axis.coords) if vertical else min(z.axis.coords))
    )
    filtered: list[Bar] = []
    for b in cand:
        bw, bh = b.x1 - b.x0, b.y1 - b.y0
        is_marker_sq = bw <= 9.0 and bh <= 9.0 and 0.6 <= bw / max(bh, 1e-6) <= 1.6
        if is_marker_sq:
            if vertical:
                on_base = min(abs(b.y0 - base_coord), abs(b.y1 - base_coord)) <= 3.0
                touches_other = any(
                    o is not b
                    and _overlap_1d(b.x0, b.x1, o.x0, o.x1) > 0.4 * min(bw, o.x1 - o.x0)
                    and min(abs(b.y0 - o.y1), abs(b.y1 - o.y0)) <= 1.5
                    for o in cand
                )
            else:
                on_base = min(abs(b.x0 - base_coord), abs(b.x1 - base_coord)) <= 3.0
                touches_other = any(
                    o is not b
                    and _overlap_1d(b.y0, b.y1, o.y0, o.y1) > 0.4 * min(bh, o.y1 - o.y0)
                    and min(abs(b.x0 - o.x1), abs(b.x1 - o.x0)) <= 1.5
                    for o in cand
                )
            if not (on_base or touches_other):
                continue
        filtered.append(b)
    cand = filtered
    if not cand:
        return []

    # Per color with >= 3 bars, drop outlier decorative rectangles whose thickness disagrees with the bar mode.
    by_color: dict[Color, list[Bar]] = defaultdict(list)
    for b in cand:
        by_color[b.color].append(b)
    kept: list[Bar] = []
    for group in by_color.values():
        if len(group) < 3:
            kept.extend(group)
            continue
        thicks = [round(((b.x1 - b.x0) if vertical else (b.y1 - b.y0)) * 2.0) / 2.0 for b in group]
        mode = max(set(thicks), key=thicks.count)
        tol = max(1.5, 0.35 * mode)
        near = [b for b, t in zip(group, thicks, strict=True) if abs(t - mode) <= tol]
        kept.extend(near if len(near) >= 2 else group)
    return kept


def _group_bars_into_slots(
    bars: list[Bar], vertical: bool
) -> tuple[list[list[Bar]], list[bool], str]:
    """Group bars into category slots and classify layout ('single', 'grouped', 'stacked', 'grouped_stacked')."""
    if not bars:
        return [], [], "single"

    lo = (lambda b: b.x0) if vertical else (lambda b: b.y0)
    hi = (lambda b: b.x1) if vertical else (lambda b: b.y1)

    # 1. Merge bars that overlap along the category axis into stacks.
    stacks: list[list[Bar]] = []
    for b in sorted(bars, key=lambda b: (lo(b), hi(b))):
        bw = max(hi(b) - lo(b), 1e-6)
        placed = False
        for st in reversed(stacks):
            st_lo = min(lo(x) for x in st)
            st_hi = max(hi(x) for x in st)
            ov = _overlap_1d(lo(b), hi(b), st_lo, st_hi)
            if ov > 0.4 * min(bw, max(st_hi - st_lo, 1e-6)):
                st.append(b)
                placed = True
                break
        if not placed:
            stacks.append([b])

    has_stacked = any(len(st) >= 2 for st in stacks)

    # 2. Detect side-by-side grouped clusters when multiple colors exist and inter-group gaps stand out.
    distinct_colors = {b.color for b in bars}
    grouped = False
    slots: list[list[Bar]] = stacks

    if len(stacks) >= 2 and len(distinct_colors) >= 2:
        st_bounds = [(min(lo(b) for b in st), max(hi(b) for b in st)) for st in stacks]
        st_widths = [b - a for a, b in st_bounds]
        med_w = median(st_widths)
        gaps = [st_bounds[i + 1][0] - st_bounds[i][1] for i in range(len(stacks) - 1)]
        if gaps:
            sorted_gaps = sorted(gaps)
            best_split: tuple[float, float] | None = None
            for i in range(len(sorted_gaps) - 1):
                small, big = sorted_gaps[i], sorted_gaps[i + 1]
                if small <= 0.9 * med_w and big >= 1.5 * max(small, 0.0) + 1.5:
                    ratio = (big - small) / max(med_w, 1.0)
                    if best_split is None or ratio > best_split[0]:
                        best_split = (ratio, (small + big) / 2.0)
            if best_split is not None:
                thresh = best_split[1]
                clusters: list[list[list[Bar]]] = [[stacks[0]]]
                for st, g in zip(stacks[1:], gaps, strict=True):
                    if g >= thresh:
                        clusters.append([st])
                    else:
                        clusters[-1].append(st)
                # Confirm it's genuinely grouped: multiple clusters, >=2 stacks in some cluster,
                # and stacks within a cluster have disjoint color sets (not repeated stacked categories).
                if len(clusters) >= 2 and any(len(c) >= 2 for c in clusters):
                    disjoint = all(
                        len({b.color for st in c for b in st}) == sum(len({b.color for b in st}) for st in c)
                        for c in clusters
                    )
                    if disjoint:
                        grouped = True
                        slots = [[b for st in c for b in st] for c in clusters]
            elif len(stacks) == len(distinct_colors) and max(gaps) <= 0.6 * med_w:
                # Single category slot with multiple side-by-side colored bars.
                if not has_stacked:
                    grouped = True
                    slots = [[b for st in stacks for b in st]]

    slot_is_stacked: list[bool] = []
    for slot in slots:
        stacked_here = False
        for i, a in enumerate(slot):
            aw = max(hi(a) - lo(a), 1e-6)
            for b in slot[i + 1 :]:
                bw = max(hi(b) - lo(b), 1e-6)
                if _overlap_1d(lo(a), hi(a), lo(b), hi(b)) > 0.4 * min(aw, bw):
                    stacked_here = True
                    break
            if stacked_here:
                break
        slot_is_stacked.append(stacked_here)

    if has_stacked and grouped:
        layout = "grouped_stacked"
    elif has_stacked:
        layout = "stacked"
    elif grouped:
        layout = "grouped"
    else:
        layout = "single"
    return slots, slot_is_stacked, layout


def _measure_bar(b: Bar, ax: Axis, stacked: bool) -> dict[str, Any]:
    if ax.orient == "v":
        v_lo, v_hi = ax.value(b.y1), ax.value(b.y0)
    else:
        v_lo, v_hi = ax.value(b.x0), ax.value(b.x1)
    if v_lo > v_hi:
        v_lo, v_hi = v_hi, v_lo

    span = max(abs(ax.span), 1e-6)
    from_zero = _zero_in_reach(ax)

    if stacked:
        if v_hi <= 0.01 * span and v_lo < 0.0:
            seg = -(v_hi - v_lo)
        else:
            seg = v_hi - v_lo
        return {
            "value": _round_axis_value(ax, seg),
            "base": _round_axis_value(ax, v_lo),
            "top": _round_axis_value(ax, v_hi),
        }

    if from_zero:
        near, far = (v_lo, v_hi) if abs(v_lo) <= abs(v_hi) else (v_hi, v_lo)
        if abs(near) <= 0.05 * span:
            return {"value": _round_axis_value(ax, far)}
        return {
            "value": _round_axis_value(ax, far - near),
            "base": _round_axis_value(ax, v_lo),
            "top": _round_axis_value(ax, v_hi),
        }

    # Truncated non-zero axis: the bar's reading is its far (top/right) end.
    return {"value": _round_axis_value(ax, v_hi)}


def _order_colors(color_order: list[Color], legend_map: dict[Color, str]) -> list[Color]:
    legend_rank = {c: i for i, c in enumerate(legend_map)}
    seen_rank = {c: i for i, c in enumerate(color_order)}
    return sorted(color_order, key=lambda c: (0, legend_rank[c]) if c in legend_rank else (1, seen_rank[c]))


def _extract_bar_series(
    z: _PlotZone,
    bars: Sequence[Bar],
    words: Sequence[PtWord],
    legend_map: dict[Color, str],
    tick_word_ids: set[int],
) -> tuple[list[dict[str, Any]], list[Bar]]:
    clean = _clean_zone_bars(z, bars, words)
    if not clean:
        return [], []
    vertical = z.axis.orient == "v"
    slots, slot_is_stacked, layout = _group_bars_into_slots(clean, vertical)
    if not slots:
        return [], []

    lo = (lambda b: b.x0) if vertical else (lambda b: b.y0)
    hi = (lambda b: b.x1) if vertical else (lambda b: b.y1)
    slot_centers = [(min(lo(b) for b in sl) + max(hi(b) for b in sl)) / 2.0 for sl in slots]
    cat_labels = _slot_category_labels(slot_centers, z, words, tick_word_ids)

    by_color: dict[Color, list[dict[str, Any]]] = defaultdict(list)
    color_order: list[Color] = []
    for slot_idx, (slot, is_st) in enumerate(zip(slots, slot_is_stacked, strict=True)):
        # Order bars in a stacked slot from bottom to top (or left to right).
        ordered_slot = sorted(slot, key=(lambda b: -b.cy) if vertical else (lambda b: b.cx))
        for b in ordered_slot:
            if b.color not in by_color:
                color_order.append(b.color)
            m = _measure_bar(b, z.axis, is_st)
            pt: dict[str, Any] = {"slot": slot_idx, "x": cat_labels[slot_idx], **m}
            by_color[b.color].append(pt)

    # Order colors: legend colors first (in legend order), then remaining in appearance order.
    color_order = _order_colors(color_order, legend_map)

    series: list[dict[str, Any]] = []
    for color in color_order:
        pts = by_color[color]
        total_pts = len(pts)
        pts = _cap_points(pts, MAX_POINTS_PER_SERIES)
        entry: dict[str, Any] = {
            "type": "bar",
            "layout": layout,
            "orientation": "vertical" if vertical else "horizontal",
            "color": _color_hex(color),
            "rgb": list(color),
            "label": legend_map.get(color, ""),
            "axis_index": z.axis_index,
            "points": pts,
        }
        if total_pts > len(pts):
            entry["total_points"] = total_pts
        series.append(entry)
    return series, clean


def _cap_points(points: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    n = len(points)
    if n <= cap:
        return points
    if cap <= 1:
        return points[:cap]
    idxs = [round(i * (n - 1) / (cap - 1)) for i in range(cap)]
    return [points[i] for i in idxs]


def _is_l_frame(pts: list[tuple[float, float]], plot_box: BBox) -> bool:
    """Detect a 3-point L-shaped plot border (1 vertical + 1 horizontal edge on the plot frame)."""
    if len(pts) != 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = pts
    seg1_v = abs(x0 - x1) <= 0.5 and abs(y0 - y1) > 10.0
    seg1_h = abs(y0 - y1) <= 0.5 and abs(x0 - x1) > 10.0
    seg2_v = abs(x1 - x2) <= 0.5 and abs(y1 - y2) > 10.0
    seg2_h = abs(y1 - y2) <= 0.5 and abs(x1 - x2) > 10.0
    if not ((seg1_v and seg2_h) or (seg1_h and seg2_v)):
        return False
    bx0, by0, bx1, by1 = plot_box
    on_edge = (
        min(abs(x1 - bx0), abs(x1 - bx1)) <= 8.0 and min(abs(y1 - by0), abs(y1 - by1)) <= 8.0
    )
    return on_edge


def _dedupe_xy(pts: list[tuple[float, float]], tol: float = 0.25) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in pts:
        if out and abs(p[0] - out[-1][0]) <= tol and abs(p[1] - out[-1][1]) <= tol:
            continue
        out.append(p)
    return out


def _extract_line_series(
    z: _PlotZone,
    lines: Sequence[Line],
    words: Sequence[PtWord],
    legend_map: dict[Color, str],
    tick_word_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[Color, list[tuple[float, float]]]]:
    vertical = z.axis.orient == "v"
    by_color: dict[Color, list[tuple[float, float]]] = defaultdict(list)
    color_order: list[Color] = []

    for ln in lines:
        inside = [p for p in ln.points if _pt_in_box(p[0], p[1], z.search_box, 2.0)]
        if len(inside) < 2 or len(inside) < 0.5 * len(ln.points):
            continue
        xs = [p[0] for p in inside]
        ys = [p[1] for p in inside]
        if len(inside) == 2 and (max(xs) - min(xs) < 12.0 and max(ys) - min(ys) < 12.0):
            continue  # short callout/leader tick, not a series
        if _is_l_frame(inside, z.plot_box):
            continue
        if ln.color not in by_color:
            color_order.append(ln.color)
        by_color[ln.color].extend(inside)

    color_order = _order_colors(color_order, legend_map)

    series: list[dict[str, Any]] = []
    line_pts_by_color: dict[Color, list[tuple[float, float]]] = {}
    for color in color_order:
        pts = sorted(by_color[color], key=(lambda p: (p[0], p[1])) if vertical else (lambda p: (p[1], p[0])))
        pts = _dedupe_xy(pts, tol=0.25)
        if len(pts) < 2:
            continue
        line_pts_by_color[color] = pts
        total_pts = len(pts)
        sampled = _cap_points([{"_p": p} for p in pts], MAX_POINTS_PER_SERIES)
        coords = [item["_p"] for item in sampled]
        cat_positions = [p[0] if vertical else p[1] for p in coords]
        cat_labels = (
            [_cat_axis_value(z.cat_axes, c) for c in cat_positions]
            if z.cat_axes
            else (
                _slot_category_labels(cat_positions, z, words, tick_word_ids)
                if len(cat_positions) <= 20
                else [i + 1 for i in range(len(cat_positions))]
            )
        )
        measured_pts = [
            {
                "x": cat_labels[i],
                "value": _round_axis_value(z.axis, z.axis.value(p[1] if vertical else p[0])),
            }
            for i, p in enumerate(coords)
        ]
        entry: dict[str, Any] = {
            "type": "line",
            "orientation": "vertical" if vertical else "horizontal",
            "color": _color_hex(color),
            "rgb": list(color),
            "label": legend_map.get(color, ""),
            "axis_index": z.axis_index,
            "points": measured_pts,
        }
        if total_pts > len(measured_pts):
            entry["total_points"] = total_pts
        series.append(entry)
    return series, line_pts_by_color


def _point_near_polyline(pt: tuple[float, float], poly: list[tuple[float, float]], tol: float = 2.0) -> bool:
    px, py = pt
    for x, y in poly:
        if abs(px - x) <= tol and abs(py - y) <= tol:
            return True
    return False


def _extract_marker_series(
    z: _PlotZone,
    markers: Sequence[tuple[float, float, Color]],
    words: Sequence[PtWord],
    legend_map: dict[Color, str],
    tick_word_ids: set[int],
    used_bars: Sequence[Bar],
    line_pts_by_color: dict[Color, list[tuple[float, float]]],
) -> list[dict[str, Any]]:
    vertical = z.axis.orient == "v"
    by_color: dict[Color, list[tuple[float, float]]] = defaultdict(list)
    color_order: list[Color] = []

    bars_by_color: dict[Color, list[Bar]] = defaultdict(list)
    for b in used_bars:
        bars_by_color[b.color].append(b)

    for x, y, color in markers:
        if not _pt_in_box(x, y, z.search_box, 1.5):
            continue
        if _is_legend_marker(x, y, words):
            continue
        # Skip markers that are merely small bar rectangles already reported as bars.
        if any(b.x0 - 0.5 <= x <= b.x1 + 0.5 and b.y0 - 0.5 <= y <= b.y1 + 0.5 for b in bars_by_color.get(color, ())):
            continue
        if color not in by_color:
            color_order.append(color)
        by_color[color].append((x, y))

    color_order = _order_colors(color_order, legend_map)

    series: list[dict[str, Any]] = []
    for color in color_order:
        pts = sorted(by_color[color], key=(lambda p: (p[0], p[1])) if vertical else (lambda p: (p[1], p[0])))
        pts = _dedupe_xy(pts, tol=0.5)
        if len(pts) < 2:
            continue
        # Skip marker series that duplicate a line series of the exact same colour.
        poly = line_pts_by_color.get(color)
        if poly and sum(_point_near_polyline(p, poly) for p in pts) >= 0.7 * len(pts):
            continue
        total_pts = len(pts)
        sampled = _cap_points([{"_p": p} for p in pts], MAX_POINTS_PER_SERIES)
        coords = [item["_p"] for item in sampled]
        cat_positions = [p[0] if vertical else p[1] for p in coords]
        cat_labels = (
            [_cat_axis_value(z.cat_axes, c) for c in cat_positions]
            if z.cat_axes
            else (
                _slot_category_labels(cat_positions, z, words, tick_word_ids)
                if len(cat_positions) <= 20
                else [i + 1 for i in range(len(cat_positions))]
            )
        )
        measured_pts = [
            {
                "x": cat_labels[i],
                "value": _round_axis_value(z.axis, z.axis.value(p[1] if vertical else p[0])),
            }
            for i, p in enumerate(coords)
        ]
        entry: dict[str, Any] = {
            "type": "marker",
            "orientation": "vertical" if vertical else "horizontal",
            "color": _color_hex(color),
            "rgb": list(color),
            "label": legend_map.get(color, ""),
            "axis_index": z.axis_index,
            "points": measured_pts,
        }
        if total_pts > len(measured_pts):
            entry["total_points"] = total_pts
        series.append(entry)
    return series


# --------------------------------------------------------------------------- printed labels & nearby text


def _group_label_phrases(words: Sequence[PtWord]) -> list[list[PtWord]]:
    """Group horizontally adjacent words on the same baseline into label phrases."""
    phrases: list[list[PtWord]] = []
    for line in lines_of(list(words)):
        if not line:
            continue
        cur = [line[0]]
        for w in line[1:]:
            prev = cur[-1]
            h = max(prev.y1 - prev.y0, w.y1 - w.y0, 4.0)
            if 0.0 <= w.x0 - prev.x1 <= 0.75 * h and abs(w.cy - prev.cy) <= 3.0:
                cur.append(w)
            else:
                phrases.append(cur)
                cur = [w]
        phrases.append(cur)
    return phrases


def _extract_printed_labels(
    words: Sequence[PtWord],
    plot_boxes: list[BBox],
    tick_word_ids: set[int],
    legend_labels: set[str],
    marks: Marks,
) -> tuple[list[dict[str, Any]], set[int]]:
    if not plot_boxes:
        return [], set()

    in_plot: list[PtWord] = []
    seen_ids: set[int] = set()
    for w in words:
        if id(w) in tick_word_ids or id(w) in seen_ids:
            continue
        if any(_pt_in_box(w.cx, w.cy, box) for box in plot_boxes):
            in_plot.append(w)
            seen_ids.add(id(w))

    # Swatch / key right edges for excluding inset legend text.
    key_boxes: list[tuple[float, float, float, float]] = [
        (b.x0, b.y0, b.x1, b.y1) for b in marks.bars if b.x1 - b.x0 <= 16.0 and b.y1 - b.y0 <= 16.0
    ]
    key_boxes += [(k[0], k[1], k[2], k[3]) for k in marks.keys]
    key_boxes += [(x - 2.0, y - 2.0, x + 2.0, y + 2.0) for x, y, _ in marks.markers]

    labels: list[dict[str, Any]] = []
    numeric_label_word_ids: set[int] = set()
    for phrase in _group_label_phrases(in_plot):
        text = " ".join(w.text for w in phrase).strip()
        if not text:
            continue
        x0 = min(w.x0 for w in phrase)
        x1 = max(w.x1 for w in phrase)
        y0 = min(w.y0 for w in phrase)
        y1 = max(w.y1 for w in phrase)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # Skip inset legend label sitting right beside a legend key.
        if text in legend_labels and any(
            0.0 <= x0 - kx1 <= 16.0 and abs(cy - (ky0 + ky1) / 2.0) <= max(ky1 - ky0, 5.0)
            for _kx0, ky0, kx1, ky1 in key_boxes
        ):
            continue
        val = tick_value(text.strip(".,;:"))
        if val is not None:
            numeric_label_word_ids.update(id(w) for w in phrase)
        labels.append(
            {
                "text": text,
                "value": _clean_float(val) if val is not None else None,
                "x": round(cx, 1),
                "y": round(cy, 1),
            }
        )

    labels.sort(key=lambda item: (round(item["y"] / 6.0), item["x"], item["text"]))
    return labels[:MAX_PRINTED_LABELS], numeric_label_word_ids


def _extract_nearby_text(
    words: Sequence[PtWord],
    region: BBox | None,
    tick_word_ids: set[int],
    numeric_label_word_ids: set[int],
) -> list[str]:
    if region is None:
        return []
    bx0, by0, bx1, by1 = region
    near_box = (bx0 - 24.0, by0 - 60.0, bx1 + 24.0, by1 + 60.0)
    mid_y = (by0 + by1) / 2.0
    selected = [
        w
        for w in words
        if id(w) not in tick_word_ids
        and id(w) not in numeric_label_word_ids
        and _pt_in_box(w.cx, w.cy, near_box)
    ]
    if not selected:
        return []
    line_objs = lines_of(selected)
    # Keep lines closest to the chart region if there are more than MAX_NEARBY_LINES.
    if len(line_objs) > MAX_NEARBY_LINES:
        ranked = sorted(
            enumerate(line_objs),
            key=lambda pair: abs(sum(w.cy for w in pair[1]) / len(pair[1]) - mid_y),
        )[:MAX_NEARBY_LINES]
        line_objs = [ln for _, ln in sorted(ranked, key=lambda pair: pair[0])]

    out: list[str] = []
    for ln in line_objs:
        txt = " ".join(w.text for w in ln).strip()
        if txt:
            out.append(txt)
    return out


# --------------------------------------------------------------------------- public API


def extract_chart_hint(
    words: RenderedPage | Sequence[PtWord],
    shapes: Sequence[Shape],
    bbox: BBox | None = None,
    *,
    page_size: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Extract structured vector chart hints from PDF words and shapes.

    Parameters
    ----------
    words:
        Either a `RenderedPage` (converted via `words_in_points`) or a sequence of `PtWord` in PDF points.
    shapes:
        Vector paths from `docparse.pdf.graphics.load_shapes`.
    bbox:
        Optional `(x0, y0, x1, y1)` chart region in PDF points (origin top-left). When omitted, the
        whole page is inspected.
    page_size:
        Optional `(width_pt, height_pt)` override when `words` is a sequence of `PtWord`.
    """
    if isinstance(words, RenderedPage):
        pt_words = words_in_points(words)
        page_w, page_h = page_size if page_size is not None else (words.width_pt, words.height_pt)
    else:
        pt_words = list(words)
        page_w, page_h = page_size if page_size is not None else _infer_page_size(pt_words, shapes, bbox)

    norm_box = _norm_bbox(bbox) if bbox is not None else None
    empty_hint: dict[str, Any] = {
        "axes": [],
        "legend": [],
        "series": [],
        "printed_labels": [],
        "nearby_text": [],
    }
    if not pt_words and not shapes:
        return empty_hint

    page_marks = find_marks(list(shapes), page_w, page_h) if shapes else Marks()

    # 1. Detect value axes (inside/touching bbox when given) and snap to gridlines/ticks.
    axis_words = (
        [w for w in pt_words if _pt_in_box(w.cx, w.cy, norm_box, 20.0)]
        if norm_box is not None
        else pt_words
    )
    raw_axes = find_axes(axis_words, lost=True) if axis_words else []

    snapped_axes: list[Axis] = []
    axes_dicts: list[dict[str, Any]] = []
    tick_word_ids: set[int] = set()

    axis_entries: list[tuple[Axis, list[tuple[str, float]], set[int]]] = []
    for raw_ax in raw_axes:
        snapped = snap_axis(raw_ax, page_marks)
        if norm_box is not None and not _axis_touches_bbox(snapped, norm_box):
            continue
        ticks, used_ids = _extract_axis_ticks(raw_ax, axis_words)
        axis_entries.append((snapped, ticks, used_ids))

    # Deterministic order: vertical axes first (row-banded top-to-bottom, then left-to-right), then horizontal.
    axis_entries.sort(
        key=lambda e: (
            0 if e[0].orient == "v" else 1,
            round(e[0].extent[0] / 40.0) if e[0].orient == "v" else round(e[0].pos / 40.0),
            round(e[0].pos, 1) if e[0].orient == "v" else round(e[0].extent[0], 1),
        )
    )

    for snapped, ticks, used_ids in axis_entries:
        snapped_axes.append(snapped)
        tick_word_ids.update(used_ids)
        axes_dicts.append(
            {
                "orientation": "vertical" if snapped.orient == "v" else "horizontal",
                "log": snapped.log,
                "ticks": ticks,
                "range": [_clean_float(min(snapped.values)), _clean_float(max(snapped.values))],
                "step": None if snapped.log else _clean_float(snapped.step),
                "position": round(snapped.pos, 1),
                "extent": [round(snapped.extent[0], 1), round(snapped.extent[1], 1)],
                "snapped": snapped.rules is not None,
            }
        )

    # 2. Filter marks to bbox and extract legend inside/near bbox.
    region_marks = _filter_marks(page_marks, norm_box) if norm_box is not None else page_marks
    if norm_box is not None:
        leg_marks = _filter_marks(page_marks, _expand_bbox(norm_box, 36.0), slack=0.0)
        leg_words = [
            w
            for w in pt_words
            if norm_box[0] - 36.0 <= w.cx <= norm_box[2] + 84.0 and norm_box[1] - 36.0 <= w.cy <= norm_box[3] + 36.0
        ]
        legend_map = find_legend(leg_marks, leg_words)
    else:
        legend_map = find_legend(page_marks, pt_words)

    legend_list = [
        {"color": _color_hex(c), "rgb": [c[0], c[1], c[2]], "label": label}
        for c, label in list(legend_map.items())[:MAX_LEGEND_ENTRIES]
    ]

    # 3. Build plot zones and measure bar, line, and marker series.
    bounds: BBox = norm_box if norm_box is not None else (0.0, 0.0, page_w, page_h)
    zones = _build_zones(snapped_axes, region_marks, pt_words, bounds) if snapped_axes else []

    all_series: list[dict[str, Any]] = []
    for z in zones:
        bar_series, used_bars = _extract_bar_series(z, region_marks.bars, pt_words, legend_map, tick_word_ids)
        line_series, line_pts_by_color = _extract_line_series(
            z, region_marks.lines, pt_words, legend_map, tick_word_ids
        )
        marker_series = _extract_marker_series(
            z, region_marks.markers, pt_words, legend_map, tick_word_ids, used_bars, line_pts_by_color
        )
        all_series.extend(bar_series)
        all_series.extend(line_series)
        all_series.extend(marker_series)

    # Prioritize legend-named and multi-point series when capping at MAX_SERIES.
    total_series = len(all_series)
    if total_series > MAX_SERIES:
        ranked = sorted(
            enumerate(all_series),
            key=lambda pair: (0 if pair[1].get("label") else 1, -len(pair[1].get("points", [])), pair[0]),
        )[:MAX_SERIES]
        all_series = [s for _, s in sorted(ranked, key=lambda pair: pair[0])]

    # 4. Printed labels inside the plot frame(s) (excluding axis ticks and inset legend labels).
    plot_boxes: list[BBox] = []
    for z in zones:
        px0, py0, px1, py1 = z.plot_box
        if z.axis.orient == "v":
            plot_boxes.append((px0 - 4.0, py0 - 14.0, px1 + 4.0, py1 + 2.0))
        else:
            plot_boxes.append((px0 - 2.0, py0 - 4.0, px1 + 22.0, py1 + 4.0))

    if not plot_boxes and (axes_dicts or legend_list):
        # Fallback plot frame for axis-less charts (e.g. pie/donut charts with legend) or axes without marks.
        pts_x: list[float] = []
        pts_y: list[float] = []
        for b in region_marks.bars:
            if not _is_swatch(b, pt_words) and (b.x1 - b.x0) < 0.85 * page_w:
                pts_x.extend((b.x0, b.x1))
                pts_y.extend((b.y0, b.y1))
        for ln in region_marks.lines:
            lx0, ly0, lx1, ly1 = ln.bbox
            pts_x.extend((lx0, lx1))
            pts_y.extend((ly0, ly1))
        for x, y, _ in region_marks.markers:
            if not _is_legend_marker(x, y, pt_words):
                pts_x.append(x)
                pts_y.append(y)
        if pts_x and pts_y:
            plot_boxes.append((min(pts_x) - 12.0, min(pts_y) - 12.0, max(pts_x) + 12.0, max(pts_y) + 12.0))
        elif norm_box is not None:
            plot_boxes.append(norm_box)

    legend_labels = set(legend_map.values())
    printed_labels, numeric_label_word_ids = _extract_printed_labels(
        pt_words, plot_boxes, tick_word_ids, legend_labels, region_marks
    )

    # 5. Nearby text (titles, subtitles, category axis labels, units).
    if norm_box is not None:
        text_region: BBox | None = norm_box
    elif plot_boxes or snapped_axes:
        rx0s = [b[0] for b in plot_boxes] + [a.lo if a.orient == "v" else a.extent[0] for a in snapped_axes]
        ry0s = [b[1] for b in plot_boxes] + [a.extent[0] if a.orient == "v" else a.lo for a in snapped_axes]
        rx1s = [b[2] for b in plot_boxes] + [a.hi if a.orient == "v" else a.extent[1] for a in snapped_axes]
        ry1s = [b[3] for b in plot_boxes] + [a.extent[1] if a.orient == "v" else a.hi for a in snapped_axes]
        text_region = (min(rx0s), min(ry0s), max(rx1s), max(ry1s))
    else:
        text_region = None

    nearby_text = (
        _extract_nearby_text(pt_words, text_region, tick_word_ids, numeric_label_word_ids)
        if (axes_dicts or legend_list or all_series)
        else []
    )

    result: dict[str, Any] = {
        "axes": axes_dicts,
        "legend": legend_list,
        "series": all_series,
        "printed_labels": printed_labels,
        "nearby_text": nearby_text,
    }
    if total_series > len(all_series):
        result["omitted_series"] = total_series - len(all_series)
    return result


# --------------------------------------------------------------------------- formatting


def _fmt_point(pt: dict[str, Any], stacked: bool) -> str:
    val_s = _fmt_num(pt["value"]) if isinstance(pt.get("value"), (int, float)) else str(pt.get("value", ""))
    x = pt.get("x")
    x_s = _fmt_num(x) if isinstance(x, (int, float)) else (str(x) if x is not None and x != "" else "")
    prefix = f"{x_s}=" if x_s != "" else ""
    if stacked and "base" in pt and "top" in pt:
        return f"{prefix}{val_s} ({_fmt_num(pt['base'])}..{_fmt_num(pt['top'])})"
    if "base" in pt and "top" in pt:
        return f"{prefix}{val_s} [{_fmt_num(pt['base'])}..{_fmt_num(pt['top'])}]"
    return f"{prefix}{val_s}"


def format_chart_hint(hint: dict[str, Any], max_chars: int = 2500) -> str:
    """Format a structured chart hint dict into a compact, VLM-readable text summary.

    Returns ``""`` when no axes, legend, or measured series are present.
    """
    if not hint or max_chars <= 0:
        return ""
    axes = hint.get("axes") or []
    legend = hint.get("legend") or []
    series = hint.get("series") or []
    printed = hint.get("printed_labels") or []
    nearby = hint.get("nearby_text") or []

    if not (axes or legend or series):
        return ""

    blocks: list[list[str]] = []

    # 1. Axes
    if axes:
        ax_lines = ["Axes:"]
        for i, ax in enumerate(axes, start=1):
            orient = ax.get("orientation", "vertical")
            scale = "log" if ax.get("log") else "linear"
            rng = ax.get("range")
            rng_s = f"{_fmt_num(rng[0])} to {_fmt_num(rng[1])}" if rng and len(rng) == 2 else ""
            step = ax.get("step")
            step_s = f", step {_fmt_num(step)}" if step is not None else ""
            ticks = ax.get("ticks") or []
            tick_labels = [t[0] if isinstance(t, (list, tuple)) else str(t) for t in ticks]
            if len(tick_labels) > 12:
                tick_s = ", ".join(tick_labels[:5] + ["..."] + tick_labels[-5:])
            else:
                tick_s = ", ".join(tick_labels)
            label = f"Axis {i}" if len(axes) > 1 else ("Y-axis" if orient == "vertical" else "X-axis")
            ax_lines.append(f"- {label} ({orient}, {scale}): {rng_s}{step_s}; ticks: [{tick_s}]")
        blocks.append(ax_lines)

    # 2. Legend
    if legend:
        leg_items = [f"{item.get('color', '')} {item.get('label', '')}".strip() for item in legend]
        blocks.append(["Legend: " + "; ".join(leg_items)])

    # 3. Measured series
    if series:
        ser_lines = ["Series (vector-measured):"]
        multi_axis = len({s.get("axis_index") for s in series}) > 1
        for s in series:
            name = s.get("label") or s.get("color", "series")
            color = s.get("color", "")
            kind = s.get("type", "series")
            orient = s.get("orientation", "")
            layout = s.get("layout", "")
            kind_str = f"{orient} {kind}s".strip() if kind == "bar" else kind
            meta_parts = [p for p in (color if name != color else "", kind_str, layout) if p]
            if multi_axis and s.get("axis_index") is not None:
                meta_parts.append(f"axis {int(s['axis_index']) + 1}")
            stacked = layout in ("stacked", "grouped_stacked")
            pts_s = ", ".join(_fmt_point(p, stacked) for p in s.get("points") or [])
            total = s.get("total_points")
            suffix = f" [sampled from {total} pts]" if total else ""
            ser_lines.append(f"- {name} ({', '.join(meta_parts)}): {pts_s}{suffix}")
        omitted = hint.get("omitted_series")
        if omitted:
            ser_lines.append(f"- (+{omitted} more series omitted)")
        blocks.append(ser_lines)

    # 4. Printed in-plot labels
    if printed:
        texts = [item["text"] if isinstance(item, dict) else str(item) for item in printed]
        blocks.append(["Printed labels: " + ", ".join(texts)])

    # 5. Nearby text
    if nearby:
        near_lines = ["Nearby text:"] + [f"- {line}" for line in nearby]
        blocks.append(near_lines)

    # Assemble within max_chars budget, keeping higher-priority sections intact first.
    out_lines: list[str] = ["[Vector Chart Hint]"]
    used = len(out_lines[0])
    for block in blocks:
        if len(block) > 1:
            # Ensure header + at least the start of the first item fit; never emit a dangling header.
            min_first = min(len(block[1]), 24)
            if used + 1 + len(block[0]) + 1 + min_first > max_chars:
                break
        for j, line in enumerate(block):
            needed = 1 + len(line)
            if used + needed <= max_chars:
                out_lines.append(line)
                used += needed
            else:
                rem = max_chars - used - 1
                if j == 0 and len(block) > 1:
                    break
                if rem >= 24:
                    out_lines.append(line[: rem - 3].rstrip(" ,;:") + "...")
                    used = max_chars
                break
        if used >= max_chars:
            break

    text = "\n".join(out_lines)
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 3)] + ("..." if max_chars >= 3 else "")
    return text

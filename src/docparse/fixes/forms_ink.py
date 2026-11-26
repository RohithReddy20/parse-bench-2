"""Check-box states read from the page render, for check boxes drawn as vector squares (not yet wired in).

Where the markdown already carries a `[ ]`/`[x]` mark right before the printed label of a vector check box (the
box immediately left of the label on its line, as `forms_acro._box_left_of` finds it), the mark is set to what the
render shows, only when the ink is unambiguous (`classify_box_ink`): ink <= empty_max is unchecked, checked_min <=
ink <= bullet_max is checked (an X, a tick, a dot); anything else (a faint speck, a solid `■` bullet) keeps the read.
Ink is the share of the box interior, inset past its border, darker than the box's own surround (a ring just outside
its border) by `dark_delta` grey levels, so shaded form bands and yellowed scans calibrate themselves. Comb fields
(>= 3 touching character boxes in a row) and grids of evenly spaced cells are not check boxes.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
from PIL import Image

from docparse.fixes.align import align
from docparse.fixes.forms import apply_edits
from docparse.fixes.forms_acro import _BOX_MARK, _MARK_GAP, _MARK_WINDOW, _UNIT_END, _box_left_of, _pt_words
from docparse.pdf.graphics import load_shapes
from docparse.pdf.render import RenderedPage

Rect = tuple[float, float, float, float]  # (x0, y0, x1, y1), top-down

MIN_SIDE, MAX_SIDE = 4.5, 18.0  # pts
ASPECT = (0.75, 1.33)  # width / height
CORNER_TOL = 0.75  # pts: every point of a box's subpath lies this close to one of its corners
DEDUP = 3.0  # pts between the centres of the two outlines of one double-stroked box
COMB_DY, COMB_GAP, COMB_MIN = 0.4, 0.35, 3  # same line (x height), touching (x width), boxes in a comb field
GRID_COLS, GRID_ROWS, GRID_EVEN = 4, 3, 1.25  # a grid of cells: this many columns and rows, equal steps (+-25%)
MIN_BOXES = 2  # a page needs this many candidate boxes
INSET = 0.22  # of the box's smaller side, cut off each side so its border is never counted
RING_GAP, RING_WIDTH = 3, 4  # px: the background ring starts this far outside the box and is this wide
BG_PERCENTILE, MIN_BG = 85, 110  # surround luminance; a darker surround (a reversed-out box) is not measured
DARK_DELTA = 45  # grey levels below the surround that count as ink
EMPTY_MAX, CHECKED_MIN, BULLET_MAX = 0.025, 0.12, 0.82


def _size(r: Rect) -> tuple[float, float]:
    return r[2] - r[0], r[3] - r[1]


def _centre(r: Rect) -> tuple[float, float]:
    return (r[0] + r[2]) / 2, (r[1] + r[3]) / 2


def _checkbox_sized(r: Rect) -> bool:
    w, h = _size(r)
    return MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE and ASPECT[0] <= w / h <= ASPECT[1]


def _rect_of(points: Sequence[tuple[float, float]]) -> Rect | None:
    """The bounds of a subpath whose every point is one of their corners (an axis-aligned rectangle), else None."""
    if len(points) < 4:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    for x, y in points:
        if min(abs(x - x0), abs(x - x1)) > CORNER_TOL or min(abs(y - y0), abs(y - y1)) > CORNER_TOL:
            return None
    return x0, y0, x1, y1


def vector_checkbox_rects(pdf_bytes: bytes, page_index: int = 0) -> list[Rect]:
    """Check boxes drawn as vector squares (`checkbox_candidates`), top-down points; [] for images, unreadable pages."""
    try:
        shapes = load_shapes(pdf_bytes, page_index)
    except Exception:  # an unreadable page simply has no boxes
        return []
    return checkbox_candidates([r for s in shapes for sub in s.subpaths if (r := _rect_of(sub)) is not None])


def checkbox_candidates(rects: Sequence[Rect]) -> list[Rect]:
    """Check-box-sized squares in reading order; a double stroke kept as its outer outline, combs and grids dropped."""
    boxes: list[Rect] = []
    for r in sorted(filter(_checkbox_sized, rects), key=lambda q: (-(q[2] - q[0]) * (q[3] - q[1]), q[1], q[0])):
        if all(math.dist(_centre(r), _centre(b)) > DEDUP for b in boxes):
            boxes.append(r)
    drop = _comb_members(boxes) | _grid_members(boxes)
    return sorted((b for i, b in enumerate(boxes) if i not in drop), key=lambda b: (b[1], b[0]))


def _touching(a: Rect, b: Rect) -> bool:
    """Neighbouring character boxes of one comb: about the same height, on one line, touching or nearly."""
    (wa, ha), (wb, hb) = _size(a), _size(b)
    h, gap = max(ha, hb), max(a[0], b[0]) - min(a[2], b[2])
    same_line = abs(_centre(a)[1] - _centre(b)[1]) <= COMB_DY * h
    return min(ha, hb) >= 0.8 * h and same_line and gap <= COMB_GAP * min(wa, wb)


def _comb_members(boxes: list[Rect]) -> set[int]:
    """Boxes in a chain of >= COMB_MIN touching squares on one line: a comb field (a date, a postcode, an ID)."""
    group = list(range(len(boxes)))
    for i, j in combinations(range(len(boxes)), 2):
        if _touching(boxes[i], boxes[j]) and group[i] != group[j]:
            gi, gj = group[i], group[j]
            group = [gi if g == gj else g for g in group]
    sizes = Counter(group)
    return {i for i, g in enumerate(group) if sizes[g] >= COMB_MIN}


def _even(values: list[float], n: int) -> bool:
    """At least n distinct positions with equal steps (within GRID_EVEN) between neighbours."""
    v = sorted(values)
    steps = [b - a for a, b in zip(v, v[1:], strict=False)]
    return len(v) >= n and min(steps) > 0 and max(steps) <= GRID_EVEN * min(steps)


def _grid_members(boxes: list[Rect]) -> set[int]:
    """Boxes whose row holds >= GRID_COLS and whose column >= GRID_ROWS evenly spaced boxes: a grid of cells."""
    out: set[int] = set()
    for i, a in enumerate(boxes):
        (ax, ay), (aw, ah) = _centre(a), _size(a)
        row = [_centre(b)[0] for b in boxes if abs(_centre(b)[1] - ay) <= COMB_DY * ah]
        col = [_centre(b)[1] for b in boxes if abs(_centre(b)[0] - ax) <= COMB_DY * aw]
        if _even(row, GRID_COLS) and _even(col, GRID_ROWS):
            out.add(i)
    return out


def box_ink_ratio(image: Image.Image, box_px: Rect, *, dark_delta: int = DARK_DELTA) -> float | None:
    """Share of a box's interior (inset by INSET of its smaller side, at least 2 px) darker than its surround by
    `dark_delta` grey levels. The surround is the BG_PERCENTILE luminance of a RING_WIDTH-pixel ring just outside
    the border. None when the box is off the image, too small to inset, or on a dark surround."""
    x0, y0, x1, y1 = (round(v) for v in box_px)
    side = min(x1 - x0, y1 - y0)
    inset = max(2, round(INSET * side))
    if side - 2 * inset < 3 or x0 < 0 or y0 < 0 or x1 > image.width or y1 > image.height:
        return None
    pad = RING_GAP + RING_WIDTH
    ox0, oy0 = max(0, x0 - pad), max(0, y0 - pad)
    crop = image.crop((ox0, oy0, min(image.width, x1 + pad), min(image.height, y1 + pad))).convert("L")
    gray = np.asarray(crop, dtype=np.int16)
    ys, xs = np.ogrid[oy0 : oy0 + gray.shape[0], ox0 : ox0 + gray.shape[1]]
    ring = gray[(xs < x0 - RING_GAP) | (xs >= x1 + RING_GAP) | (ys < y0 - RING_GAP) | (ys >= y1 + RING_GAP)]
    if ring.size < 2 * RING_WIDTH * side:
        return None  # the box touches the page edge: too little surround to calibrate on
    bg = float(np.percentile(ring, BG_PERCENTILE))
    if bg < MIN_BG:
        return None
    inner = gray[y0 + inset - oy0 : y1 - inset - oy0, x0 + inset - ox0 : x1 - inset - ox0]
    return float(np.count_nonzero(inner < bg - dark_delta)) / inner.size


def classify_box_ink(
    ink_ratio: float, *, empty_max: float = EMPTY_MAX, checked_min: float = CHECKED_MIN, bullet_max: float = BULLET_MAX
) -> bool | None:
    """True (ticked), False (empty) or None (ambiguous: a faint speck, or a solid `■` bullet)."""
    if ink_ratio <= empty_max:
        return False
    return True if checked_min <= ink_ratio <= bullet_max else None


def ink_checkbox_edits(
    markdown: str,
    rendered: RenderedPage,
    pdf_bytes: bytes,
    page_index: int = 0,
    *,
    boxes: Sequence[Rect] | None = None,
    skip_boxes: Sequence[Sequence[float]] | None = None,
    empty_max: float = EMPTY_MAX,
    checked_min: float = CHECKED_MIN,
) -> list[tuple[int, int, str]]:
    """Set the `[ ]`/`[x]` mark right before each aligned printed label (within `_MARK_WINDOW` characters in its
    unit, only spaces or emphasis between) to the ink state of the check box (`boxes` in top-down points, default
    `vector_checkbox_rects`) immediately left of it. Boxes overlapping `skip_boxes` (AcroForm widgets already set)
    and ambiguous boxes keep their marks; pages with < MIN_BOXES boxes or without a text layer are left alone."""
    if not markdown or not rendered.text_words or rendered.image is None:
        return []
    boxes = vector_checkbox_rects(pdf_bytes, page_index) if boxes is None else boxes
    if len(boxes) < MIN_BOXES:
        return []
    s = rendered.scale
    states: list[bool | None] = []
    for b in boxes:
        skipped = any(b[0] < k[2] and k[0] < b[2] and b[1] < k[3] and k[1] < b[3] for k in skip_boxes or ())
        ink = None if skipped else box_ink_ratio(rendered.image, (b[0] / s, b[1] / s, b[2] / s, b[3] / s))
        states.append(None if ink is None else classify_box_ink(ink, empty_max=empty_max, checked_min=checked_min))
    if all(st is None for st in states):
        return []
    words = _pt_words(rendered)
    al = align(markdown, words, reorder=True)
    # token -> PDF word of its first matched character (reversed, so the first character's word wins)
    first = {al.a_tok[k]: al.p_word[p] for k, p in reversed(list(enumerate(al.amatch))) if p is not None}
    units = [0] + [m.end() for m in _UNIT_END.finditer(markdown)]
    widgets = [(*b, False) for b in boxes]  # the shared geometry check reads only the rect
    used: set[int] = set()
    edits: list[tuple[int, int, str]] = []
    for t, w in sorted(first.items()):
        bi = _box_left_of(words[w], widgets, words, used)
        if bi is None:
            continue
        start = al.tokens[t].start
        u0 = units[bisect.bisect_right(units, start) - 1]
        marks = list(_BOX_MARK.finditer(markdown, max(u0, start - _MARK_WINDOW), start))
        if not marks or not _MARK_GAP.fullmatch(markdown, marks[-1].end(), start):
            continue
        used.add(bi)
        mark, state = marks[-1], states[bi]
        if state is not None and bool(mark.group(1)) != state:
            edits.append((mark.start(), mark.end(), "[x]" if state else "[ ]"))
    return edits


def repair_checkbox_ink(
    markdown: str, rendered: RenderedPage, pdf_bytes: bytes, page_index: int = 0, **opts: Any
) -> str:
    """The markdown with its check-box marks set from the render's ink (`ink_checkbox_edits`)."""
    edits = ink_checkbox_edits(markdown, rendered, pdf_bytes, page_index, **opts)
    return apply_edits(markdown, edits) if edits else markdown

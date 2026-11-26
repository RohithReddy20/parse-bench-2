"""Check-box states read from the page render, for check boxes drawn as vector squares (not yet wired in).

Where the markdown already carries a `[ ]`/`[x]` mark beside the printed label of a vector check box, the mark
is set to what the render shows: the mark right before a label whose box sits immediately left of it on its
line (`[ ] Yes`, as `forms_acro._box_left_of` finds it), else the mark right after a label whose box sits
immediately right of it (`ja [ ] nein [x]`). A box labels only the word beside it that it is clearly nearer
(`_box_owner_side`): `Yes [x]   No [ ]` read as `[x] Yes [ ] No` never gives `No` the box of `Yes`, and a box
midway between two words labels neither. Only unambiguous ink counts (`classify_box_ink`): ink <= empty_max
is unchecked; checked_min <= ink <= bullet_max over a contrasted interior (grey-level std >= min_std: dark
strokes on a light ground) is checked (an X, a tick); anything else (a faint speck, a solid `■` bullet, a shaded
or coloured section box) keeps the read. Ink is the share of the box interior, inset past its border, darker
than the box's own surround (a ring just outside its border) by `dark_delta` grey levels, so shaded form bands
and yellowed scans calibrate themselves. Comb fields (>= 3 touching character boxes in a row), grids of evenly
spaced cells and squares around a printed character (a section or field number: `6`, `01`) are not check boxes.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
from PIL import Image

from docparse.fixes.align import align
from docparse.fixes.forms import apply_edits
from docparse.fixes.forms_acro import (
    _BOX_MARK,
    _MARK_GAP,
    _MARK_WINDOW,
    _UNIT_END,
    CheckWidget,
    _box_left_of,
    _pt_words,
)
from docparse.fixes.layout_refine import PtWord
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
EMPTY_MAX, CHECKED_MIN, BULLET_MAX = 0.025, 0.12, 0.58
MIN_STD = 28.0  # grey levels across a ticked interior (strokes on a light ground); a shaded box is flatter
OWNER_GAP = 0.85  # a box labels the word beside it whose gap is at most this share of the other word's gap
# text a check box may hold (a check glyph); any other word centred in a box is a printed field number
_CHECK_GLYPHS = frozenset({"", "x", "X", "×", "✓", "✔", "✗", "✘", "√", "☑", "☒", "☐", "■", "□", "•", "v", "V"})
_MARK_GAP_AFTER = re.compile(r"(?:[ \t*_:]|</?(?:strong|b|em|i|u)>)*")  # what may sit between a word and its mark


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


def _holds_print(b: Rect, words: Sequence[PtWord]) -> bool:
    """A printed word other than a check glyph is centred in box b: a section or field number, not a check box."""
    return any(
        b[0] - 1 < w.cx < b[2] + 1 and b[1] - 1 < w.cy < b[3] + 1 and w.text.strip() not in _CHECK_GLYPHS
        for w in words
    )


def box_ink_ratio(image: Image.Image, box_px: Rect, *, dark_delta: int = DARK_DELTA) -> float | None:
    """Share of a box's interior (inset by INSET of its smaller side, at least 2 px) darker than its surround by
    `dark_delta` grey levels. The surround is the BG_PERCENTILE luminance of a RING_WIDTH-pixel ring just outside
    the border. None when the box is off the image, too small to inset, or on a dark surround."""
    stats = _box_ink_stats(image, box_px, dark_delta=dark_delta)
    return None if stats is None else stats[0]


def _box_ink_stats(image: Image.Image, box_px: Rect, *, dark_delta: int = DARK_DELTA) -> tuple[float, float] | None:
    """(`box_ink_ratio`, grey-level standard deviation of the same inset interior)."""
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
    return float(np.count_nonzero(inner < bg - dark_delta)) / inner.size, float(inner.std())


def classify_box_ink(
    ink_ratio: float,
    inner_std: float | None = None,
    *,
    empty_max: float = EMPTY_MAX,
    checked_min: float = CHECKED_MIN,
    bullet_max: float = BULLET_MAX,
    min_std: float = MIN_STD,
) -> bool | None:
    """True (ticked), False (empty) or None (ambiguous: a faint speck, a solid `■` bullet or shaded box, or, given
    the interior's grey-level `inner_std`, a flat shade under min_std)."""
    if ink_ratio <= empty_max:
        return False
    flat = inner_std is not None and inner_std < min_std
    return True if checked_min <= ink_ratio <= bullet_max and not flat else None


def _box_right_of(w0: PtWord, cboxes: list[CheckWidget], words: list[PtWord], skip: set[int]) -> int | None:
    """`_box_left_of` mirrored: the first unused box immediately right of printed word w0 on its line, with no
    printed word between them (`ja [ ] nein [x]`)."""
    h = max(6.0, w0.y1 - w0.y0)
    for bi, (bx0, by0, _bx1, by1, _chk) in enumerate(cboxes):
        if bi in skip:
            continue
        bcy = (by0 + by1) / 2
        if w0.y0 - 0.6 * h <= bcy <= w0.y1 + 0.6 * h and 0 <= bx0 - w0.x1 <= 2.5 * h:
            if not any(w0.x1 < p.cx < bx0 and abs(p.cy - w0.cy) < 0.5 * h for p in words):
                return bi
    return None


def _box_owner_side(box: Rect, words: Sequence[PtWord]) -> str | None:
    """Which side's printed word a box labels: "right" for a box left of its label (`[ ] Yes`), "left" for one
    right of its label (`Yes [ ]`). With words on both sides of it on its line, its gap to its label is at most
    OWNER_GAP of its gap to the other word; None with no word beside it, or neither gap clearly the smaller."""
    bx0, by0, bx1, by1 = box
    bcy, tol = (by0 + by1) / 2, max(6.0, 0.8 * (by1 - by0))
    line = [w for w in words if abs(w.cy - bcy) <= tol]
    gap_l = min((max(0.0, bx0 - w.x1) for w in line if w.x1 <= bx0 + 1.0), default=None)
    gap_r = min((max(0.0, w.x0 - bx1) for w in line if w.x0 >= bx1 - 1.0), default=None)
    if gap_l is None:
        return None if gap_r is None else "right"
    if gap_r is None:
        return "left"
    if gap_l > 0 and gap_r <= OWNER_GAP * gap_l:
        return "right"
    return "left" if gap_r > 0 and gap_l <= OWNER_GAP * gap_r else None


def _mark_before(md: str, lo: int, start: int) -> re.Match[str] | None:
    """The last `[ ]`/`[x]` mark in md[lo:start] with only spaces or emphasis between it and `start`."""
    marks = list(_BOX_MARK.finditer(md, lo, start))
    return marks[-1] if marks and _MARK_GAP.fullmatch(md, marks[-1].end(), start) else None


def _mark_after(md: str, end: int, hi: int) -> re.Match[str] | None:
    """The first `[ ]`/`[x]` mark in md[end:hi] with only spaces, emphasis or a colon between `end` and it."""
    m = _BOX_MARK.search(md, end, hi)
    return m if m and _MARK_GAP_AFTER.fullmatch(md, end, m.start()) else None


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
    """Set the `[ ]`/`[x]` mark beside each aligned printed label to the ink state of its check box (`boxes` in
    top-down points, default `vector_checkbox_rects`): the mark right before the label (within `_MARK_WINDOW`
    characters in its unit, only spaces or emphasis between) for a box immediately left of it, else the mark
    right after it (only spaces, emphasis or a colon between) for a box immediately right of it; a box counts
    only from the side of the word it labels (`_box_owner_side`). Each box and each mark is used once. Squares
    around a printed character are not boxes; boxes overlapping `skip_boxes` (AcroForm widgets already set) and
    ambiguous boxes keep their marks; pages with < MIN_BOXES boxes or no text layer are left alone."""
    if not markdown or not rendered.text_words or rendered.image is None:
        return []
    words = _pt_words(rendered)
    boxes = vector_checkbox_rects(pdf_bytes, page_index) if boxes is None else boxes
    boxes = [b for b in boxes if not _holds_print(b, words)]
    if len(boxes) < MIN_BOXES:
        return []
    s = rendered.scale
    states: list[bool | None] = []
    for b in boxes:
        skipped = any(b[0] < k[2] and k[0] < b[2] and b[1] < k[3] and k[1] < b[3] for k in skip_boxes or ())
        ink = None if skipped else _box_ink_stats(rendered.image, (b[0] / s, b[1] / s, b[2] / s, b[3] / s))
        states.append(None if ink is None else classify_box_ink(*ink, empty_max=empty_max, checked_min=checked_min))
    if all(st is None for st in states):
        return []
    al = align(markdown, words, reorder=True)
    # token -> PDF word of its first (left-of check) and last (right-of check) matched character
    matched = [(al.a_tok[k], al.p_word[p]) for k, p in enumerate(al.amatch) if p is not None]
    first, last = dict(reversed(matched)), dict(matched)
    units = [0] + [m.end() for m in _UNIT_END.finditer(markdown)] + [len(markdown)]
    widgets = [(*b, False) for b in boxes]  # the shared geometry checks read only the rect
    # a box is read only from the side of the word it labels: `[ ] Yes` before its word, `Yes [ ]` after it
    sides = [_box_owner_side(b, words) for b in boxes]
    not_before = {i for i, sd in enumerate(sides) if sd != "right"}
    not_after = {i for i, sd in enumerate(sides) if sd != "left"}
    used: set[int] = set()
    taken: set[tuple[int, int]] = set()  # spans of marks already set from a box
    edits: list[tuple[int, int, str]] = []
    for t, w in sorted(first.items()):
        tok = al.tokens[t]
        i = bisect.bisect_right(units, tok.start) - 1
        before = _mark_before(markdown, max(units[i], tok.start - _MARK_WINDOW), tok.start)
        after = _mark_after(markdown, tok.end, min(units[i + 1], tok.end + _MARK_WINDOW))
        pairs = ((before, _box_left_of, words[w], not_before), (after, _box_right_of, words[last[t]], not_after))
        for mark, near, word, wrong_side in pairs:
            if mark is None or mark.span() in taken or (bi := near(word, widgets, words, used | wrong_side)) is None:
                continue
            used.add(bi)
            taken.add(mark.span())
            state = states[bi]
            if state is not None and bool(mark.group(1)) != state:
                edits.append((mark.start(), mark.end(), "[x]" if state else "[ ]"))
            break
    return edits


def repair_checkbox_ink(
    markdown: str, rendered: RenderedPage, pdf_bytes: bytes, page_index: int = 0, **opts: Any
) -> str:
    """The markdown with its check-box marks set from the render's ink (`ink_checkbox_edits`)."""
    edits = ink_checkbox_edits(markdown, rendered, pdf_bytes, page_index, **opts)
    return apply_edits(markdown, edits) if edits else markdown

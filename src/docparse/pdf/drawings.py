"""Horizontal lines drawn on a PDF page, and the characters they strike through or underline.

A strikethrough or an underline is not text: word processors draw it as a stroked line or a thin filled rectangle
across the characters. `horizontal_lines` collects such lines from the page's path objects (inside form XObjects
too); `decoration_marks` says, per character, whether a line crosses it at mid height (struck) or runs just below
its baseline (underlined). Everything is in PDF user space (points, origin bottom-left), where text on an
ordinary page runs horizontally.

A line that runs far beyond the characters it crosses is a rule (a table border, a heading rule behind its text),
not a decoration, and marks nothing. With a probe of the rendered page, a line must also show between the
characters it crosses: a border hidden behind a heading's background panel is not a strike.

An underline also starts and ends with its text and follows a run of it: a line that runs on well past the
characters above it, or under three or more groups of characters a column apart, is a rule too (a cell border,
the rule under a row of figures). One wide gap is allowed: an underline does cross a tab. A dashed stroke is
never a decoration (the leaders a word processor draws from a tracked change, under the line of text, to its
balloon in the margin).
"""

from __future__ import annotations

import ctypes
import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass

MAX_THICK = 3.0  # points: thicker shapes are bars or boxes, not lines
STRIKE_BAND = (0.10, 0.40)  # line centre above the baseline, in character heights (font ascent - descent)
UNDER_BAND = (-0.35, -0.02)
LINE_THICK = 0.2  # a decoration is at most this many character heights thick
CHAR_COVER = 0.5  # the line covers at least this share of a character's width
LINE_COVER = 0.6  # the characters a line crosses span at least this share of its length
DARK = 160  # luminance under which a rendered pixel is ink
OVERHANG = 2.0  # an underline runs at most this many character heights past its text, on either side
COLUMN_GAP = 3.0  # characters further apart than this many character heights stand in different columns
SHOWS = 0.6  # share of the gaps between crossed characters where the line must show as ink ...
CONTRAST = 0.3  # ... and how much more often than in the same gaps a little higher up


@dataclass(frozen=True)
class Line:
    x0: float
    x1: float
    y: float  # centre, PDF user space
    thick: float


@dataclass(frozen=True)
class CharGeom:
    x0: float
    x1: float
    baseline: float  # y of the character origin
    height: float  # font ascent - descent at the drawn size


def _mul(a: tuple, b: tuple) -> tuple:
    """Affine matrices (a, b, c, d, e, f): first a, then b."""
    return (
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    )


def _matrix(raw, obj) -> tuple:
    m = raw.FS_MATRIX()
    if not raw.FPDFPageObj_GetMatrix(obj, ctypes.byref(m)):
        return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    return (m.a, m.b, m.c, m.d, m.e, m.f)


def _path_lines(raw, obj, ctm: tuple, out: list[Line]) -> None:
    m = _mul(_matrix(raw, obj), ctm)
    fill, stroke = ctypes.c_int(), ctypes.c_int()
    raw.FPDFPath_GetDrawMode(obj, ctypes.byref(fill), ctypes.byref(stroke))
    if not fill.value and not stroke.value:
        return  # a clipping path or an invisible shape
    width = ctypes.c_float()
    raw.FPDFPageObj_GetStrokeWidth(obj, ctypes.byref(width))
    stroke_w = width.value * math.sqrt(abs(m[0] * m[3] - m[1] * m[2])) if stroke.value else 0.0
    dashed = bool(stroke.value) and raw.FPDFPageObj_GetDashCount(obj) > 0
    x, y = ctypes.c_float(), ctypes.c_float()
    subpaths: list[list[tuple[float, float]]] = []
    curved: list[bool] = []
    for i in range(raw.FPDFPath_CountSegments(obj)):
        seg = raw.FPDFPath_GetPathSegment(obj, i)
        raw.FPDFPathSegment_GetPoint(seg, ctypes.byref(x), ctypes.byref(y))
        px, py = x.value * m[0] + y.value * m[2] + m[4], x.value * m[1] + y.value * m[3] + m[5]
        kind = raw.FPDFPathSegment_GetType(seg)
        if kind == raw.FPDF_SEGMENT_MOVETO or not subpaths:
            subpaths.append([])
            curved.append(False)
        if kind == raw.FPDF_SEGMENT_BEZIERTO:
            curved[-1] = True
        subpaths[-1].append((px, py))
    for points, bent in zip(subpaths, curved, strict=True):
        if bent or len(points) < 2:
            continue
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        height = max(ys) - min(ys)
        length = max(xs) - min(xs)
        if length <= 0:
            continue
        if fill.value and len(points) >= 4 and height <= MAX_THICK and length > 2 * height:
            out.append(Line(min(xs), max(xs), (max(ys) + min(ys)) / 2, max(height, stroke_w)))
        elif stroke.value and stroke_w <= MAX_THICK and not dashed:
            for (ax, ay), (bx, by) in zip(points, points[1:], strict=False):
                if abs(ay - by) <= 0.1 * abs(ax - bx) and abs(ax - bx) > 0:
                    out.append(Line(min(ax, bx), max(ax, bx), (ay + by) / 2, stroke_w))


def _walk(raw, get_count, get_object, ctm: tuple, out: list[Line], depth: int) -> None:
    for i in range(get_count()):
        obj = get_object(i)
        kind = raw.FPDFPageObj_GetType(obj)
        if kind == raw.FPDF_PAGEOBJ_PATH:
            _path_lines(raw, obj, ctm, out)
        elif kind == raw.FPDF_PAGEOBJ_FORM and depth < 8:
            inner = _mul(_matrix(raw, obj), ctm)
            _walk(
                raw,
                lambda o=obj: raw.FPDFFormObj_CountObjects(o),
                lambda k, o=obj: raw.FPDFFormObj_GetObject(o, k),
                inner,
                out,
                depth + 1,
            )


def horizontal_lines(page) -> list[Line]:
    """Horizontal stroked segments and thin filled rectangles of a pypdfium2 page, in PDF user space."""
    import pypdfium2.raw as raw

    out: list[Line] = []
    _walk(
        raw,
        lambda: raw.FPDFPage_CountObjects(page.raw),
        lambda k: raw.FPDFPage_GetObject(page.raw, k),
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        out,
        0,
    )
    return out


def _shows(line: Line, cis: list[int], chars: list[CharGeom], mark: str, ink: Callable[[float, float], int]) -> bool:
    """Is the line drawn where nothing else is, between the characters it crosses? Compared with the same gaps a
    little higher up, so a dark background or a glyph does not count as the line."""
    lo, hi = min(chars[c].x0 for c in cis), max(chars[c].x1 for c in cis)
    boxes = sorted((chars[c].x0, chars[c].x1) for c in cis)
    step = max(0.2, (hi - lo) / 400)
    gaps, x, k = [], lo, 0
    while x <= hi:
        while k < len(boxes) and boxes[k][1] < x:
            k += 1
        if not any(a <= x <= b for a, b in boxes[k : k + 3]):
            gaps.append(x)
        x += step
    if len(gaps) < 4:
        return True  # nothing between the glyphs to look at
    height = sorted(chars[c].height for c in cis)[len(cis) // 2]
    up = line.y + (0.25 if mark == "s" else 0.3) * height
    on = sum(ink(x, line.y) < DARK for x in gaps) / len(gaps)
    off = sum(ink(x, up) < DARK for x in gaps) / len(gaps)
    return on >= SHOWS and on - off >= CONTRAST


def _is_rule(line: Line, over: list[CharGeom]) -> bool:
    """Does the line under these characters run on past them, or under three or more columns of them?"""
    height = sorted(c.height for c in over)[len(over) // 2]
    boxes = sorted((c.x0, c.x1) for c in over)
    if max(boxes[0][0] - line.x0, line.x1 - max(b for _, b in boxes)) > OVERHANG * height:
        return True
    reach, columns = boxes[0][1], 1
    for a, b in boxes[1:]:
        columns += a - reach > COLUMN_GAP * height
        reach = max(reach, b)
    return columns >= 3


def decoration_marks(chars: list[CharGeom], lines: list[Line], ink: Callable[[float, float], int] | None = None) -> str:
    """One mark per character: "s" struck through, "u" underlined, "." neither. `ink(x, y)` is the luminance of
    the rendered page at a point of user space (darkest of the pixels around it), when a render is at hand."""
    marks = ["."] * len(chars)
    if not lines:
        return "".join(marks)
    ordered = sorted(lines, key=lambda ln: ln.y)
    ys = [ln.y for ln in ordered]
    crossed: dict[tuple[int, str], list[int]] = {}
    for ci, ch in enumerate(chars):
        width = ch.x1 - ch.x0
        if ch.height <= 0 or width <= 0:
            continue
        lo, hi = ch.baseline + UNDER_BAND[0] * ch.height, ch.baseline + STRIKE_BAND[1] * ch.height
        for li in range(bisect_left(ys, lo), bisect_right(ys, hi)):
            ln = ordered[li]
            if ln.thick > LINE_THICK * ch.height or min(ch.x1, ln.x1) - max(ch.x0, ln.x0) < CHAR_COVER * width:
                continue
            rel = (ln.y - ch.baseline) / ch.height
            if STRIKE_BAND[0] <= rel <= STRIKE_BAND[1]:
                crossed.setdefault((li, "s"), []).append(ci)
            elif UNDER_BAND[0] <= rel <= UNDER_BAND[1]:
                crossed.setdefault((li, "u"), []).append(ci)
    for (li, mark), cis in sorted(crossed.items(), key=lambda kv: kv[0][1] == "s"):  # strikes win
        ln = ordered[li]
        span = max(chars[c].x1 for c in cis) - min(chars[c].x0 for c in cis)
        if span < LINE_COVER * (ln.x1 - ln.x0):
            continue  # a rule across a column or a table
        if mark == "u" and _is_rule(ln, [chars[c] for c in cis]):
            continue  # a cell border or the rule under a row, not an underline
        if ink is not None and not _shows(ln, cis, chars, mark, ink):
            continue  # hidden behind something drawn over it
        for c in cis:
            marks[c] = mark
    return "".join(marks)

"""Vector paths of a PDF page in display points (origin top-left, CropBox and /Rotate applied), via pypdfium2.

Each path object becomes one `Shape`: its subpaths as point lists, whether each is closed, its fill and stroke
colours (None when not painted that way) and its stroke width in points. Paths inside Form XObjects are placed
with the composed matrices of their containers. Bezier segments keep only their end points (a curve through
data points still passes through them).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

Color = tuple[int, int, int]
Matrix = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class Shape:
    subpaths: tuple[tuple[tuple[float, float], ...], ...]
    closed: tuple[bool, ...]
    fill: Color | None
    stroke: Color | None
    width: float  # stroke width in points
    alpha: int = 255  # fill (or stroke) opacity, 0..255


def _mul(m1: Matrix, m2: Matrix) -> Matrix:
    """m1 then m2 (PDF row-vector convention)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def _matrix(obj) -> Matrix:
    m = obj.get_matrix()
    return (m.a, m.b, m.c, m.d, m.e, m.f)


def _color(fn, obj) -> tuple[Color, int] | None:
    r, g, b, a = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
    if not fn(obj, ctypes.byref(r), ctypes.byref(g), ctypes.byref(b), ctypes.byref(a)):
        return None
    return (r.value, g.value, b.value), a.value


def page_shapes(page) -> list[Shape]:
    """Every painted path of a pypdfium2 page, in display points."""
    import pypdfium2.raw as raw

    cx0, cy0, cx1, cy1 = page.get_cropbox()
    rot = page.get_rotation() % 360

    def disp(ux: float, uy: float) -> tuple[float, float]:
        if rot == 90:
            return uy - cy0, ux - cx0
        if rot == 180:
            return cx1 - ux, uy - cy0
        if rot == 270:
            return cy1 - uy, cx1 - ux
        return ux - cx0, cy1 - uy

    out: list[Shape] = []
    x, y = ctypes.c_float(), ctypes.c_float()
    for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_PATH], max_depth=8):
        m = _matrix(obj)
        cont = obj.container
        while cont is not None:
            m = _mul(m, _matrix(cont))
            cont = cont.container
        mode, stroke_on = ctypes.c_int(), ctypes.c_int()
        if not raw.FPDFPath_GetDrawMode(obj, ctypes.byref(mode), ctypes.byref(stroke_on)):
            continue
        fill = _color(raw.FPDFPageObj_GetFillColor, obj) if mode.value else None
        stroke = _color(raw.FPDFPageObj_GetStrokeColor, obj) if stroke_on.value else None
        if fill is None and stroke is None:
            continue  # a clip or an unpainted path
        w = ctypes.c_float()
        raw.FPDFPageObj_GetStrokeWidth(obj, ctypes.byref(w))
        scale = abs(m[0] * m[3] - m[1] * m[2]) ** 0.5
        a, b, c, d, e, f = m
        subs: list[list[tuple[float, float]]] = []
        closed: list[bool] = []
        bezier = 0
        for i in range(raw.FPDFPath_CountSegments(obj)):
            seg = raw.FPDFPath_GetPathSegment(obj, i)
            kind = raw.FPDFPathSegment_GetType(seg)
            raw.FPDFPathSegment_GetPoint(seg, ctypes.byref(x), ctypes.byref(y))
            p = disp(a * x.value + c * y.value + e, b * x.value + d * y.value + f)
            if kind == raw.FPDF_SEGMENT_MOVETO or not subs:
                subs.append([p])
                closed.append(False)
                bezier = 0
            elif kind == raw.FPDF_SEGMENT_BEZIERTO:
                bezier += 1
                if bezier % 3 == 0:  # control points are skipped; the end point is on the curve
                    subs[-1].append(p)
            else:
                bezier = 0
                subs[-1].append(p)
            if raw.FPDFPathSegment_GetClose(seg):
                closed[-1] = True
        if not subs:
            continue
        paint = fill or stroke
        out.append(
            Shape(
                subpaths=tuple(tuple(s) for s in subs),
                closed=tuple(closed),
                fill=fill[0] if fill else None,
                stroke=stroke[0] if stroke else None,
                width=w.value * scale,
                alpha=paint[1] if paint else 255,
            )
        )
    return out


def load_shapes(data: bytes, index: int) -> list[Shape]:
    """The shapes of one page of a PDF given as bytes ([] for an image input)."""
    import pypdfium2 as pdfium

    from docparse.pdf.render import _PDFIUM_LOCK

    if not data.lstrip()[:5].startswith(b"%PDF-"):
        return []
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(data)
        try:
            return page_shapes(pdf[index])
        finally:
            pdf.close()

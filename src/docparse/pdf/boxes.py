"""Check boxes drawn on a PDF page: small, nearly square vector paths.

A form draws each check box as a path (often twice: an outer and an inner border). Their outlines, in points in
the displayed page frame (CropBox and /Rotate applied, origin top-left), tell where an option's box is, so a tick
can be put in a box and a box next to its label.
"""

from __future__ import annotations

from docparse.pdf.render import _PDFIUM_LOCK, _to_image

MIN_SIDE, MAX_SIDE = 4.0, 20.0  # points
SQUARE = 1.45  # the longer side at most this many times the shorter


def check_boxes(data: bytes, page_index: int = 0) -> list[tuple[float, float, float, float]]:
    """(x0, y0, x1, y1) of each check box on the page; nested duplicates merged into the outer one."""
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw

        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(data)
            try:
                page = pdf[page_index]
                rot, crop = page.get_rotation() % 360, page.get_cropbox()
                found = []
                for obj in page.get_objects(max_depth=4):
                    if obj.type != raw.FPDF_PAGEOBJ_PATH:
                        continue
                    left, bottom, right, top = obj.get_bounds()
                    w, h = right - left, top - bottom
                    if MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE and max(w, h) <= SQUARE * min(w, h):
                        found.append(tuple(_to_image(left, bottom, right, top, crop, rot, 1.0)))
            finally:
                pdf.close()
    except Exception:  # an unreadable page simply has no boxes
        return []
    found.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    out: list[tuple[float, float, float, float]] = []
    for b in found:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if not any(o[0] <= cx <= o[2] and o[1] <= cy <= o[3] for o in out):
            out.append(b)
    return out

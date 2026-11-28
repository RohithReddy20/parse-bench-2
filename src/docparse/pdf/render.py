"""Page renders and text-layer words in OCRia-1's pixel frame.

This follows the ParseBench harness's `ocria_1` provider (`OCRia1Provider._pages` and `page_words`, at
run-llama/ParseBench@6e4479a) step for step, because OCRia-1's answers and its grounding code are only
reproducible on the same image: pypdfium2, CropBox and /Rotate applied, 300 DPI, and a lower scale for sheets
that would exceed 80 million pixels.
"""

from __future__ import annotations

import hashlib
import io
import threading
from dataclasses import dataclass, field
from pathlib import Path

from docparse.core.errors import ParseError

DPI = 300
MAX_PIXELS = 80_000_000  # vLLM's image loader refuses images over Pillow's ~89M-pixel guard
_PDFIUM_LOCK = threading.Lock()  # PDFium is not thread-safe


@dataclass
class Word:
    text: str
    box: list[float]  # [x0, y0, x1, y1] in pixels of the rendered image
    font: str = ""  # base font of the first character (text_words only), without a subset prefix
    # per character of `text`: "s" struck through, "u" underlined, "." neither (text_words, when decorations
    # were requested; empty otherwise)
    marks: str = ""
    bold: float = 0.0  # share of the characters set in a bold face (text_words, when styles were requested)


@dataclass
class RenderedPage:
    image: object  # PIL.Image.Image, RGB
    dpi: float  # the DPI this page was actually rendered at
    width_pt: float  # displayed page size in PDF points
    height_pt: float
    rotation: int = 0
    words: list[Word] | None = field(default=None)  # None: the input has no text layer at all (an image)
    # the same text layer split at the spaces PDFium reports, not at glyph gaps: use these to *read* text.
    # (`words` follows OCRia-1's gap rule, which its grounding code needs but which breaks words apart in
    # fonts with loose glyph boxes: "R egul atory".)
    text_words: list[Word] | None = field(default=None)
    # how far the text layer can be trusted to say what the page shows (when requested with `layer`): the share
    # of its characters nobody sees (invisible render mode, or under an image drawn over the page: an OCR layer),
    # and the share whose glyph has no proper unicode mapping
    hidden: float | None = None
    map_errors: float | None = None

    @property
    def scale(self) -> float:
        """Pixels -> PDF points."""
        return 72.0 / self.dpi


def _to_image(left: float, b: float, r: float, t: float, crop, rot: int, s: float) -> list[float]:
    cx0, cy0, cx1, cy1 = crop

    def m(ux: float, uy: float) -> tuple[float, float]:
        return {
            0: (ux - cx0, cy1 - uy),
            90: (uy - cy0, ux - cx0),
            180: (cx1 - ux, uy - cy0),
            270: (cy1 - uy, cx1 - ux),
        }[rot]

    xs, ys = zip(m(left, b), m(r, t), strict=True)
    return [min(xs) * s, min(ys) * s, max(xs) * s, max(ys) * s]


def page_words(page, dpi: float = DPI) -> list[Word]:
    """Text-layer words of a pypdfium2 page in pixels of its `page.render(scale=dpi / 72)` image: characters
    joined while the gap is under 0.35 line heights on one line (the harness's and `ocria_grounding`'s rule)."""
    s, rot, crop = dpi / 72, page.get_rotation() % 360, page.get_cropbox()
    tp = page.get_textpage()
    out: list[Word] = []
    cur: Word | None = None
    for i in range(tp.count_chars()):
        ch = tp.get_text_range(i, 1)
        if not ch or ch.isspace() or ch in "\r\n\x00￾":
            if cur:
                out.append(cur)
                cur = None
            continue
        box = _to_image(*tp.get_charbox(i), crop, rot, s)
        hgt = box[3] - box[1]
        if cur and box[0] - cur.box[2] < 0.35 * hgt and abs(box[1] - cur.box[1]) < 0.5 * hgt:
            cur.text += ch
            cb = cur.box
            cur.box = [min(cb[0], box[0]), min(cb[1], box[1]), max(cb[2], box[2]), max(cb[3], box[3])]
        else:
            if cur:
                out.append(cur)
            cur = Word(text=ch, box=box)
    if cur:
        out.append(cur)
    return out


def page_text_words(page, dpi: float = DPI, decorations: bool = False, image=None, styles: bool = False) -> list[Word]:
    """Text-layer words split at whitespace characters (PDFium inserts a space where the page shows one)
    and at line changes. Boxes are pixels of the page render, as in `page_words`. With `decorations`, each word
    also gets the strikethrough / underline marks of its characters (`pdf.drawings`), checked against the page
    render `image` when given; with `styles`, the share of its characters set in a bold face."""
    s, rot, crop = dpi / 72, page.get_rotation() % 360, page.get_cropbox()
    tp = page.get_textpage()
    out: list[Word] = []
    cur: Word | None = None
    geom: list[tuple[Word, int, object]] = []  # per kept character: its word, its length, its geometry
    faces: list[tuple[Word, bool]] = []  # per kept character: its word, whether its face is bold
    for i in range(tp.count_chars()):
        ch = tp.get_text_range(i, 1)
        if not ch or ch.isspace() or ch in "\r\n\x00\ufffe":
            if cur:
                out.append(cur)
                cur = None
            continue
        user_box = tp.get_charbox(i)
        box = _to_image(*user_box, crop, rot, s)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue  # a glyph without extent
        if cur:
            height = max(cur.box[3] - cur.box[1], box[3] - box[1])
            same_line = box[1] < cur.box[3] + 0.3 * height and box[3] > cur.box[1] - 0.3 * height
            if not same_line:
                out.append(cur)
                cur = None
        if cur:
            cur.text += ch
            cb = cur.box
            cur.box = [min(cb[0], box[0]), min(cb[1], box[1]), max(cb[2], box[2]), max(cb[3], box[3])]
        else:
            cur = Word(text=ch, box=box, font=_char_font(tp, i))
        if decorations:
            geom.append((cur, len(ch), _char_geom(tp, i, user_box)))
        if styles:
            faces.append((cur, _char_bold(tp, i)))
    if cur:
        out.append(cur)
    if styles:
        counts: dict[int, list[int]] = {}
        for word, bold in faces:
            c = counts.setdefault(id(word), [0, 0])
            c[0] += bold
            c[1] += 1
        for word in out:
            if id(word) in counts:
                word.bold = counts[id(word)][0] / counts[id(word)][1]
    if decorations:
        from docparse.pdf.drawings import decoration_marks, horizontal_lines

        ink = _ink_probe(image, crop, rot, s) if image is not None else None
        marks = decoration_marks([g for _, _, g in geom], horizontal_lines(page), ink)
        for (word, n, _), mark in zip(geom, marks, strict=True):
            word.marks += mark * n
    return out


def _ink_probe(image, crop, rot: int, s: float):
    """Luminance of the render at a point of PDF user space: the darkest pixel of the 3 x 3 around it."""
    gray = image.convert("L")
    px, (width, height) = gray.load(), gray.size

    def ink(x: float, y: float) -> int:
        ix, iy = (int(v) for v in _to_image(x, y, x, y, crop, rot, s)[:2])
        near = [(i, j) for i in (ix - 1, ix, ix + 1) for j in (iy - 1, iy, iy + 1)]
        vals = [px[i, j] for i, j in near if 0 <= i < width and 0 <= j < height]
        return min(vals) if vals else 255

    return ink


def layer_health(page) -> tuple[float, float]:
    """(hidden, map_errors) shares of the page's text-layer characters. Hidden: drawn in the invisible render mode,
    or all of them when an image covering most of the page is drawn after the text (a scan over its OCR layer)."""
    import ctypes

    import pypdfium2.raw as raw

    tp = page.get_textpage()
    n = tp.count_chars()
    if not n:
        return 0.0, 0.0
    invisible = errors = 0
    for i in range(n):
        obj = raw.FPDFText_GetTextObject(tp.raw, i)
        if obj and raw.FPDFTextObj_GetTextRenderMode(obj) == raw.FPDF_TEXTRENDERMODE_INVISIBLE:
            invisible += 1
        if raw.FPDFText_HasUnicodeMapError(tp.raw, i) == 1:
            errors += 1
    w, h = page.get_size()
    seen_text = False
    box = [ctypes.c_float() for _ in range(4)]
    for k in range(raw.FPDFPage_CountObjects(page.raw)):
        obj = raw.FPDFPage_GetObject(page.raw, k)
        kind = raw.FPDFPageObj_GetType(obj)
        if kind == raw.FPDF_PAGEOBJ_TEXT:
            seen_text = True
        elif kind == raw.FPDF_PAGEOBJ_IMAGE and seen_text:
            raw.FPDFPageObj_GetBounds(obj, *(ctypes.byref(v) for v in box))
            left, bottom, right, top = (v.value for v in box)
            if (right - left) * (top - bottom) >= 0.8 * w * h:
                invisible = n
    return invisible / n, errors / n


def _char_geom(textpage, index: int, user_box):
    """A character's extent, baseline and height (font ascent - descent) in PDF user space."""
    import ctypes

    import pypdfium2.raw as raw

    from docparse.pdf.drawings import CharGeom

    x, y = ctypes.c_double(), ctypes.c_double()
    raw.FPDFText_GetCharOrigin(textpage.raw, index, ctypes.byref(x), ctypes.byref(y))
    loose = raw.FS_RECTF()
    ok = raw.FPDFText_GetLooseCharBox(textpage.raw, index, ctypes.byref(loose))
    height = abs(loose.top - loose.bottom) if ok else 0.0
    return CharGeom(x0=user_box[0], x1=user_box[2], baseline=y.value, height=height)


_BOLD_NAMES = ("bold", "black", "heavy", "ultra", "extrab", "semib", "demib", "debold")


def _char_bold(textpage, index: int) -> bool:
    """Is the character set in a bold face? Its font's name says so (PostScript names carry the weight), or its
    weight is 600 or more, or its font forces bold, or it is filled and stroked (synthetic bold)."""
    import ctypes

    import pypdfium2.raw as raw

    buf = ctypes.create_string_buffer(256)
    flags = ctypes.c_int()
    n = raw.FPDFText_GetFontInfo(textpage.raw, index, buf, 256, ctypes.byref(flags))
    name = buf.value.decode("utf-8", "replace").lower() if n else ""
    if any(k in name for k in _BOLD_NAMES) or flags.value & (1 << 18):
        return True
    if raw.FPDFText_GetFontWeight(textpage.raw, index) >= 600:
        return True
    obj = raw.FPDFText_GetTextObject(textpage.raw, index)
    return bool(obj) and raw.FPDFTextObj_GetTextRenderMode(obj) == raw.FPDF_TEXTRENDERMODE_FILL_STROKE


def _char_font(textpage, index: int) -> str:
    import ctypes

    import pypdfium2.raw as raw

    from docparse.pdf.overlay import font_name

    buf = ctypes.create_string_buffer(256)
    flags = ctypes.c_int()
    n = raw.FPDFText_GetFontInfo(textpage.raw, index, buf, 256, ctypes.byref(flags))
    return font_name(buf.value.decode("utf-8", "replace")) if n else ""


def _render_image(data: bytes, dpi: float) -> list[RenderedPage]:
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise ParseError("the input is neither a readable PDF nor an image") from exc
    image = image.convert("RGB")
    return [RenderedPage(image=image, dpi=dpi, width_pt=image.width * 72 / dpi, height_pt=image.height * 72 / dpi)]


def render_pages(
    data: bytes,
    dpi: float = DPI,
    max_pages: int = 400,
    decorations: bool = False,
    styles: bool = False,
    layer: bool = False,
) -> list[RenderedPage]:
    """One RenderedPage per page of a PDF (or one for an image file). Raises ParseError on unreadable bytes.
    `decorations`: also mark struck-through and underlined characters of the text words; `styles`: also measure
    how bold each text word is; `layer`: also measure how far the text layer can be trusted (`layer_health`)."""
    import pypdfium2 as pdfium

    if not data:
        raise ParseError("empty input")
    if not data.lstrip()[:5].startswith(b"%PDF-"):
        return _render_image(data, dpi)
    with _PDFIUM_LOCK:
        try:
            pdf = pdfium.PdfDocument(data)
        except Exception as exc:
            raise ParseError(f"unreadable PDF (corrupt or encrypted): {exc}") from exc
        try:
            if len(pdf) == 0:
                raise ParseError("the PDF has no pages")
            if len(pdf) > max_pages:
                raise ParseError(f"the PDF has {len(pdf)} pages (limit {max_pages})")
            out = []
            for i in range(len(pdf)):
                page = pdf[i]
                scale = dpi / 72
                w, h = page.get_size()
                if w * h * scale * scale > MAX_PIXELS:  # large sheets render smaller; words follow the same scale
                    scale = (MAX_PIXELS / (w * h)) ** 0.5
                image = page.render(scale=scale).to_pil().convert("RGB")
                hidden, map_errors = layer_health(page) if layer else (None, None)
                out.append(
                    RenderedPage(
                        image=image,
                        dpi=scale * 72,
                        width_pt=float(w),
                        height_pt=float(h),
                        rotation=page.get_rotation() % 360,
                        words=page_words(page, scale * 72),
                        text_words=page_text_words(
                            page, scale * 72, decorations, image if decorations else None, styles
                        ),
                        hidden=hidden,
                        map_errors=map_errors,
                    )
                )
            return out
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"failed to render the PDF: {exc}") from exc
        finally:
            pdf.close()


_PIXEL_ALIAS: dict[str, str] | None = None


def _pixel_alias() -> dict[str, str]:
    """`$DOCPARSE_PIXEL_ALIAS`: a JSON map from this machine's hash of a page image to the hash the cache knows the
    page by. PDFium rasterises some fonts a shade differently on Linux and macOS (the pages look the same), and a
    cache filled on one must still answer on the other; `evalkit.pixel_alias` builds the map from page identity
    (document hash, page number, derivation)."""
    global _PIXEL_ALIAS
    if _PIXEL_ALIAS is None:
        import json
        import os

        path = os.environ.get("DOCPARSE_PIXEL_ALIAS")
        _PIXEL_ALIAS = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    return _PIXEL_ALIAS


def raw_pixel_hash(image) -> str:
    h = hashlib.sha256(f"{image.mode}:{image.width}x{image.height}:".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def pixel_hash(image) -> str:
    """Identity of a rendered page: its pixels, not its PNG bytes (PNG encoding varies across zlib builds)."""
    digest = raw_pixel_hash(image)
    return _pixel_alias().get(digest, digest)


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

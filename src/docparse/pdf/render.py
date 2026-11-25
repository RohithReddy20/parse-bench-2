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

from docparse.core.errors import ParseError

DPI = 300
MAX_PIXELS = 80_000_000  # vLLM's image loader refuses images over Pillow's ~89M-pixel guard
_PDFIUM_LOCK = threading.Lock()  # PDFium is not thread-safe


@dataclass
class Word:
    text: str
    box: list[float]  # [x0, y0, x1, y1] in pixels of the rendered image
    font: str = ""  # base font of the first character (text_words only), without a subset prefix


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


def page_text_words(page, dpi: float = DPI) -> list[Word]:
    """Text-layer words split at whitespace characters (PDFium inserts a space where the page shows one)
    and at line changes. Boxes are pixels of the page render, as in `page_words`."""
    s, rot, crop = dpi / 72, page.get_rotation() % 360, page.get_cropbox()
    tp = page.get_textpage()
    out: list[Word] = []
    cur: Word | None = None
    for i in range(tp.count_chars()):
        ch = tp.get_text_range(i, 1)
        if not ch or ch.isspace() or ch in "\r\n\x00\ufffe":
            if cur:
                out.append(cur)
                cur = None
            continue
        box = _to_image(*tp.get_charbox(i), crop, rot, s)
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
    if cur:
        out.append(cur)
    return out


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


def render_pages(data: bytes, dpi: float = DPI, max_pages: int = 400) -> list[RenderedPage]:
    """One RenderedPage per page of a PDF (or one for an image file). Raises ParseError on unreadable bytes."""
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
                out.append(
                    RenderedPage(
                        image=image,
                        dpi=scale * 72,
                        width_pt=float(w),
                        height_pt=float(h),
                        rotation=page.get_rotation() % 360,
                        words=page_words(page, scale * 72),
                        text_words=page_text_words(page, scale * 72),
                    )
                )
            return out
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"failed to render the PDF: {exc}") from exc
        finally:
            pdf.close()


def pixel_hash(image) -> str:
    """Identity of a rendered page: its pixels, not its PNG bytes (PNG encoding varies across zlib builds)."""
    h = hashlib.sha256(f"{image.mode}:{image.width}x{image.height}:".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

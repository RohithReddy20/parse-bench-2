"""Pages printed sideways or upside down, set upright before they are read.

The text layer says which way the text runs (`fixes.tables_align.upright_turn`). `turn_page` turns the
rendered image and the word boxes together, so the reader and every later correction see one consistent
upright page; `unturn_box` takes a box found on the upright page back to the page as it is stored.
"""

from __future__ import annotations

from dataclasses import replace

from docparse.core.ir import Box
from docparse.pdf.render import RenderedPage, Word


def page_turn(rendered: RenderedPage) -> int:
    """The clockwise turn (0, 90, 180, 270) that sets this page's text upright; 0 without firm evidence."""
    from docparse.fixes.tables_align import page_words, upright_turn

    words = page_words(rendered)
    return upright_turn(words, rendered.width_pt, rendered.height_pt) if words else 0


def _turn_xy(x: float, y: float, w: float, h: float, angle: int) -> tuple[float, float]:
    if angle == 90:
        return h - y, x
    if angle == 180:
        return w - x, h - y
    if angle == 270:
        return y, w - x
    return x, y


def _turn_box(box, w: float, h: float, angle: int) -> list[float]:
    (ax, ay), (bx, by) = _turn_xy(box[0], box[1], w, h, angle), _turn_xy(box[2], box[3], w, h, angle)
    return [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]


def turn_page(rendered: RenderedPage, angle: int) -> RenderedPage:
    """The page turned clockwise by `angle` degrees: image, word boxes and page size."""
    if angle not in (90, 180, 270):
        return rendered
    from PIL import Image

    how = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[angle]
    w, h = rendered.image.size

    def words(source: list[Word] | None) -> list[Word] | None:
        return None if source is None else [replace(x, box=_turn_box(x.box, w, h, angle)) for x in source]

    swap = angle in (90, 270)
    return replace(
        rendered,
        image=rendered.image.transpose(how),
        width_pt=rendered.height_pt if swap else rendered.width_pt,
        height_pt=rendered.width_pt if swap else rendered.height_pt,
        words=words(rendered.words),
        text_words=words(rendered.text_words),
    )


def unturn_box(box: Box, angle: int, width_pt: float, height_pt: float) -> Box:
    """A box on the turned page (whose size is `width_pt` x `height_pt`) as a box on the stored page."""
    if angle not in (90, 180, 270):
        return box
    x0, y0, x1, y1 = _turn_box([box.x0, box.y0, box.x1, box.y1], width_pt, height_pt, 360 - angle)
    return Box(x0=x0, y0=y0, x1=x1, y1=y1)

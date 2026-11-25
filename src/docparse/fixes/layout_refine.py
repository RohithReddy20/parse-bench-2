"""Layout refinements on pages with a PDF text layer (plan Task 16). Each is one option, measured on its own.

    tighten      a text item's box becomes the union of the PDF words it holds
    tighten_pad  ... grown on every side by tighten_pad x the item's median word height
    native_text  a text item's text becomes those PDF words, in reading order, when the text layer is healthy
                 there: at least native_support (default 0.8) of the item's PDF tokens also occur in OCRia-1's
                 reading of the page. A broken font map or a stale OCR layer fails that test and the model's
                 text stays.
    split_gap    a text item is split where the vertical gap between two of its lines exceeds
                 split_gap x the item's median line height (each part gets its own words as text)
    orphans      PDF words that lie in no item at all become new Text items, one per block of lines, when the
                 text layer is healthy (the words the items do hold agree with the reading). orphan_gap is the
                 largest vertical gap inside one block, in line heights (default 1.0); orphan_min is the
                 fewest letters or digits a block needs (default 1). Three optional filters: orphan_ink
                 (the page shows something there), orphan_support (that share of the block's tokens is in
                 the reading) and orphan_clear (the block's box holds no word that already has an item)

A word belongs to the smallest text item holding its centre. Pictures and tables are never changed, and an
item holding no PDF words (a scan, text drawn as an image) is left exactly as it was.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import median

from docparse.core.ir import Box, LayoutItem
from docparse.pdf.render import RenderedPage

TEXT_TYPES = frozenset(
    {"Text", "Section-header", "List-item", "Caption", "Footnote", "Page-header", "Page-footer", "Title", "Code"}
)
SPLIT_TYPES = frozenset({"Text", "List-item"})
EDGE = 1.0  # points of slack when testing whether a word centre lies in a box


@dataclass(frozen=True)
class PtWord:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def words_in_points(rendered: RenderedPage) -> list[PtWord]:
    s = rendered.scale
    source = rendered.text_words if rendered.text_words is not None else rendered.words
    return [PtWord(w.text, w.box[0] * s, w.box[1] * s, w.box[2] * s, w.box[3] * s) for w in source or []]


def assign_words(items: list[LayoutItem], words: list[PtWord]) -> list[list[PtWord]]:
    """Per item, the words whose centre it holds; a word in several text items goes to the smallest."""
    owned: list[list[PtWord]] = [[] for _ in items]
    text_items = [(i, it.bbox, it.bbox.width * it.bbox.height) for i, it in enumerate(items) if it.type in TEXT_TYPES]
    for w in words:
        best = None
        for i, b, area in text_items:
            if b.x0 - EDGE <= w.cx <= b.x1 + EDGE and b.y0 - EDGE <= w.cy <= b.y1 + EDGE:
                if best is None or area < best[1]:
                    best = (i, area)
        if best is not None:
            owned[best[0]].append(w)
    return owned


def lines_of(words: list[PtWord]) -> list[list[PtWord]]:
    """Words grouped into lines, top to bottom, each line left to right."""
    lines: list[list[PtWord]] = []
    for w in sorted(words, key=lambda w: (w.cy, w.x0)):
        line = lines[-1] if lines else None
        if line is not None:
            top, bottom = min(x.y0 for x in line), max(x.y1 for x in line)
            if top <= w.cy <= bottom or abs(w.cy - (top + bottom) / 2) < 0.4 * (w.y1 - w.y0):
                line.append(w)
                continue
        lines.append([w])
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def _union(words: list[PtWord], pad: float = 0.0, page: tuple[float, float] | None = None) -> Box:
    m = pad * median(w.y1 - w.y0 for w in words) if pad else 0.0
    x0, y0 = min(w.x0 for w in words) - m, min(w.y0 for w in words) - m
    x1, y1 = max(w.x1 for w in words) + m, max(w.y1 for w in words) + m
    if page:
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(page[0], x1), min(page[1], y1)
    return Box(x0=x0, y0=y0, x1=x1, y1=y1)


def _text(lines: list[list[PtWord]]) -> str:
    return "\n".join(" ".join(w.text for w in line) for line in lines)


def split_lines(lines: list[list[PtWord]], split_gap: float) -> list[list[list[PtWord]]]:
    """Runs of lines separated by a vertical gap over split_gap x the median line height."""
    if len(lines) < 2:
        return [lines]
    heights = [max(w.y1 for w in ln) - min(w.y0 for w in ln) for ln in lines]
    limit = split_gap * median(heights)
    parts, current = [], [lines[0]]
    for prev, line in zip(lines, lines[1:], strict=False):
        gap = min(w.y0 for w in line) - max(w.y1 for w in prev)
        if gap > limit:
            parts.append(current)
            current = []
        current.append(line)
    parts.append(current)
    return parts


def _segments(line: list[PtWord], gap: float) -> list[list[PtWord]]:
    """A line cut where two neighbouring words are more than gap x the line height apart (a column gutter)."""
    limit = gap * median(w.y1 - w.y0 for w in line)
    parts = [[line[0]]]
    for prev, word in zip(line, line[1:], strict=False):
        if word.x0 - prev.x1 > limit:
            parts.append([])
        parts[-1].append(word)
    return parts


def _holds(b: Box, w: PtWord, edge: float = EDGE) -> bool:
    return b.x0 - edge <= w.cx <= b.x1 + edge and b.y0 - edge <= w.cy <= b.y1 + edge


def has_ink(rendered: RenderedPage, box: Box, contrast: int = 48) -> bool:
    """Does the rendered page show anything inside the box? Invisible text leaves a flat patch of background."""
    k = 1.0 / rendered.scale
    crop = rendered.image.convert("L").crop(
        (int(box.x0 * k), int(box.y0 * k), int(box.x1 * k) + 1, int(box.y1 * k) + 1)
    )
    low, high = crop.getextrema()
    return high - low >= contrast


def orphan_blocks(items: list[LayoutItem], words: list[PtWord], gap: float = 1.0) -> list[list[list[PtWord]]]:
    """Words whose centre lies in no item, as blocks of lines. A line joins the block above it when the two
    overlap sideways, are under gap x the line height apart and have about the same height."""
    free = [w for w in words if not any(_holds(it.bbox, w) for it in items)]
    blocks: list[list[list[PtWord]]] = []
    for line in lines_of(free):
        for seg in _segments(line, 1.5):
            x0, x1 = seg[0].x0, seg[-1].x1
            top, height = min(w.y0 for w in seg), median(w.y1 - w.y0 for w in seg)
            home = None
            for block in blocks:
                last = block[-1]
                last_height = median(w.y1 - w.y0 for w in last)
                apart = top - max(w.y1 for w in last)
                overlap = min(x1, max(w.x1 for w in last)) - max(x0, min(w.x0 for w in last))
                if overlap > 0 and -0.5 * height <= apart <= gap * height and 0.77 <= height / last_height <= 1.3:
                    home = block
            if home is None:
                blocks.append([seg])
            else:
                home.append(seg)
    return blocks


_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def supported(words: list[PtWord], page_tokens: Counter | None, minimum: float) -> bool:
    """Is the text layer healthy under these words? True when enough of their tokens occur in the model's
    reading of the page. With no reading to compare against, the text layer is trusted."""
    if page_tokens is None or minimum <= 0:
        return True
    mine = [t for w in words for t in tokens(w.text)]
    if not mine:
        return False
    return sum(1 for t in mine if page_tokens.get(t)) / len(mine) >= minimum


def refine(
    items: list[LayoutItem], rendered: RenderedPage, options: dict, page_markdown: str | None = None
) -> list[LayoutItem]:
    """Apply the enabled options; with none enabled, or no text layer, the items come back unchanged."""
    tighten = bool(options.get("tighten"))
    native = bool(options.get("native_text"))
    split_gap = float(options.get("split_gap") or 0.0)
    pad = float(options.get("tighten_pad") or 0.0)
    page = (rendered.width_pt, rendered.height_pt)
    words = words_in_points(rendered)
    orphans = bool(options.get("orphans"))
    if not words or not (tighten or native or split_gap or orphans):
        return items
    owned = assign_words(items, words)
    support = float(options.get("native_support", 0.8))
    page_tokens = Counter(tokens(page_markdown)) if page_markdown is not None else None
    out: list[LayoutItem] = []
    for item, mine in zip(items, owned, strict=True):
        if item.type not in TEXT_TYPES or not mine:
            out.append(item)
            continue
        lines = lines_of(mine)
        parts = split_lines(lines, split_gap) if split_gap and item.type in SPLIT_TYPES else [lines]
        if len(parts) > 1:
            for part in parts:
                flat = [w for line in part for w in line]
                out.append(item.model_copy(update={"bbox": _union(flat, pad, page), "md": _text(part)}))
            continue
        update: dict = {}
        if tighten:
            update["bbox"] = _union(mine, pad, page)
        if native and supported(mine, page_tokens, support):
            update["md"] = _text(lines)
        out.append(item.model_copy(update=update) if update else item)
    if orphans:
        held = [w for mine in owned for w in mine]
        if supported(held or words, page_tokens, support):
            least = int(options.get("orphan_min", 1))
            need = float(options.get("orphan_support") or 0.0)
            taken = [w for w in words if any(_holds(it.bbox, w) for it in items)]
            for block in orphan_blocks(items, words, float(options.get("orphan_gap", 1.0))):
                flat = [w for line in block for w in line]
                box = _union(flat, pad, page)
                if sum(len(t) for w in flat for t in tokens(w.text)) < least:
                    continue
                if need and not supported(flat, page_tokens, need):
                    continue  # the model did not read these words: hidden text, or text the page does not show
                if options.get("orphan_ink") and not has_ink(rendered, _union(flat)):
                    continue  # nothing is drawn there: invisible text
                if options.get("orphan_clear") and any(_holds(box, w, 0.0) for w in taken):
                    continue  # the block wraps around words that already have an item
                out.append(LayoutItem(type="Text", label="text", md=_text(block), bbox=box))
    return out

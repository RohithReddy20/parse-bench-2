"""Layout refinements on pages with a PDF text layer (plan Task 16). Each is one option, measured on its own.

    tighten      a text item's box becomes the union of the PDF words it holds
    native_text  a text item's text becomes those PDF words, in reading order
    split_gap    a text item is split where the vertical gap between two of its lines exceeds
                 split_gap x the item's median line height (each part gets its own words as text)

A word belongs to the smallest text item holding its centre. Pictures and tables are never changed, and an
item holding no PDF words (a scan, text drawn as an image) is left exactly as it was.
"""

from __future__ import annotations

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
    return [PtWord(w.text, w.box[0] * s, w.box[1] * s, w.box[2] * s, w.box[3] * s) for w in rendered.words or []]


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


def _union(words: list[PtWord]) -> Box:
    return Box(x0=min(w.x0 for w in words), y0=min(w.y0 for w in words),
               x1=max(w.x1 for w in words), y1=max(w.y1 for w in words))  # fmt: skip


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


def refine(items: list[LayoutItem], rendered: RenderedPage, options: dict) -> list[LayoutItem]:
    """Apply the enabled options; with none enabled, or no text layer, the items come back unchanged."""
    tighten = bool(options.get("tighten"))
    native = bool(options.get("native_text"))
    split_gap = float(options.get("split_gap") or 0.0)
    words = words_in_points(rendered)
    if not words or not (tighten or native or split_gap):
        return items
    owned = assign_words(items, words)
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
                out.append(item.model_copy(update={"bbox": _union(flat), "md": _text(part)}))
            continue
        update: dict = {}
        if tighten:
            update["bbox"] = _union(mine)
        if native:
            update["md"] = _text(lines)
        out.append(item.model_copy(update=update) if update else item)
    return out

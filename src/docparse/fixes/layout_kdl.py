"""Layout regions from a second detector, KDL-Frontier-Parser-nano-v2 (plan Task 16, step 4), on pages with a
PDF text layer. Each option is measured on its own; the `layout_refine` options run afterwards on the result.

    kdl_replace  the items are rebuilt with OCRia-1's grounding code on KDL's regions instead of heron's
    kdl_words    the items are KDL's regions, each holding the PDF words whose centre it holds
    kdl_add      KDL regions that overlap no existing item become new items
    kdl_split    a Table or Picture item holding several KDL text regions with PDF words that no other item
                 covers, where KDL sees no table or picture of about the same box, is replaced by those KDL
                 regions
    kdl_children a Table item holding several KDL text regions that no other item covers keeps its box, and
                 those regions are added as text items inside it (a tight text box inside a true table is
                 neutral to the scorer)
    kdl_pictures a Picture item holding several KDL pictures that no other item covers, where KDL sees no
                 picture of about the same box, is replaced by the KDL regions inside it (also without text layer)
    kdl_picbox   a Picture item takes the box of the KDL picture it matches (IoU >= 0.5)

A new text item's text is its PDF words in reading order; a region holding no words is dropped, except
pictures (which have no text). On a page without a text layer the items are returned unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from docparse.core.ir import Box, LayoutItem
from docparse.fixes.layout_refine import PtWord, _holds, _text, lines_of, words_in_points
from docparse.models.adapters.heron import Detection
from docparse.pdf.render import RenderedPage

KDL_OPTIONS = ("kdl_replace", "kdl_words", "kdl_add", "kdl_split", "kdl_children", "kdl_pictures", "kdl_picbox")
PICTURES = frozenset({"Picture", "Chart"})
GRAPHIC = PICTURES | {"Table"}
SEGMENT = {"Section-header": "paragraph_title", "Title": "doc_title", "Picture": "image", "Table": "table",
           "Page-header": "header", "Page-footer": "footer"}  # fmt: skip
INSIDE = 0.85  # share of a region's area inside an item for it to count as inside
FREE = 0.1  # largest share of a region's area that may overlap existing items for it to count as uncovered
SAME = 0.5  # IoU at which a KDL table or picture is the same region as an item


@dataclass(frozen=True)
class PtRegion:
    label: str  # Docling class name ("Text", "Title", "Picture", "Chart", "Table", ...)
    box: Box  # PDF points


def regions_in_points(detections: list[Detection], rendered: RenderedPage) -> list[PtRegion]:
    s = rendered.scale
    return [
        PtRegion(d.label, Box(x0=d.box[0] * s, y0=d.box[1] * s, x1=d.box[2] * s, y1=d.box[3] * s)) for d in detections
    ]


def _inter(a: Box, b: Box) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _area(a: Box) -> float:
    return max(0.0, a.width) * max(0.0, a.height)


def _iou(a: Box, b: Box) -> float:
    i = _inter(a, b)
    return i / max(1e-9, _area(a) + _area(b) - i)


def _share(a: Box, b: Box) -> float:
    """Share of a's area inside b."""
    return _inter(a, b) / max(1e-9, _area(a))


def new_item(region: PtRegion, words: list[PtWord]) -> LayoutItem | None:
    """An item for a KDL region: pictures as they are, text-bearing regions with the PDF words they hold."""
    if region.label in PICTURES:
        return LayoutItem(type="Picture", label="image", md="", bbox=region.box)
    mine = [w for w in words if _holds(region.box, w, 0.0)]
    if not mine:
        return None
    return LayoutItem(
        type=region.label, label=SEGMENT.get(region.label, "text"), md=_text(lines_of(mine)), bbox=region.box
    )


def words_as_items(regions: list[PtRegion], words: list[PtWord]) -> list[LayoutItem]:
    return [it for r in regions if (it := new_item(r, words)) is not None]


def add_uncovered(items: list[LayoutItem], regions: list[PtRegion], words: list[PtWord]) -> list[LayoutItem]:
    out = list(items)
    for r in regions:
        if sum(_inter(r.box, it.bbox) for it in items) <= FREE * _area(r.box):
            if (it := new_item(r, words)) is not None:
                out.append(it)
    return out


def split_coarse(items: list[LayoutItem], regions: list[PtRegion], words: list[PtWord]) -> list[LayoutItem]:
    out: list[LayoutItem] = []
    for item in items:
        if item.type in ("Table", "Picture"):
            others = [it for it in items if it is not item]
            inner = [r for r in regions if _share(r.box, item.bbox) >= INSIDE]
            inner = [r for r in inner if sum(_inter(r.box, it.bbox) for it in others) <= FREE * _area(r.box)]
            texts = [r for r in inner if r.label not in GRAPHIC and any(_holds(r.box, w, 0.0) for w in words)]
            same = any(r.label in GRAPHIC and _iou(r.box, item.bbox) >= SAME for r in regions)
            if len(texts) >= 2 and not same:
                parts = words_as_items(inner, words)
                if parts:
                    out.extend(parts)
                    continue
        out.append(item)
    return out


def table_children(items: list[LayoutItem], regions: list[PtRegion], words: list[PtWord]) -> list[LayoutItem]:
    others = [it for it in items if it.type != "Table"]
    out: list[LayoutItem] = []
    for item in items:
        out.append(item)
        if item.type == "Table":
            texts = [r for r in regions if r.label not in GRAPHIC and _share(r.box, item.bbox) >= INSIDE]
            texts = [r for r in texts if sum(_inter(r.box, it.bbox) for it in others) <= FREE * _area(r.box)]
            if len(texts) >= 2:
                out.extend(words_as_items(texts, words))
    return out


def split_pictures(items: list[LayoutItem], regions: list[PtRegion], words: list[PtWord]) -> list[LayoutItem]:
    out: list[LayoutItem] = []
    for item in items:
        if item.type == "Picture":
            others = [it for it in items if it is not item]
            inner = [r for r in regions if _share(r.box, item.bbox) >= INSIDE]
            inner = [r for r in inner if sum(_inter(r.box, it.bbox) for it in others) <= FREE * _area(r.box)]
            pics = [r for r in inner if r.label in PICTURES]
            same = any(r.label in GRAPHIC and _iou(r.box, item.bbox) >= SAME for r in regions)
            if len(pics) >= 2 and not same:
                out.extend(words_as_items(inner, words))
                continue
        out.append(item)
    return out


def picture_boxes(items: list[LayoutItem], regions: list[PtRegion]) -> list[LayoutItem]:
    pics = [r for r in regions if r.label in PICTURES]
    out = []
    for item in items:
        best = max(pics, key=lambda r: _iou(r.box, item.bbox), default=None) if item.type == "Picture" else None
        if best is not None and _iou(best.box, item.bbox) >= SAME:
            item = item.model_copy(update={"bbox": best.box})
        out.append(item)
    return out


def apply_kdl(
    items: list[LayoutItem],
    rendered: RenderedPage,
    detections: list[Detection],
    options: dict,
    page_markdown: str,
) -> list[LayoutItem]:
    """Apply the enabled KDL options; with none enabled, no text layer or no regions, the items come back as is."""
    if not any(options.get(k) for k in KDL_OPTIONS) or not detections:
        return items
    regions = regions_in_points(detections, rendered)
    if options.get("kdl_picbox"):  # pictures need no text layer
        items = picture_boxes(items, regions)
    words = words_in_points(rendered)
    if options.get("kdl_pictures"):
        items = split_pictures(items, regions, words)
    if not words:
        return items
    if options.get("kdl_replace"):
        from docparse.fixes.layout import ground

        dets = [Detection("Picture" if d.label == "Chart" else d.label, d.score, d.box) for d in detections]
        items = ground(page_markdown, rendered, dets) or items
    elif options.get("kdl_words"):
        items = words_as_items(regions, words) or items
    if options.get("kdl_split"):
        items = split_coarse(items, regions, words)
    if options.get("kdl_children"):
        items = table_children(items, regions, words)
    if options.get("kdl_add"):
        items = add_uncovered(items, regions, words)
    return items

"""The layout writer: IR layout items -> the page payload the harness's default layout adapter reads.

Contract (design §5.4): one entry per page with `page_number` starting at 1 and the size in PDF points;
every item carries both `bbox` and `layout_segments`, a segment label, and its own text in `md`.
"""

from __future__ import annotations

from docparse.core.ir import Box, LayoutItem, Page


def _segment(box: Box, label: str) -> dict:
    return {"x": box.x0, "y": box.y0, "w": box.width, "h": box.height, "label": label}


def layout_page(page: Page, page_markdown: str = "") -> dict | None:
    """None when the page has no layout (the harness then scores the page as having none)."""
    if page.layout is None:
        return None
    items = []
    for it in page.layout:
        if it.bbox.width <= 0 or it.bbox.height <= 0:
            continue
        seg = _segment(it.bbox, it.label)
        items.append({"type": it.type, "md": it.md, "value": it.md, "bbox": dict(seg), "layout_segments": [seg]})
    return {
        "page_number": page.index + 1,
        "width": page.width,
        "height": page.height,
        "md": page_markdown,
        "items": items,
    }


def items_from_layout_page(layout: dict) -> list[LayoutItem]:
    """A ParseBench layout page (as published, or as `ocria_grounding` serialises it) -> IR layout items."""
    out = []
    for it in layout.get("items") or []:
        seg = it.get("bbox") or (it.get("layout_segments") or [None])[0]
        if not seg:
            continue
        box = Box(x0=seg["x"], y0=seg["y"], x1=seg["x"] + seg["w"], y1=seg["y"] + seg["h"])
        out.append(
            LayoutItem(type=it.get("type") or "Text", label=seg.get("label") or "text", md=it.get("md") or "", bbox=box)
        )
    return out

import pytest

from docparse.base.ocria_parse import parse_page
from docparse.core.ir import Box, LayoutItem
from docparse.write.layout_items import items_from_layout_page, layout_page
from docparse.write.lint import lint_page


def _page(items):
    page = parse_page("# T\n\nbody", index=2)
    page.width, page.height, page.layout = 612.0, 792.0, items
    return page


def test_layout_page_shape():
    items = [LayoutItem(type="Section-header", label="paragraph_title", md="# T", bbox=Box(x0=24, y0=59.28, x1=387.696, y1=81.288)),
             LayoutItem(type="Picture", label="image", md="", bbox=Box(x0=0, y0=0, x1=10, y1=10)),
             LayoutItem(type="Text", label="text", md="degenerate", bbox=Box(x0=5, y0=5, x1=5, y1=9))]  # fmt: skip
    out = layout_page(_page(items), "# T\n\nbody")
    assert out["page_number"] == 3 and (out["width"], out["height"]) == (612.0, 792.0) and out["md"] == "# T\n\nbody"
    assert len(out["items"]) == 2  # the zero-width box is dropped
    first = out["items"][0]
    assert first["type"] == "Section-header" and first["md"] == first["value"] == "# T"
    assert first["bbox"] == first["layout_segments"][0]
    assert first["bbox"]["label"] == "paragraph_title"
    assert abs(first["bbox"]["w"] - 363.696) < 1e-9 and abs(first["bbox"]["h"] - 22.008) < 1e-9


def test_no_layout_gives_none():
    assert layout_page(_page(None)) is None
    assert layout_page(_page([]))["items"] == []


def test_published_layout_page_round_trips():
    seg = {"x": 24.0, "y": 59.28, "w": 363.696, "h": 22.008, "label": "paragraph_title", "r": None, "confidence": None}
    published = {"page_number": 1, "width": 612.0, "height": 792.24, "items": [
        {"type": "Section-header", "md": "# T", "value": "# T", "bbox": seg, "layout_segments": [seg], "html": "", "regions": []}]}  # fmt: skip
    page = _page(items_from_layout_page(published))
    page.index, page.height = 0, 792.24
    again = layout_page(page)
    box = again["items"][0]["bbox"]
    assert box["label"] == "paragraph_title" and set(box) == {"x", "y", "w", "h", "label"}
    assert [box[k] for k in "xywh"] == pytest.approx([24.0, 59.28, 363.696, 22.008], abs=1e-9)
    assert items_from_layout_page(again) == page.layout


def test_lint_flags_each_trap():
    good_seg = {"x": 0, "y": 0, "w": 1, "h": 1, "label": "text"}
    layout = {"page_number": 2, "items": [
        {"type": "Text", "md": "a", "bbox": good_seg, "layout_segments": []},
        {"type": "Text", "md": "b", "bbox": good_seg, "layout_segments": [{"x": 0, "y": 0, "w": 1, "h": 1}]},
        {"type": "Text", "md": "c", "bbox": good_seg, "layout_segments": [good_seg]}]}  # fmt: skip
    looped = " ".join(f"w{i}" for i in range(40)) + " "
    checks = [f.check for f in lint_page("text <!-- hidden -->\n\n" + looped * 6, layout, page_number=1)]
    assert sorted(checks) == [
        "html_comment",
        "item_without_boxes",
        "looped_text",
        "page_number",
        "segment_without_label",
    ]


def test_lint_clean_page_and_long_unrepeated_text():
    seg = {"x": 0, "y": 0, "w": 1, "h": 1, "label": "text"}
    layout = {"page_number": 1, "items": [{"type": "Text", "md": "a", "bbox": seg, "layout_segments": [seg]}]}
    assert lint_page(" ".join(f"word{i}" for i in range(3000)), layout) == []
    assert lint_page("", None) == []

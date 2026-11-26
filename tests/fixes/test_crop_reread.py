"""Unit tests for layout-guided cropping, band stitching, table matching, and furniture stripping."""

from __future__ import annotations

from PIL import Image

from docparse.fixes.crop_reread import (
    chart_crops,
    column_crops,
    match_cropped_table,
    render_crop,
    stitch_bands,
    strip_furniture_markdown,
    table_crops,
)
from docparse.models.adapters.heron import Detection


def test_table_crops_merges_overlapping_and_attaches_caption() -> None:
    dets = [
        Detection("caption", 0.9, (100, 160, 800, 195)),
        Detection("table", 0.9, (100, 200, 800, 700)),
        Detection("table", 0.8, (110, 210, 790, 690)),
    ]
    crops = table_crops(dets, 1000, 1400)
    assert len(crops) == 1
    assert crops[0].box[1] <= 160  # caption attached above
    assert crops[0].box[3] >= 700


def test_chart_crops_expands_around_picture_and_caption() -> None:
    dets = [
        Detection("picture", 0.9, (150, 300, 850, 900)),
        Detection("caption", 0.85, (150, 240, 800, 285)),
    ]
    crops = chart_crops(dets, 1000, 1400)
    assert len(crops) == 1
    x0, y0, x1, y1 = crops[0].box
    assert x0 < 150 and y0 < 240 and x1 > 850 and y1 > 900


def test_column_crops_detects_clean_two_column_page() -> None:
    dets = [
        Detection("title", 0.9, (80, 50, 920, 120)),
        Detection("text", 0.9, (80, 150, 470, 350)),
        Detection("text", 0.9, (80, 370, 470, 600)),
        Detection("text", 0.9, (80, 620, 470, 900)),
        Detection("text", 0.9, (530, 150, 920, 380)),
        Detection("text", 0.9, (530, 400, 920, 650)),
        Detection("text", 0.9, (530, 670, 920, 920)),
    ]
    crops = column_crops(dets, 1000, 1400)
    labels = [c.label for c in crops]
    assert labels == ["left", "right"]


def test_stitch_bands_deduplicates_overlap_seam() -> None:
    top = "First paragraph here.\nSecond line of text.\nShared boundary sentence across overlap."
    bot = "Shared boundary sentence across overlap.\nFourth line below seam.\nFinal line."
    stitched = stitch_bands([top, bot])
    assert stitched.count("Shared boundary sentence across overlap.") == 1
    assert "First paragraph here." in stitched
    assert "Final line." in stitched


def test_match_cropped_table_finds_best_region() -> None:
    t1 = "<table><tr><th>Assets</th><th>2024</th></tr><tr><td>Cash</td><td>100</td></tr></table>"
    t2 = "<table><tr><th>Region</th><th>Sales</th></tr><tr><td>North</td><td>55</td></tr></table>"
    crop = "<table><tr><th>Region</th><th>Sales</th></tr><tr><td>North</td><td>55.0</td></tr></table>"
    assert match_cropped_table(crop, [("r0", t1), ("r1", t2)]) == "r1"


def test_render_crop_upscales_small_box_and_adds_border() -> None:
    im = Image.new("RGB", (1000, 1400), (255, 255, 255))
    out = render_crop(im, (100, 100, 500, 400), target_long_side=1600, max_upscale=2.0, border=20)
    assert out.size == (840, 640)


def test_strip_furniture_markdown_removes_scan_page_number() -> None:
    dets = [Detection("page_footer", 0.8, (450, 1300, 550, 1360))]
    md = "Body paragraph one.\nBody paragraph two.\n31\n"
    cleaned = strip_furniture_markdown(md, dets, words=None, w=1000, h=1400)
    assert "31" not in cleaned
    assert "Body paragraph one." in cleaned

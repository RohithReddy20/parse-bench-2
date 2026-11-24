"""The layout corrector on generated pages. It runs OCRia-1's own grounding code from the fetched snapshot."""

import pytest

from docparse.fixes.layout import ground, ocria_code_dir
from docparse.models.adapters.heron import Detection
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf

pytestmark = pytest.mark.skipif(
    not (ocria_code_dir() / "ocria_grounding" / "layout.py").exists(),
    reason="OCRia-1 grounding code not fetched (python -m evalkit fetch ocria)",
)

HEADING = "Quarterly results for the northern region"
PARA1 = "Revenue grew by twelve percent compared with the same quarter of the previous year."
PARA2 = "Operating costs were unchanged and the margin improved to nineteen percent overall."


def _page(**page):
    pdf = make_pdf([{"texts": [(72, 720, 16, HEADING), (72, 660, 11, PARA1), (72, 560, 11, PARA2)], **page}])
    return render_pages(pdf)[0]


def _box_around(rendered, text, pad=12):
    first = text.split()[0]
    words = [w for w in rendered.words if w.box[1] == next(x.box[1] for x in rendered.words if x.text == first)]
    return (min(w.box[0] for w in words) - pad, min(w.box[1] for w in words) - pad,
            max(w.box[2] for w in words) + pad, max(w.box[3] for w in words) + pad)  # fmt: skip


def test_each_region_with_text_becomes_one_item_in_points():
    rendered = _page()
    detections = [
        Detection("Section-header", 0.95, _box_around(rendered, HEADING)),
        Detection("Text", 0.93, _box_around(rendered, PARA1)),
        Detection("Text", 0.91, _box_around(rendered, PARA2)),
        Detection("Text", 0.9, (1800.0, 2900.0, 2400.0, 3100.0)),  # a region no answer text falls into
        Detection("Picture", 0.8, (1500.0, 1500.0, 2200.0, 2100.0)),  # pictures are kept without text
    ]
    items = ground(f"# {HEADING}\n\n{PARA1}\n\n{PARA2}", rendered, detections)
    assert [(i.type, i.label) for i in items if i.type != "Picture"] == [
        ("Section-header", "paragraph_title"), ("Text", "text"), ("Text", "text")]  # fmt: skip
    heading, para1, para2 = (i for i in items if i.type != "Picture")
    assert heading.md == f"# {HEADING}" and para1.md == PARA1 and para2.md == PARA2
    assert [i.type for i in items].count("Picture") == 1
    # boxes are PDF points in the displayed frame: the heading baseline is at 792 - 720 = 72 pt from the top
    assert heading.bbox.x0 == pytest.approx(72 - 12 * 72 / 300, abs=1.5)
    assert heading.bbox.y0 < 72 < heading.bbox.y1 + 6
    for item in items:
        assert 0 <= item.bbox.x0 < item.bbox.x1 <= 612 and 0 <= item.bbox.y0 < item.bbox.y1 <= 792
    assert para1.bbox.y1 <= para2.bbox.y0  # reading order follows the answer


def test_empty_answer_and_no_detections():
    rendered = _page()
    assert ground("", rendered, [Detection("Text", 0.9, (0.0, 0.0, 100.0, 100.0))]) == []
    assert ground("   \n", rendered, []) == []
    # with no regions at all, the answer's own blocks still get boxes from their PDF words or are dropped; no crash
    assert isinstance(ground(f"{PARA1}\n\n{PARA2}", rendered, []), list)


def test_large_format_page_converts_with_its_own_scale():
    rendered = _page(size=(5000, 5000))
    assert rendered.dpi < 300
    box = _box_around(rendered, PARA1)
    (item,) = ground(PARA1, rendered, [Detection("Text", 0.9, box)])
    assert item.bbox.x0 == pytest.approx(box[0] * 72 / rendered.dpi, abs=0.05)  # boxes are rounded to 0.1 px
    assert item.bbox.x1 <= 5000

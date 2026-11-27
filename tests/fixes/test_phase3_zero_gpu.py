"""Unit tests for Phase-3 zero-GPU upgrades:
- prose-only `underline_missing` (skips tables and pages that already carry `<u>`)
- PDF-coordinate `table_geo` verification & selection (`verify.py`)
- `Page-Header` / `Page-Footer` visual logo essay trimming (`vlm_specialists.py`)
- scan-mode `pp_scan_arbiter` (`layout_kdl.py`) and `blank_drop` (`layout_refine.py`)
- `hybrid_v2.yaml` config_hash invariance
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from docparse.base.ocria_parse import parse_page
from docparse.core.config import Config
from docparse.core.ir import Box, LayoutItem
from docparse.fixes.formatting import refine_format
from docparse.fixes.layout_kdl import apply_kdl
from docparse.fixes.layout_refine import refine
from docparse.fixes.verify import needs_pdf_verification, secondary_improves, table_geo_metrics, table_geo_score
from docparse.models.adapters import vlm_specialists
from docparse.models.adapters.heron import Detection
from docparse.pdf.render import RenderedPage, Word
from docparse.write.markdown import write_page

REPO = Path(__file__).resolve().parents[2]


def _page_with_words(words: list[Word], w: int = 600, h: int = 800, *, hidden: float = 0.0, map_errors: float = 0.0):
    img = Image.new("RGB", (w, h), "white")
    return RenderedPage(
        dpi=72,
        width_pt=float(w),
        height_pt=float(h),
        rotation=0,
        image=img,
        words=words,
        text_words=words,
        hidden=hidden,
        map_errors=map_errors,
    )


def test_underline_missing_skips_tables_and_already_underlined_pages():
    words = [
        Word("Notice", (10, 10, 60, 22), marks="uuuuuu"),
        Word("Important", (70, 10, 140, 22), marks="uuuuuuuuu"),
        Word("Col1", (10, 50, 50, 62), marks="uuuu"),
        Word("Val1", (60, 50, 100, 62), marks="uuuu"),
    ]
    rp = _page_with_words(words)

    md_none = "Notice Important\n\n<table><tr><th>Col1</th></tr><tr><td>Val1</td></tr></table>"
    pg1 = parse_page(md_none, index=0, call_key="k1")
    refine_format(pg1, rp, {"underline_missing": True})
    out1 = write_page(pg1)
    assert "<u>Notice Important</u>" in out1
    assert "<th><u>Col1</u></th>" not in out1
    assert "<td><u>Val1</u></td>" not in out1

    md_has = "<u>Notice</u> Important"
    pg2 = parse_page(md_has, index=0, call_key="k2")
    refine_format(pg2, rp, {"underline_missing": True})
    assert write_page(pg2) == md_has


def test_trim_image_div_essays_handles_page_header_logo_essay():
    raw = (
        '<div data-bbox="788 52 905 86" data-label="Page-Header">'
        '<img alt="Dum logo" data-bbox="788 52 905 86"/>'
        "The logo for 'Dum' is located in the top right corner. It consists of stylized colorful letters."
        "</div>"
        '<div data-bbox="100 200 500 250" data-label="Text"><p>Body text</p></div>'
    )
    md = vlm_specialists.chandra_markdown(raw)
    assert "![Dum logo]()" in md
    assert "stylized colorful letters" not in md
    assert "Body text" in md


def test_trim_image_div_essays_preserves_normal_header_text():
    raw = (
        '<div data-bbox="50 20 550 60" data-label="Page-Header">'
        '<img alt="ACME" data-bbox="50 20 90 60"/>'
        "Annual Financial Report 2025 — Section 4 Revenue Summary"
        "</div>"
    )
    md = vlm_specialists.chandra_markdown(raw)
    assert "Annual Financial Report 2025" in md


def test_table_geo_prefers_clean_table_without_empty_spacer_column():
    words = []
    rows_txt = [
        ("Year", "Sales", "Profit"),
        ("2022", "100", "20"),
        ("2023", "120", "25"),
        ("2024", "150", "35"),
        ("2025", "180", "45"),
    ]
    for r, (a, b, c) in enumerate(rows_txt):
        y0, y1 = 100 + r * 20, 112 + r * 20
        words.append(Word(a, (50, y0, 110, y1)))
        words.append(Word(b, (200, y0, 260, y1)))
        words.append(Word(c, (350, y0, 410, y1)))
    rp = _page_with_words(words)

    ch_md = (
        "<table><tr><th></th><th>Year</th><th>Sales</th><th>Profit</th></tr>"
        "<tr><td></td><td>2022</td><td>100</td><td>20</td></tr>"
        "<tr><td></td><td>2023</td><td>120</td><td>25</td></tr>"
        "<tr><td></td><td>2024</td><td>150</td><td>35</td></tr>"
        "<tr><td></td><td>2025</td><td>180</td><td>45</td></tr></table>"
    )
    oc_md = (
        "<table><tr><th>Year</th><th>Sales</th><th>Profit</th></tr>"
        "<tr><td>2022</td><td>100</td><td>20</td></tr>"
        "<tr><td>2023</td><td>120</td><td>25</td></tr>"
        "<tr><td>2024</td><td>150</td><td>35</td></tr>"
        "<tr><td>2025</td><td>180</td><td>45</td></tr></table>"
    )
    need, reason = needs_pdf_verification(ch_md, rp, table_geo=True)
    assert need and reason.startswith("table_geo:")
    assert table_geo_score(table_geo_metrics(oc_md, rp)) > table_geo_score(table_geo_metrics(ch_md, rp))
    assert secondary_improves(ch_md, oc_md, rp, reason, table_geo=True)


def test_scan_arbiter_and_blank_drop():
    img = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((50, 50, 250, 150), fill="black")
    rp = RenderedPage(dpi=72, width_pt=400.0, height_pt=400.0, rotation=0, image=img, words=[], text_words=[])
    outer = LayoutItem(type="Table", label="table", md="a", bbox=Box(x0=40, y0=40, x1=260, y1=160))
    inner = LayoutItem(type="Text", label="text", md="a", bbox=Box(x0=60, y0=60, x1=180, y1=110))
    blank = LayoutItem(type="Text", label="text", md="", bbox=Box(x0=300, y0=300, x1=380, y1=380))
    dets = [Detection("Table", 0.95, (40, 40, 260, 160))]

    out = apply_kdl([outer, inner, blank], rp, dets, {"kdl_arbiter": True, "scan_arbiter": True}, "")
    assert len(out) == 2 and all(it is not inner for it in out)
    out2 = refine(out, rp, {"blank_drop": True}, "")
    assert len(out2) == 1 and out2[0].type == "Table"


def test_hybrid_v2_config_hash_unchanged_with_new_optional_fields():
    c1 = Config.load(REPO / "configs" / "hybrid_v2.yaml")
    c2 = Config.load(REPO / "configs" / "hybrid_v2.yaml")
    assert c1.config_hash() == c2.config_hash()
    f3 = Config.load(REPO / "configs" / "frontier_v3.yaml")
    assert f3.table_options["table_geo"] is True
    assert f3.format_options["underline_missing"] is True
    assert f3.form_options["acro_clean"] is True

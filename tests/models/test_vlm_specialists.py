"""Unit tests for specialist VLM ModelCall builders and parsers."""

from __future__ import annotations

from PIL import Image

from docparse.models.adapters.vlm_specialists import (
    CHANDRA_MAX_SIDE,
    OLMOCR_LONG_SIDE,
    build_chandra_call,
    build_chart_call,
    build_chart_verify_call,
    build_olmocr_call,
    chandra_markdown,
    fuse_figure_charts,
    has_chart_figure,
    read_chandra_result,
    read_chart_result,
    read_chart_verify_tables,
    read_olmocr_result,
)
from docparse.models.calls import ModelResult


def _img(w: int = 2000, h: int = 1000) -> Image.Image:
    return Image.new("RGB", (w, h), (255, 255, 255))


def test_chart_call_keys_image_and_hint_and_drops_invented_total() -> None:
    call1, blobs1 = build_chart_call(_img(400, 300), "Axes: 0..100")
    call2, _ = build_chart_call(_img(400, 300), "Axes: 0..200")
    assert call1.key() != call2.key()
    assert len(call1.inputs) == 2
    assert "Axes: 0..100" in blobs1[call1.inputs[1]].decode("utf-8")

    raw = (
        "Title: Revenue Growth\n"
        "Axes: x = Year; y = USD ($M)\n"
        "Legend: North, South\n"
        "<table><tr><th>Year</th><th>North</th></tr>"
        "<tr><td>2023</td><td>10</td></tr>"
        "<tr><td>Total</td><td>10</td></tr></table>"
    )
    res = ModelResult(key=call1.key(), output={"answer": raw, "finish_reason": "stop"})
    parsed = read_chart_result(res, hint="2023=10")
    assert parsed.title == "Revenue Growth"
    assert parsed.table_html is not None
    assert "2023" in parsed.table_html
    assert "Total" not in parsed.table_html


def test_olmocr_call_resizes_and_strips_front_matter() -> None:
    call, blobs = build_olmocr_call(_img(2576, 1288))
    assert call.params["text_first"] is True
    im = Image.open(__import__("io").BytesIO(blobs[call.inputs[0]]))
    assert max(im.size) == OLMOCR_LONG_SIDE

    raw = "---\nprimary_language: en\nis_rotation_valid: true\nrotation_correction: 0\n---\nHello world."
    res = ModelResult(key=call.key(), output={"answer": raw, "finish_reason": "stop"})
    parsed = read_olmocr_result(res)
    assert parsed.front_matter["is_rotation_valid"] is True
    assert parsed.front_matter["rotation_correction"] == 0
    assert parsed.markdown == "Hello world."


def test_chandra_call_converts_html_keeping_tables() -> None:
    call, blobs = build_chandra_call(_img(3600, 1800))
    im = Image.open(__import__("io").BytesIO(blobs[call.inputs[0]]))
    assert max(im.size) == CHANDRA_MAX_SIDE

    raw = (
        '<div data-bbox="0 0 1000 50" data-label="Page-Header">Header 12</div>'
        '<div data-bbox="0 60 1000 900" data-label="Text">'
        '<h1>Report</h1><p>Check: <input type="checkbox" checked> Yes '
        '<input type="checkbox"> No</p>'
        "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        "</div>"
        '<div data-bbox="0 950 1000 1000" data-label="Page-Footer">Page 1</div>'
    )
    res = ModelResult(key=call.key(), output={"answer": raw, "finish_reason": "stop"})
    parsed = read_chandra_result(res)
    assert "Header 12" in parsed.markdown
    assert "# Report" in parsed.markdown
    assert "[x] Yes" in parsed.markdown
    assert "[ ] No" in parsed.markdown
    assert "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>" in parsed.markdown

    stripped = chandra_markdown(raw, drop_furniture=True)
    assert "Header 12" not in stripped
    assert "Page 1" not in stripped
    assert "# Report" in stripped

    tbl_math = (
        "<table><tr><th>Mean <math>\\pm</math> SD</th>"
        '<td><math>6.21 \\pm 0.14</math> <del>old</del> <input type="checkbox" checked></td></tr></table>'
    )
    cleaned_tbl = chandra_markdown(tbl_math)
    assert "<math>" not in cleaned_tbl
    assert "Mean ± SD" in cleaned_tbl
    assert "6.21 ± 0.14" in cleaned_tbl
    assert "~~old~~" in cleaned_tbl
    assert "[x]" in cleaned_tbl

    fmt_html = "<ul><li>• item</li></ul><p><del>old</del> and <u>important</u> on 1<sup>st</sup> CO<sub>2</sub></p>"
    fmt_md = chandra_markdown(fmt_html)
    assert "- item" in fmt_md and "•" not in fmt_md
    assert "~~old~~" in fmt_md
    assert "<u>important</u>" in fmt_md
    assert "1st" in fmt_md
    assert "CO2" in fmt_md


def test_chart_verify_call_and_table_extraction() -> None:
    draft = "<table><tr><th>Year</th><th>A</th></tr><tr><td>2020</td><td>10</td></tr></table>"
    call1, blobs1 = build_chart_verify_call(_img(3000, 2000), draft, "2020: 10.4")
    call2, _ = build_chart_verify_call(_img(3000, 2000), draft, "2020: 11.2")
    assert call1.key() != call2.key()
    im = Image.open(__import__("io").BytesIO(blobs1[call1.inputs[0]]))
    assert max(im.size) == 1800
    prompt_txt = blobs1[call1.inputs[1]].decode("utf-8")
    assert "<draft_tables>" in prompt_txt and "2020: 10.4" in prompt_txt

    raw = (
        "Chart 1\n"
        "<table><tr><th>Year</th><th>A</th></tr>"
        "<tr><td>2020</td><td>10.4</td></tr>"
        "<tr><td>Total</td><td>10.4</td></tr></table>\n"
        "Chart 2\n"
        "<table><tr><th>Year</th><th>B</th></tr><tr><td>2020</td><td>5.0</td></tr></table>"
    )
    res = ModelResult(key=call1.key(), output={"answer": raw, "finish_reason": "stop"})
    tables = read_chart_verify_tables(res, hint="2020: 10.4")
    assert len(tables) == 2
    assert "10.4" in tables[0]
    assert "Total" not in tables[0]
    assert "5.0" in tables[1]


def test_fuse_figure_charts_injects_missing_and_preserves_inset_tables() -> None:
    # 1. Chart Figure with alt-text only -> injects secondary chart table
    raw_no_tbl = (
        '<div data-label="Page-Header">Header</div>'
        '<div data-label="Text"><p>Intro text</p></div>'
        '<div data-label="Figure"><img alt="Bar chart of GDP growth by country"/></div>'
    )
    assert has_chart_figure(raw_no_tbl)
    chart_tbl = "<table><tr><th>Country</th><th>GDP</th></tr><tr><td>US</td><td>2.5</td></tr></table>"
    md, fused = fuse_figure_charts(raw_no_tbl, [chart_tbl], drop_furniture=True)
    assert fused is True
    assert "Header" not in md
    assert "Intro text" in md
    assert "<table><tr><th>Country</th>" in md

    # 2. Raster Figure with printed inset table (no vector measurement) -> preserves Chandra's inset table
    raw_inset = (
        '<div data-label="Figure">'
        '<img alt="ROC curve plot with performance metrics"/>'
        "<table><tr><td>accuracy</td><td>95%</td></tr><tr><td>MCC</td><td>0.90</td></tr></table>"
        "</div>"
    )
    other_tbl = "<table><tr><td>bad</td><td>10%</td></tr></table>"
    md2, fused2 = fuse_figure_charts(raw_inset, [other_tbl], ocria_vector_measured=False)
    assert fused2 is False
    assert "95%" in md2 and "bad" not in md2

    # 3. Same Figure when PDF vector measurement improved the secondary table -> fuses vector table
    md3, fused3 = fuse_figure_charts(raw_inset, [chart_tbl], ocria_vector_measured=True)
    assert fused3 is True
    assert "GDP" in md3 and "95%" not in md3

"""Regression tests for real outside-benchmark bugs found in the 2026-11-26 audit."""

from __future__ import annotations

from docparse.fixes.fallback import has_math, trim_loop_tail
from docparse.fixes.tables import _insert_empties, _leaf_header_spans, right_align_leaf
from docparse.fixes.tables_html import parse_table
from docparse.fixes.text import _text_tokens


def test_respell_skips_latex_math_but_keeps_currency() -> None:
    raw = (
        "The fee is $500 and the cap is $1,000 for valid claims.\n\n"
        "$$(x + 1)^{p^d} \\equiv x^{p^d} + 1 \\pmod p.$$\n\n"
        "Inline $I(C_n^{\\boxtimes d}, x) \\equiv (x+1)^m \\pmod{p}$ holds."
    )
    tokens = [tok for _s, _e, tok in _text_tokens(raw)]
    assert "500" in tokens
    assert "1,000" in tokens or ("1" in tokens and "000" in tokens)
    assert "pmod" not in tokens
    assert "equiv" not in tokens
    assert "boxtimes" not in tokens


def test_trim_loop_tail_strips_scanned_table_td_loop_and_closes_table() -> None:
    header = "Form 5500\n<table><tr><td>Name</td><td>Midwest Trust</td></tr>"
    looped = header + "<tr><td>" + ("</td><td>" * 2000)
    trimmed = trim_loop_tail(looped)
    assert len(trimmed) < len(header) + 200
    assert "Midwest Trust" in trimmed
    assert trimmed.endswith("</table>")


def test_trim_loop_tail_strips_latex_quad_loop_preserving_math() -> None:
    body = "A **subrepresentation** $V \\longrightarrow W$ satisfies\n\n$$\\pi_1: V \\longrightarrow W.$$\n\n" + ("\\quad " * 2000)
    trimmed = trim_loop_tail(body)
    assert "\\pi_1: V \\longrightarrow W" in trimmed
    assert trimmed.count("\\quad") <= 1
    assert has_math(trimmed)


def test_right_align_leaf_does_not_shift_single_row_header() -> None:
    raw = (
        "<table><tr><th>Fund Type</th><th>5 Year Return %</th><th>Diversity</th></tr>"
        "<tr><td>Balanced</td><td>13.6</td><td>C</td><td>extra</td></tr></table>"
    )
    table = parse_table(raw)
    assert table is not None
    assert right_align_leaf(table, 3) is False


def test_regrid_widens_colspan_under_spanning_header_instead_of_empty_cell() -> None:
    raw = (
        '<table><tr><th colspan="4">Course Year</th></tr>'
        '<tr><th colspan="2">Learning activity</th><th>Ects</th><th>Sector</th></tr>'
        '<tr><td colspan="2">Economics</td><td>12</td><td>SECS</td></tr>'
        "<tr><td>Conflict Management</td><td>9</td><td>M-PSI</td></tr></table>"
    )
    table = parse_table(raw)
    assert table is not None
    spans = _leaf_header_spans(table)
    assert spans[0] == (0, 2)
    row = table.rows[3]
    target = {id(row.cells[1]): 2, id(row.cells[2]): 3}
    new_cells = _insert_empties(row.cells, target, set(), spans)
    assert new_cells is not None
    assert len(new_cells) == 3
    assert new_cells[0].colspan == 2
    assert new_cells[0].text == "Conflict Management"
    assert new_cells[1].text == "9"


def test_chandra_trims_image_div_essays_and_closes_micro_units() -> None:
    from docparse.models.adapters.vlm_specialists import chandra_markdown

    raw = (
        '<div data-label="Image"><img alt="Company logo"/>The logo features a stylized blue triangle '
        "above bold text.</div>\n"
        '<div data-label="Text"><p>Real body text.</p></div>\n'
        '<div data-label="Table"><table><tr><td><math>250 \\mu M</math></td>'
        "<td><math>I_D</math></td></tr></table></div>"
    )
    md = chandra_markdown(raw)
    assert "stylized blue triangle" not in md
    assert "![Company logo]()" in md
    assert "Real body text." in md
    assert "250 μM" in md
    assert "I<sub>D</sub>" in md


def test_strikes_masks_html_attributes_and_recovers_reordered_pdf_footer() -> None:
    from types import SimpleNamespace

    from docparse.base.ocria_parse import parse_page
    from docparse.fixes.formatting import refine_format

    md = (
        "Header line\n\n"
        '<table border="1" style="width: 100%;"><tr><td colspan="7" style="text-align: center;">'
        "Fall 2022 and Winter 2023</td></tr></table>\n\n"
        "[Proposed] Order 8"
    )
    pg = parse_page(md, index=0, call_key="k")
    # PDF stream stores footer BEFORE body (out of visual reading order)
    words = [
        SimpleNamespace(text="[Proposed]", marks=".ssssssss."),
        SimpleNamespace(text="Order", marks="....."),
        SimpleNamespace(text="8", marks="."),
        SimpleNamespace(text="Header", marks="......"),
        SimpleNamespace(text="line", marks="...."),
        SimpleNamespace(text="Fall", marks="...."),
        SimpleNamespace(text="2022", marks="...."),
        SimpleNamespace(text="and", marks="sss"),
        SimpleNamespace(text="Winter", marks="ssssss"),
        SimpleNamespace(text="2023", marks="ssss"),
    ]
    refine_format(pg, SimpleNamespace(text_words=words), {"strike": True})
    full = "\n\n".join(r.chosen().raw for r in pg.regions)
    assert "<s>and Winter 2023</s>" in full
    assert "[~~Proposed~~] Order 8" in full


def test_chandra_layout_boxes_and_three_way_arbiter() -> None:
    from docparse.core.ir import Box, LayoutItem
    from docparse.fixes.layout_kdl import PtRegion, arbitrate
    from docparse.models.adapters.vlm_specialists import chandra_layout_boxes

    raw = (
        '<div data-bbox="100 200 900 600" data-label="Table"><table><tr><td>A</td></tr></table></div>\n'
        '<div data-label="Figure" data-bbox="100 650 500 900"><img alt="Chart"/></div>'
    )
    boxes = chandra_layout_boxes(raw, 600.0, 800.0)
    assert len(boxes) == 2
    assert boxes[0][0] == "Table" and boxes[0][1] == (60.0, 160.0, 540.0, 480.0)
    assert boxes[1][0] == "Picture"

    # Outer table item matched by Chandra (not KDL), inner text matched by KDL -> 2-of-3 keeps the Table!
    tbl = LayoutItem(type="Table", label="table", md="", bbox=Box(x0=60, y0=160, x1=540, y1=480))
    txt = LayoutItem(type="Text", label="text", md="A", bbox=Box(x0=80, y0=180, x1=300, y1=260))
    kdl_regs = [PtRegion("Text", Box(x0=80, y0=180, x1=300, y1=260))]
    ch_regs = [PtRegion(lbl, Box(x0=b[0], y0=b[1], x1=b[2], y1=b[3])) for lbl, b in boxes]
    # Without Chandra, KDL alone drops the nested Table
    assert [it.type for it in arbitrate([tbl, txt], kdl_regs)] == ["Text"]
    # With Chandra as 3rd voter, both are matched so the true Table is preserved
    assert [it.type for it in arbitrate([tbl, txt], kdl_regs, extra_regions=ch_regs)] == ["Table", "Text"]


def test_pdf_self_verification_flags_low_recall_and_flattened_table() -> None:
    from types import SimpleNamespace

    from docparse.fixes.layout_refine import PtWord
    from docparse.fixes.verify import needs_pdf_verification, pdf_looks_tabular, pdf_word_recall

    words = [
        PtWord(f"r{r}c{c}", c * 120.0, r * 18.0, c * 120.0 + 40.0, r * 18.0 + 10.0)
        for r in range(10)
        for c in range(6)
    ]
    assert pdf_looks_tabular(words) is True
    assert pdf_word_recall("r0c0 r0c1", words) < 0.1
    rp = SimpleNamespace(hidden=0.0, map_errors=0.0, scale=1.0, text_words=None, words=None)
    # Build mock rendered words (in px at scale 1.0)
    rp.text_words = [SimpleNamespace(text=w.text, box=(w.x0, w.y0, w.x1, w.y1)) for w in words]
    need, reason = needs_pdf_verification("r0c0 r0c1", rp)
    assert need and reason.startswith("low_recall:")
    flat_all = " ".join(w.text for w in words)
    need2, reason2 = needs_pdf_verification(flat_all, rp)
    assert need2 and reason2.startswith("missing_table:")
    need3, _ = needs_pdf_verification(f"<table><tr><td>{flat_all}</td></tr></table>", rp)
    assert not need3


def test_chandra_promotes_section_header_and_unwraps_overwide_strike() -> None:
    from types import SimpleNamespace

    from docparse.base.ocria_parse import parse_page
    from docparse.fixes.formatting import refine_format
    from docparse.models.adapters.vlm_specialists import chandra_markdown

    raw = '<div data-label="Section-Header"><p><b>PART I. FINANCIAL INFORMATION</b></p></div>'
    assert chandra_markdown(raw) == "## **PART I. FINANCIAL INFORMATION**"

    pg = parse_page("~~Old clause stays and new clause added~~", index=0, call_key="k")
    words = [
        SimpleNamespace(text="Old", marks="sss"),
        SimpleNamespace(text="clause", marks="ssssss"),
        SimpleNamespace(text="stays", marks="....."),
        SimpleNamespace(text="and", marks="..."),
        SimpleNamespace(text="new", marks="..."),
        SimpleNamespace(text="clause", marks="......"),
        SimpleNamespace(text="added", marks="....."),
    ]
    refine_format(pg, SimpleNamespace(text_words=words), {"strike": True})
    assert pg.regions[0].chosen().raw == "~~Old clause~~ stays and new clause added"


def test_chandra_heading_breaks_stay_on_one_line() -> None:
    from docparse.models.adapters.vlm_specialists import chandra_markdown

    assert chandra_markdown("<h2>Two<br>lines</h2>") == "## Two lines"
    assert chandra_markdown("<h6>Two \r\n <br/> lines</h6>") == "###### Two lines"
    assert chandra_markdown("<h1>Two&#10;lines</h1>") == "# Two lines"


def test_chandra_preserves_block_breaks_in_inferred_section_headers() -> None:
    from docparse.models.adapters.vlm_specialists import chandra_markdown

    wrapper = '<div data-label="Section-Header">{}</div>'
    assert chandra_markdown(wrapper.format("<p>Instructions<br>For submissions<br>Use this form</p>")) == (
        "Instructions\nFor submissions\nUse this form"
    )
    assert chandra_markdown(wrapper.format("<p>First paragraph</p><p>Second paragraph</p>")) == (
        "First paragraph\n\nSecond paragraph"
    )
    assert chandra_markdown(wrapper.format("<ul><li>Water quality</li></ul>")) == "- Water quality"


def test_chandra_serializes_nested_ordered_and_unordered_lists() -> None:
    from docparse.models.adapters.vlm_specialists import chandra_markdown

    raw = (
        "<ol start='3'>\n<li><b>Overview</b><ul>\n"
        "<li>• <i>Value</i> <math>x^2</math> <u>check</u> <del>old</del> <code>run()</code> "
        '<a href="/guide">guide</a><ol><li>Detail</li></ol></li>\n'
        "<li>Next</li></ul>Summary</li>\n<li>Conclusion</li></ol>"
    )
    assert chandra_markdown(raw) == (
        "3. **Overview**\n"
        "   - *Value* $x^2$ <u>check</u> ~~old~~ `run()` [guide](/guide)\n"
        "     1. Detail\n"
        "   - Next\n"
        "   Summary\n"
        "4. Conclusion"
    )


def test_chandra_does_not_duplicate_explicit_decimal_list_markers() -> None:
    from docparse.models.adapters.vlm_specialists import chandra_markdown

    raw = "<p>Before</p><ol start='4'><li>4. First</li><li>5. Second</li><li>2024. Year</li></ol><p>After</p>"
    assert chandra_markdown(raw) == "Before\n\n4. First\n5. Second\n6. 2024. Year\n\nAfter"


def test_chandra_keeps_ordered_list_numbering_that_markdown_cannot_express() -> None:
    from docparse.models.adapters.vlm_specialists import chandra_markdown

    roman = '<ol type="i"><li>Read the <i>guide</i>.</li><li>Complete the form.</li></ol>'
    nested = f"<ul><li>Instructions{roman}</li></ul>"
    reversed_list = "<ol reversed><li>Last</li><li>First</li></ol>"
    assert chandra_markdown(f"<p>Before</p>{roman}<p>After</p>") == f"Before\n\n{roman}\n\nAfter"
    assert chandra_markdown(nested) == nested
    assert chandra_markdown(reversed_list) == reversed_list
    table = f"<table><tr><td>{roman}</td></tr></table>"
    assert chandra_markdown(table) == table


def test_fuse_figure_charts_keeps_all_panels_of_a_single_figure() -> None:
    from docparse.models.adapters.vlm_specialists import fuse_figure_charts

    prose = "<table><tr><th>Glossary</th><td>Term</td></tr></table>"
    north = "<table><tr><th>North</th><td>23</td></tr></table>"
    south = "<table><tr><th>South</th><td>46</td></tr></table>"
    digits = "<table><tr><td>1</td><td>2</td></tr></table>"
    raw = f'{prose}<div data-label="Figure"><img alt="Bar chart with two panels"/></div>'
    md, fused = fuse_figure_charts(raw, [north, prose, south, digits])
    assert fused
    assert md.count(prose) == 1
    assert md.index(prose) < md.index(north) < md.index(south) < md.index(digits)


def test_fuse_figure_charts_assigns_multiple_panels_and_keeps_original_insets() -> None:
    from docparse.models.adapters.vlm_specialists import fuse_figure_charts

    inset = "<table>\n<tr><td>Calibration</td><td><math>0.95</math></td></tr>  \n</table>"
    original = "<table><tr><th>Ridership</th><td>20</td></tr></table>"
    unmatched = "<table>\n<tr><th>Flights</th><td>12</td></tr>  \n</table>"
    north = "<table><tr><th>Ridership North</th><td>23</td></tr></table>"
    south = "<table><tr><th>Ridership South</th><td>46</td></tr></table>"
    rainfall = "<table><tr><th>Rainfall</th><td>42</td></tr></table>"
    raw = (
        f'<div data-label="Figure"><img alt="Ridership chart"/>{inset}{original}</div>'
        '<div data-label="Figure"><img alt="Rainfall plot"/></div>'
        f'<div data-label="Figure"><img alt="Flights chart"/>{unmatched}</div>'
    )
    md, fused = fuse_figure_charts(raw, [north, rainfall, south])
    assert fused
    assert original not in md
    parts = [inset, north, south, "Rainfall plot", rainfall, "Flights chart", unmatched]
    assert [md.index(part) for part in parts] == sorted(md.index(part) for part in parts)


def test_fuse_figure_charts_preserves_an_unmatched_inset_table() -> None:
    from docparse.models.adapters.vlm_specialists import fuse_figure_charts

    inset = "<table>\n  <tr><td>Calibration</td><td><math>0.95</math></td></tr>  \n</table>"
    chart = "<table><tr><th>Month</th><th>Rainfall</th></tr><tr><td>May</td><td>42</td></tr></table>"
    raw = (
        f'<div data-label="Figure"><img alt="Calibration plot"/>{inset}</div>'
        '<div data-label="Figure"><img alt="Bar chart"/><div data-label="Caption">Monthly Rainfall</div>'
        "<table><tr><th>Period</th><th>Reading</th></tr><tr><td>April</td><td>99</td></tr></table></div>"
    )
    for vector_measured in (False, True):
        md, fused = fuse_figure_charts(raw, [chart], ocria_vector_measured=vector_measured)
        assert fused
        assert md.count("<table>") == 2
        assert md.index("Calibration plot") < md.index(inset) < md.index("Bar chart") < md.index(chart)


def test_fuse_figure_charts_prefers_overlap_and_limits_positional_fallback() -> None:
    from docparse.models.adapters.vlm_specialists import fuse_figure_charts

    first = '<div data-label="Figure"><img alt="Ridership chart"/></div>'
    second = """<div data-label="Figure"><img alt='Rainfall chart'/></div>"""
    ridership = "<table><tr><th>Ridership</th><td>25</td></tr></table>"
    rainfall = "<table><tr><th>Rainfall</th><td>70</td></tr></table>"
    unrelated = "<table><tr><th>Capacity</th><td>96</td></tr></table>"
    md, fused = fuse_figure_charts(first + second, [rainfall, ridership])
    assert fused
    assert md.index("Ridership chart") < md.index(ridership) < md.index("Rainfall chart") < md.index(rainfall)
    md, fused = fuse_figure_charts(first + second, [unrelated, rainfall])
    assert fused
    assert md.index("Ridership chart") < md.index(unrelated) < md.index("Rainfall chart") < md.index(rainfall)
    md, fused = fuse_figure_charts(first.replace("</div>", ridership + "</div>") + second, [rainfall])
    assert fused
    assert md.index(ridership) < md.index("Rainfall chart") < md.index(rainfall)
    inset = "<table><tr><td>Calibration</td><td>95%</td></tr></table>"
    first = first.replace("</div>", inset + "</div>")
    md, fused = fuse_figure_charts(first + second, [unrelated, rainfall])
    assert fused
    assert inset in md and rainfall in md and unrelated not in md


def test_weak_table_rejects_tiny_secondary_fragment_and_skips_forms() -> None:
    from types import SimpleNamespace

    from docparse.fixes.verify import needs_pdf_verification, secondary_improves, table_alignment_stats

    words = [
        SimpleNamespace(text=f"r{r}c{c}", box=(c * 100.0, r * 18.0, c * 100.0 + 40.0, r * 18.0 + 10.0))
        for r in range(10)
        for c in range(6)
    ]
    page = SimpleNamespace(hidden=0.0, map_errors=0.0, scale=1.0, text_words=words, words=words)
    body = " ".join(w.text for w in words)
    bad_rows = "".join(f"<tr><td>r{r}c0</td><td>Unknown</td><td>Missing</td></tr>" for r in range(10))
    fragment_rows = "".join(f"<tr><td>r{r}c0</td><td>r{r}c1</td></tr>" for r in range(5))
    primary = f"{body}\n\n<table>{bad_rows}</table>"
    fragment = f"{body}\n\n<table>{fragment_rows}</table>"
    assert table_alignment_stats(primary, page)[1:3] == (10, 30)
    assert table_alignment_stats(fragment, page)[1:3] == (10, 10)
    need, reason = needs_pdf_verification(primary, page)
    assert need and reason.startswith("weak_table:")
    assert not secondary_improves(primary, fragment, page, reason)
    checkboxes = " [ ] [x] <input type='checkbox'>"
    assert needs_pdf_verification(primary + checkboxes, page) == (need, reason)
    assert needs_pdf_verification(primary + checkboxes + " [X]", page) == (False, "")


def test_verify_weak_table_alignment_and_ppdoclayout_adapter() -> None:
    from types import SimpleNamespace

    from docparse.fixes.verify import needs_pdf_verification, secondary_improves
    from docparse.models.adapters import ppdoclayout
    from docparse.models.calls import ModelResult

    words = [
        SimpleNamespace(text=f"r{r}c{c}", box=(c * 100.0, r * 18.0, c * 100.0 + 40.0, r * 18.0 + 10.0))
        for r in range(10)
        for c in range(6)
    ]
    rp = SimpleNamespace(hidden=0.0, map_errors=0.0, scale=1.0, text_words=words, words=words)
    # Mangled table keeps words (high page recall) but mis-groups cells across columns so PDF alignment is low
    bad_rows = "".join(
        f"<tr><td>r{r}c0 r{r}c2</td><td>r{r}c1 r{r}c3</td><td>r{r}c4 r{r}c5</td></tr>" for r in range(10)
    )
    good_rows = "".join(
        "<tr>" + "".join(f"<td>r{r}c{c}</td>" for c in range(6)) + "</tr>" for r in range(10)
    )
    bad_md = f"<table>{bad_rows}</table>"
    good_md = f"<table>{good_rows}</table>"
    need, reason = needs_pdf_verification(bad_md, rp)
    assert need and reason.startswith("weak_table:")
    assert secondary_improves(bad_md, good_md, rp, reason) is True
    assert secondary_improves(bad_md, bad_md, rp, reason) is False

    res = ModelResult(
        key="k",
        output={
            "detections": [
                {"label": "chart", "score": 0.91, "box": [10, 20, 200, 300]},
                {"label": "paragraph_title", "score": 0.60, "box": [10, 5, 100, 18]},
                {"label": "text", "score": 0.28, "box": [10, 310, 200, 350]},
            ]
        },
    )
    dets = ppdoclayout.read_detections(res, min_score=0.40)
    assert [(d.label, d.score) for d in dets] == [("Chart", 0.91), ("Section-header", 0.6)]
    raw = ppdoclayout.read_raw_detections(res, min_score=0.40)
    assert [d.label for d in raw] == ["chart", "paragraph_title"]

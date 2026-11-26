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

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

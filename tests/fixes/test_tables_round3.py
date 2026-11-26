"""Round-three table options: squeeze, sign, head_fill (cells and columns the page does not print)."""

from docparse.base.ocria_parse import parse_page
from docparse.fixes.tables import fix_tables
from docparse.pdf.render import render_pages
from docparse.write.markdown import write_page
from tests.pdfgen import make_pdf


def _page(texts):
    return render_pages(make_pdf([{"texts": [(x, y, 10, t) for x, y, t in texts]}]))[0]


def _fixed(md, rendered, options):
    page = parse_page(md)
    fix_tables(page, rendered, {"regrid": True, "regrid_wide": True, **options})
    return write_page(page)


def _row(*cells, tag="td"):
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


# --- squeeze: a value written once more than its line shows it --------------------------------------------------

XS = (72, 170, 230, 290, 350)
HEAD = ("Port", "Jan", "Feb", "Mar", "Total")
ROWS = [
    ("Alpha", "10.5", "11.5", "12.5", "34.5"),
    ("Beta", "n.a.", "n.a.", "n.a.", "7.5"),
    ("Gamma", "1.5", "2.5", "3.5", "7.5"),
]


def _grid_texts(head=HEAD, rows=ROWS, xs=XS):
    texts = [(x, 700, t) for x, t in zip(xs, head, strict=True)]
    for i, row in enumerate(rows):
        texts += [(x, 680 - 15 * i, t) for x, t in zip(xs, row, strict=True)]
    return texts


def _table(rows, head=HEAD):
    return "<table>" + _row(*head, tag="th") + "".join(_row(*r) for r in rows) + "</table>"


ONCE_TOO_OFTEN = _table([ROWS[0], ("Beta", "n.a.", "n.a.", "n.a.", "n.a.", "7.5"), ROWS[2]])


def test_a_value_written_once_more_than_its_line_shows_it_is_dropped():
    rendered = _page(_grid_texts())
    assert _fixed(ONCE_TOO_OFTEN, rendered, {"squeeze": True}) == _table(ROWS)
    assert _fixed(ONCE_TOO_OFTEN, rendered, {}) == ONCE_TOO_OFTEN  # off by default


def test_a_cell_the_line_may_print_in_other_words_is_kept():
    # the row's line holds words the reading did not match ("see note 4"): the extra cell may be those
    texts = [*_grid_texts(), (410, 665, "see"), (430, 665, "note"), (455, 665, "4")]
    md = _table([ROWS[0], (*ROWS[1], "see note four"), ROWS[2]])
    assert _fixed(md, _page(texts), {"squeeze": True}) == md
    # the same when those words stand on the line below (a cell that wraps)
    texts = [*_grid_texts(), (410, 650, "see"), (430, 650, "note"), (455, 650, "4")]
    assert _fixed(md, _page(texts), {"squeeze": True}) == md


def test_a_cell_between_two_neighbouring_columns_goes_but_a_row_with_a_gap_as_well_is_kept():
    # "9.9" stands between the cells found in the Jan and Feb columns, where the line prints nothing
    md = _table([ROWS[0], ("Beta", "n.a.", "9.9", "n.a.", "n.a.", "7.5"), ROWS[2]])
    assert _fixed(md, _page(_grid_texts()), {"squeeze": True}) == _table(ROWS)
    # two cells not on the page and one column short elsewhere: not a row with cells too many only
    md = _table([ROWS[0], ("Beta", "n.a.", "9.9", "8.8", "n.a.", "7.5"), ROWS[2]])
    assert _fixed(md, _page(_grid_texts()), {"squeeze": True}) == md


def test_cells_past_the_last_column_that_the_line_does_not_print_are_dropped():
    md = _table(ROWS, head=(*HEAD, "Jan", "Feb"))
    assert _fixed(md, _page(_grid_texts()), {"squeeze": True}) == _table(ROWS)


# --- squeeze: two grid columns in one printed column ------------------------------------------------------------

WIDE = (
    "<table><tr><th rowspan='2'>Port</th><th colspan='4'>Quarter</th><th rowspan='2'>Total</th></tr>"
    + _row("Jan", "Feb", "Mar", tag="th")
    + _row("Alpha", "10.5", "11.5", "12.5", "", "34.5")
    + _row("Beta", "0", "0", "0", "0", "7.5")
    + _row("Gamma", "1.5", "2.5", "", "3.5", "8.5")
    + _row("Delta", "0", "0", "", "0", "9.5")
    + "</table>"
)


def test_two_grid_columns_in_one_printed_column_become_one_column():
    rows = [
        ROWS[0],
        ("Beta", "0", "0", "0", "7.5"),
        ("Gamma", "1.5", "2.5", "3.5", "8.5"),
        ("Delta", "0", "0", "0", "9.5"),
    ]
    texts = [(72, 700, "Port"), (230, 712, "Quarter"), (350, 700, "Total")]
    texts += [(x, 700, t) for x, t in zip(XS[1:4], HEAD[1:4], strict=True)]
    for i, row in enumerate(rows):
        texts += [(x, 680 - 15 * i, t) for x, t in zip(XS, row, strict=True)]
    out = _fixed(WIDE, _page(texts), {"squeeze": True})
    assert "<th colspan='3'>Quarter</th>" in out or '<th colspan="3">Quarter</th>' in out
    for row in rows:
        assert _row(*row) in out
    assert out.count("<td></td>") == 0


def test_the_table_stays_as_it_is_when_a_row_cannot_be_brought_to_the_narrower_width():
    # "Beta" holds a value in each of the two grid columns and neither is a copy of its neighbour
    md = WIDE.replace(_row("Beta", "0", "0", "0", "0", "7.5"), _row("Beta", "0", "0", "0", "5", "7.5"))
    rows = [
        ROWS[0],
        ("Beta", "0", "0", "0", "7.5"),
        ("Gamma", "1.5", "2.5", "3.5", "8.5"),
        ("Delta", "0", "0", "0", "9.5"),
    ]
    texts = [(72, 700, "Port"), (230, 712, "Quarter"), (350, 700, "Total")]
    texts += [(x, 700, t) for x, t in zip(XS[1:4], HEAD[1:4], strict=True)]
    for i, row in enumerate(rows):
        texts += [(x, 680 - 15 * i, t) for x, t in zip(XS, row, strict=True)]
    texts.append((320, 665, "5"))  # printed between the columns: on the page, not a copy
    assert _fixed(md, _page(texts), {"squeeze": True}) == md


# --- sign: a currency sign in a column of its own, for which the header has no cell -----------------------------

SIGN_XS = (72, 150, 170, 250, 270, 350)
SIGN_ROWS = [("Lease", "R", "52,333.13", "R", "1,239.50", "2025/09/30"), ("Cleaning", "R", "45,204.00", "R", "-", "2025/10/01"),
             ("Security", "R", "3,105.00", "R", "1,552.50", "2025/10/02")]  # fmt: skip
SIGN_HEAD = ("Project", "Value", "Previous", "Start")
SIGNS = "<table>" + _row(*SIGN_HEAD, tag="th") + "".join(_row(*r) for r in SIGN_ROWS) + "</table>"


def _sign_texts():
    texts = [(x, 700, t) for x, t in zip((72, 170, 270, 350), SIGN_HEAD, strict=True)]
    for i, row in enumerate(SIGN_ROWS):
        texts += [(x, 680 - 15 * i, t) for x, t in zip(SIGN_XS, row, strict=True)]
    return texts


def test_a_currency_sign_column_without_a_header_joins_its_amounts():
    out = _fixed(SIGNS, _page(_sign_texts()), {"sign": True})
    assert _row(*SIGN_HEAD, tag="th") in out
    assert _row("Lease", "R 52,333.13", "R 1,239.50", "2025/09/30") in out
    assert _row("Cleaning", "R 45,204.00", "R -", "2025/10/01") in out
    assert _row(*SIGN_ROWS[0]) in _fixed(SIGNS, _page(_sign_texts()), {})  # off by default


def test_a_sign_column_with_a_header_of_its_own_stays_a_column():
    head = ("Project", "Ccy", "Value", "Ccy", "Previous", "Start")
    md = "<table>" + _row(*head, tag="th") + "".join(_row(*r) for r in SIGN_ROWS) + "</table>"
    texts = _sign_texts() + [(150, 700, "Ccy"), (250, 700, "Ccy")]
    assert _fixed(md, _page(texts), {"sign": True}) == md


def test_codes_that_differ_from_row_to_row_are_not_signs():
    rows = [("Lease", "A", "52,333.13", "R", "1,239.50", "2025/09/30"), *SIGN_ROWS[1:]]
    md = "<table>" + _row(*SIGN_HEAD, tag="th") + "".join(_row(*r) for r in rows) + "</table>"
    texts = [t for t in _sign_texts() if t != (150, 680, "R")] + [(150, 680, "A")]
    out = _fixed(md, _page(texts), {"sign": True})
    assert "<td>A</td><td>52,333.13</td>" in out and "<td>R</td><td>45,204.00</td>" in out  # "A", "R", "R": codes
    assert "<td>R 1,239.50</td>" in out  # the other column holds one sign


# --- head_fill: the last cells of a header row, from the row's line ---------------------------------------------

FILL_XS = (72, 170, 230, 290, 350)
FILL = (
    "<table><tr><th rowspan='2'>Specialty</th><th colspan='2'>2021</th><th colspan='2'>2020</th></tr>"
    + _row("%MD", "%Tot", tag="th")
    + _row("Surgery", "70.1", "99.8", "68.3", "99.9")
    + _row("Urology", "82.3", "100.0", "81.3", "97.3")
    + "</table>"
)


def _fill_texts():
    texts = [(72, 700, "Specialty"), (200, 712, "2021"), (320, 712, "2020")]
    texts += [(x, 700, t) for x, t in zip(FILL_XS[1:], ("%MD", "%Tot", "%MD", "%Tot"), strict=True)]
    for i, row in enumerate(
        [("Surgery", "70.1", "99.8", "68.3", "99.9"), ("Urology", "82.3", "100.0", "81.3", "97.3")]
    ):
        texts += [(x, 680 - 15 * i, t) for x, t in zip(FILL_XS, row, strict=True)]
    return texts


def test_a_short_header_row_gets_its_last_cells_from_its_line():
    out = _fixed(FILL, _page(_fill_texts()), {"head_fill": True})
    assert _row("%MD", "%Tot", "%MD", "%Tot", tag="th") in out
    assert _fixed(FILL, _page(_fill_texts()), {}) == FILL  # off by default


def test_a_header_the_reading_wrote_in_another_row_is_not_written_twice():
    # the reading put the last two leaf headers in the row above ("Low" and "High", not found on that row's line)
    md = (
        "<table><tr><th rowspan='2'>Specialty</th><th colspan='2'>2021</th><th>Low</th><th>High</th></tr>"
        + _row("%MD", "%Tot", tag="th")
        + _row("Surgery", "70.1", "99.8", "68.3", "99.9")
        + _row("Urology", "82.3", "100.0", "81.3", "97.3")
        + "</table>"
    )
    texts = [
        (72, 700, "Specialty"),
        (200, 712, "2021"),
        (170, 700, "%MD"),
        (230, 700, "%Tot"),
        (290, 700, "Low"),
        (350, 700, "High"),
    ]
    for i, row in enumerate(
        [("Surgery", "70.1", "99.8", "68.3", "99.9"), ("Urology", "82.3", "100.0", "81.3", "97.3")]
    ):
        texts += [(x, 680 - 15 * i, t) for x, t in zip(FILL_XS, row, strict=True)]
    assert _fixed(md, _page(texts), {"head_fill": True}) == md


# --- last_column: the value every body row prints after its last cell -------------------------------------------

LAST_XS = (72, 170, 230, 290, 350)
LAST_ROWS = [("Eldora", "201", "201", "201", "201"), ("Erath", "207", "207", "207", "207"), ("Fagus", "166", "166", "166", "166"),
             ("Honey", "61", "61", "61", "61")]  # fmt: skip
LAST_HEAD = ("Unit", "2026", "2027", "2028", "2029")
SHORT_BODY = "<table>" + _row(*LAST_HEAD, tag="th") + "".join(_row(*r[:-1]) for r in LAST_ROWS) + "</table>"


def _last_texts(notes=()):
    texts = [(x, 700, t) for x, t in zip(LAST_XS, LAST_HEAD, strict=True)]
    for i, row in enumerate(LAST_ROWS):
        texts += [(x, 680 - 15 * i, t) for x, t in zip(LAST_XS, row, strict=True)]
    return texts + list(notes)


def test_a_body_one_column_short_of_its_header_gets_the_column_the_lines_print():
    out = _fixed(SHORT_BODY, _page(_last_texts()), {"last_column": True})
    assert out == "<table>" + _row(*LAST_HEAD, tag="th") + "".join(_row(*r) for r in LAST_ROWS) + "</table>"
    assert _row(*LAST_ROWS[0]) not in _fixed(SHORT_BODY, _page(_last_texts()), {})  # off by default


def test_a_row_that_ends_in_the_last_column_gets_the_value_it_left_out_in_its_place():
    # "Honey" prints 61, -, 61, 61 and the reading wrote 61, 61, 61: its last value already stands in the new column
    rows = [*LAST_ROWS[:3], ("Honey", "61", "-", "61", "61"), ("Indigo", "150", "150", "150", "150")]
    texts = [(x, 700, t) for x, t in zip(LAST_XS, LAST_HEAD, strict=True)]
    for i, row in enumerate(rows):
        texts += [(x, 680 - 15 * i, t) for x, t in zip(LAST_XS, row, strict=True)]
    read = [*(r[:-1] for r in rows[:3]), ("Honey", "61", "61", "61"), rows[4][:-1]]
    md = "<table>" + _row(*LAST_HEAD, tag="th") + "".join(_row(*r) for r in read) + "</table>"
    out = _fixed(md, _page(texts), {"last_column": True})
    assert out == "<table>" + _row(*LAST_HEAD, tag="th") + "".join(_row(*r) for r in rows) + "</table>"


def test_notes_beside_single_rows_are_not_a_column():
    texts = [t for t in _last_texts() if t[0] != 350 or t[1] in (700, 680)]  # only the first row prints a value
    out = _fixed(SHORT_BODY, _page(texts), {"last_column": True})
    assert _row(*LAST_ROWS[0][:-1]) in out and _row(*LAST_ROWS[0]) not in out


def test_two_groups_of_words_after_the_table_are_not_one_more_column():
    notes = [(430, 680 - 15 * i, "x") for i in range(len(LAST_ROWS))]
    out = _fixed(SHORT_BODY, _page(_last_texts(notes)), {"last_column": True})
    assert _row(*LAST_ROWS[0]) not in out


def test_words_nobody_sees_are_not_put_into_a_table():
    rendered = _page(_last_texts())
    for w in rendered.text_words:  # paint over the last body column: the text layer still holds its words
        if w.box[0] * rendered.scale > 340 and w.text != "2029":
            box = (w.box[0] - 3, w.box[1] - 3, w.box[2] + 3, w.box[3] + 3)
            rendered.image.paste((255, 255, 255), tuple(int(v) for v in box))
    assert _row(*LAST_ROWS[0]) not in _fixed(SHORT_BODY, rendered, {"last_column": True})


# --- merge_apart: one table written as two with another table between them --------------------------------------

LEFT = (
    "<table><tr><th>Quarter</th><th>Exposures</th></tr><tr><td>20181</td><td>15</td></tr>"
    "<tr><td>20182</td><td>17</td></tr><tr><td>20183</td><td>18</td></tr><tr><td>20184</td><td>19</td></tr></table>"
)
RIGHT = (
    "<table><tr><th>Frequency</th><th>Severity</th></tr><tr><td>7.34</td><td>7,053</td></tr>"
    "<tr><td>4.18</td><td>57,381</td></tr><tr><td>5.52</td><td>48,191</td></tr><tr><td>4.08</td><td>33,751</td></tr></table>"
)
APART_ROWS = [("20181", "15", "7.34", "7,053"), ("20182", "17", "4.18", "57,381"), ("20183", "18", "5.52", "48,191"),
              ("20184", "19", "4.08", "33,751")]  # fmt: skip
APART_HEAD = [(72, 700, "Quarter"), (150, 700, "Exposures"), (300, 700, "Frequency"), (380, 700, "Severity")]


def _apart_texts(rows=APART_ROWS):
    texts = list(APART_HEAD)
    for i, row in enumerate(rows):
        texts += [(x, 680 - 15 * i, t) for x, t in zip((72, 150, 300, 380), row, strict=False)]
    return [*texts, (72, 500, "Frequency"), (150, 500, "2.8%"), (72, 485, "Severity"), (150, 485, "7.4%")]


BETWEEN = "<table><tr><td>Frequency</td><td>2.8%</td></tr><tr><td>Severity</td><td>7.4%</td></tr></table>"


def test_one_table_written_as_two_with_a_table_between_them_is_merged():
    md = LEFT + "\n\n" + BETWEEN + "\n\n" + RIGHT
    out = _fixed(md, _page(_apart_texts()), {"merge": True, "merge_apart": True})
    assert out.count("<table") == 2 and "<td>20181</td><td>15</td><td>7.34</td><td>7,053</td>" in out
    assert out.endswith(BETWEEN + "\n\n") or out.endswith(BETWEEN)
    assert _fixed(md, _page(_apart_texts()), {"merge": True}) == md  # `merge` joins neighbouring tables only


def test_a_short_table_beside_a_long_one_stays_a_table_of_its_own():
    more = [(f"2019{i}", f"2{i}") for i in range(1, 6)]  # five more rows on the left only
    long_left = LEFT.replace("</table>", "".join(_row(*r) for r in more) + "</table>")
    md = long_left + "\n\n" + BETWEEN + "\n\n" + RIGHT
    assert _fixed(md, _page(_apart_texts([*APART_ROWS, *more])), {"merge": True, "merge_apart": True}) == md


def test_two_tables_side_by_side_each_under_a_title_of_its_own_stay_two_with_merge_strict():
    left = LEFT.replace("<table>", '<table><tr><th colspan="2">Liability</th></tr>', 1)
    right = RIGHT.replace("<table>", '<table><tr><th colspan="2">Injury</th></tr>', 1)
    texts = [*_apart_texts(), (72, 715, "Liability"), (300, 715, "Injury")]
    md = left + "\n\n" + right
    assert _fixed(md, _page(texts), {"merge": True}).count("<table") == 1
    assert _fixed(md, _page(texts), {"merge": True, "merge_strict": True}) == md
    # a title over the left table only: the right one continues it
    out = _fixed(left + "\n\n" + RIGHT, _page(texts), {"merge": True, "merge_strict": True})
    assert out.count("<table") == 1


def test_round_three_options_leave_a_page_without_a_text_layer_alone():
    rendered = _page([(72, 700, "x")])
    rendered.text_words = rendered.words = None
    options = {
        "squeeze": True,
        "sign": True,
        "head_fill": True,
        "last_column": True,
        "merge": True,
        "merge_apart": True,
    }
    for md in (ONCE_TOO_OFTEN, WIDE, SIGNS, FILL, SHORT_BODY, LEFT + "\n\n" + BETWEEN + "\n\n" + RIGHT):
        assert _fixed(md, rendered, options) == md

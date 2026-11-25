import pytest

from docparse.base.ocria_parse import parse_page
from docparse.core.config import Config
from docparse.fixes.tables import column_of, fix_tables
from docparse.fixes.tables_align import align_table, fold, lines_of, page_words
from docparse.fixes.tables_html import parse_table
from docparse.models.adapters import ocria
from docparse.models.calls import ModelResult
from docparse.pdf.render import render_pages
from docparse.pipeline import parse
from docparse.write.markdown import write_document, write_page
from tests.pdfgen import make_pdf

# A table printed with an index column whose header cell is blank: "Year" stands over the years, not over "(1)"
HEADER = [(150, 700, "Year"), (250, 700, "Premium"), (350, 700, "Loss")]
BODY = [
    [(72, 680, "(1)"), (150, 680, "2025"), (250, 680, "100,000"), (350, 680, "50.5%")],
    [(72, 665, "(2)"), (150, 665, "2024"), (250, 665, "200,000"), (350, 665, "70.1%")],
    [(72, 650, "(3)"), (150, 650, "2023"), (250, 650, "300,000"), (350, 650, "80.9%")],
]
TITLE = "# **Annual results**\n\n"
SHORT_HEADER = (
    "<table><tr><th>Year</th><th>Premium</th><th>Loss</th></tr>"
    "<tr><td>(1)</td><td><strong>2025</strong></td><td>100,000</td><td>50.5%</td></tr>"
    "<tr><td>(2)</td><td>2024</td><td>200,000</td><td>70.1%</td></tr>"
    "<tr><td>(3)</td><td>2023</td><td>300,000</td><td>80.9%</td></tr></table>"
)
FIXED_HEADER = "<tr><th></th><th>Year</th><th>Premium</th><th>Loss</th></tr>"


def _rendered(rows=None):
    texts = [(x, y, 10, t) for x, y, t in HEADER] + [(x, y, 10, t) for row in rows or BODY for x, y, t in row]
    return render_pages(make_pdf([{"texts": texts}]))[0]


def _fixed(md, rendered, options=None):
    page = parse_page(md)
    fix_tables(page, rendered, {"regrid": True} if options is None else options)
    return page


@pytest.mark.parametrize(
    "raw",
    [
        SHORT_HEADER,
        '<table border="1">\n<tr><th colspan="2">A</th></tr>\n<tr>\n  <td rowspan=2>x</td><td>1</td>\n</tr>'
        "<tr><td>2</td></tr>\n</table>",
        "<table><tr><td>a<br/>b</td><td><sup>1</sup>&amp;</td></tr></table>",
    ],
)
def test_html_tables_round_trip_byte_for_byte(raw):
    table = parse_table(raw)
    assert table.render() == raw
    table.changed = True  # re-rendered from its parts, an unedited table is still the same bytes
    assert table.render() == raw


def test_nested_or_rowless_tables_are_not_modelled():
    assert parse_table("<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>") is None
    assert parse_table("<table></table>") is None


def test_grid_positions_widths_and_cell_text():
    t = parse_table(
        "<table><tr><th rowspan='2'>Name</th><th colspan=\"2\">Sales <strong>2025</strong></th></tr>"
        "<tr><th>Q1</th><th>Q2</th></tr><tr><td>a</td><td>1</td><td>2</td></tr><tr><td>b</td><td>3</td></tr></table>"
    )
    assert [(c.row, c.col) for c in t.rows[1].cells] == [(1, 1), (1, 2)]
    assert [t.row_width(r) for r in range(4)] == [3, 3, 3, 2] and t.width() == 3
    assert t.rows[0].cells[1].text == "Sales 2025" and t.spanned_into(1) == {0}
    assert parse_table("<table><tr><td>a<br>b c</td></tr></table>").rows[0].cells[0].lines == ["a", "b c"]


def test_fold_ignores_case_spacing_and_dash_style():
    assert fold("Net  Sales − 1,200") == fold("net sales -1,200") == "netsales-1,200"


def test_rows_are_matched_to_their_lines_and_cells_to_words():
    rendered = _rendered()
    lines = lines_of(page_words(rendered))
    table = parse_table(SHORT_HEADER)
    al = align_table(table, lines, set())
    assert al.cells == 15 and al.found == 1.0
    assert sorted(al.row_line) == [0, 1, 2, 3]
    assert al.row_line[0] < al.row_line[1] < al.row_line[2] < al.row_line[3]
    year_2024 = table.rows[2].cells[1]
    x0, x1 = al.span(year_2024)
    assert 149 < x0 < x1 < 180


def test_a_header_row_missing_its_corner_cell_gets_it_back():
    page = _fixed(TITLE + SHORT_HEADER, _rendered())
    out = write_page(page)
    assert FIXED_HEADER in out
    assert out == TITLE + SHORT_HEADER.replace("<tr><th>Year</th><th>Premium</th><th>Loss</th></tr>", FIXED_HEADER)
    region = page.regions[1]
    assert [c.reader for c in region.candidates] == ["ocria.base", "fix.tables.pdf"]
    assert region.chosen().reader == "fix.tables.pdf" and region.candidates[0].raw == SHORT_HEADER


def test_a_body_row_missing_an_empty_cell_is_moved_to_its_columns():
    rows = [BODY[0], [(72, 665, "(2)"), (250, 665, "200,000"), (350, 665, "70.1%")], BODY[2]]
    md = (
        "<table><tr><th></th><th>Year</th><th>Premium</th><th>Loss</th></tr>"
        "<tr><td>(1)</td><td>2025</td><td>100,000</td><td>50.5%</td></tr>"
        "<tr><td>(2)</td><td>200,000</td><td>70.1%</td></tr>"
        "<tr><td>(3)</td><td>2023</td><td>300,000</td><td>80.9%</td></tr></table>"
    )
    out = write_page(_fixed(md, _rendered(rows)))
    assert "<tr><td>(2)</td><td></td><td>200,000</td><td>70.1%</td></tr>" in out


def test_a_row_with_a_stray_empty_cell_is_laid_out_again_only_with_regrid_wide():
    head = "<table><tr><th></th><th>Year</th><th>Premium</th><th>Loss</th></tr>"
    stray = "<tr><td>(2)</td><td></td><td>2024</td><td>200,000</td><td>70.1%</td></tr>"
    md = (
        head
        + "<tr><td>(1)</td><td>2025</td><td>100,000</td><td>50.5%</td></tr>"
        + stray
        + "<tr><td>(3)</td><td>2023</td><td>300,000</td><td>80.9%</td></tr></table>"
    )
    assert write_page(_fixed(md, _rendered())) == md  # regrid alone leaves rows wider than the table
    out = write_page(_fixed(md, _rendered(), {"regrid": True, "regrid_wide": True}))
    assert out == md.replace(stray, "<tr><td>(2)</td><td>2024</td><td>200,000</td><td>70.1%</td></tr>")


def test_a_table_cut_off_by_the_token_limit_is_completed_from_the_text_layer():
    rows = [*BODY, [(72, 635, "(4)"), (150, 635, "2022"), (250, 635, "400,000"), (350, 635, "90.2%")]]
    rows.append([(72, 610, "Source:"), (110, 610, "annual"), (160, 610, "statement")])  # a note under the table
    cut = (
        "<table><tr><th></th><th>Year</th><th>Premium</th><th>Loss</th></tr>"
        "<tr><td>(1)</td><td>2025</td><td>100,000</td><td>50.5%</td></tr>"
        "<tr><td>(2)</td><td>2024</td><td>200,000</td><td>70.1%</td></tr><tr><td>(3)</td><td>20"
    )
    md = TITLE + cut
    assert write_page(_fixed(md, _rendered(rows), {"regrid": True})) == md  # only with "complete"
    page = _fixed(md, _rendered(rows), {"complete": True})
    done = cut[: cut.rindex("<tr>")] + (
        "<tr><td>(3)</td><td>2023</td><td>300,000</td><td>80.9%</td></tr>"
        "<tr><td>(4)</td><td>2022</td><td>400,000</td><td>90.2%</td></tr></table>"
    )
    assert write_page(page) == TITLE + done
    assert page.regions[1].chosen().prov[0].detail == "complete"


SIDE_TEXTS = [
    (180, 700, "Quarterly"),
    (350, 700, "Rolling"),
    (150, 685, "Claims"),
    (220, 685, "Paid"),
    (320, 685, "Claims"),
    (390, 685, "Paid"),
    (72, 692, "Year"),
] + [
    (x, y, t)
    for y, row in zip(
        (670, 655, 640, 625),
        (["2021", "10", "100"], ["2022", "20", "200", "30", "300"], ["2023", "40", "400", "50", "500"],
         ["2024", "60", "600", "70", "700"]),
        strict=True,
    )
    for x, t in zip((72, 150, 220, 320, 390), row, strict=False)
]  # fmt: skip
LEFT = (
    '<table><tr><th rowspan="2">Year</th><th colspan="2">Quarterly</th></tr><tr><th>Claims</th><th>Paid</th></tr>'
    "<tr><td>2021</td><td>10</td><td>100</td></tr><tr><td>2022</td><td>20</td><td>200</td></tr>"
    "<tr><td>2023</td><td>40</td><td>400</td></tr><tr><td>2024</td><td>60</td><td>600</td></tr></table>"
)
RIGHT = (
    '<table><tr><th colspan="2">Rolling</th></tr><tr><th>Claims</th><th>Paid</th></tr>'
    "<tr><td>30</td><td>300</td></tr><tr><td>50</td><td>500</td></tr><tr><td>70</td><td>700</td></tr></table>"
)


def test_one_table_read_as_two_side_by_side_is_merged():
    rendered = render_pages(make_pdf([{"texts": [(x, y, 10, t) for x, y, t in SIDE_TEXTS]}]))[0]
    md = LEFT + "\n\n" + RIGHT
    assert write_page(_fixed(md, rendered, {"regrid": True})) == md
    page = _fixed(md, rendered, {"merge": True})
    merged = (
        '<table><tr><th rowspan="2">Year</th><th colspan="2">Quarterly</th><th colspan="2">Rolling</th></tr>'
        "<tr><th>Claims</th><th>Paid</th><th>Claims</th><th>Paid</th></tr>"
        "<tr><td>2021</td><td>10</td><td>100</td><td></td><td></td></tr>"
        "<tr><td>2022</td><td>20</td><td>200</td><td>30</td><td>300</td></tr>"
        "<tr><td>2023</td><td>40</td><td>400</td><td>50</td><td>500</td></tr>"
        "<tr><td>2024</td><td>60</td><td>600</td><td>70</td><td>700</td></tr></table>"
    )
    assert write_page(page) == merged + "\n\n"
    # a title row over the left table only gets a blank cell over the right one
    titled = LEFT.replace("<table>", '<table><tr><th colspan="3">Trend</th></tr>', 1)
    out = write_page(_fixed(titled + "\n\n" + RIGHT, rendered, {"merge": True}))
    assert out.startswith('<table><tr><th colspan="3">Trend</th><th colspan="2"></th></tr>')
    assert out.endswith(merged[merged.index("<tr><td>2021") :] + "\n\n")
    # a right table repeating the left one's row-label header is a repeated column group: kept apart
    repeated = RIGHT.replace('<tr><th colspan="2">Rolling</th></tr>', '<tr><th colspan="2">Year</th></tr>')
    assert write_page(_fixed(LEFT + "\n\n" + repeated, rendered, {"merge": True})) == LEFT + "\n\n" + repeated


def test_tables_that_fit_their_page_are_written_back_unchanged():
    md = TITLE + SHORT_HEADER.replace("<tr><th>Year</th>", "<tr><th></th><th>Year</th>")
    page = _fixed(md, _rendered())
    assert write_page(page) == md and all(len(r.candidates) == 1 for r in page.regions)


def test_no_option_no_text_layer_or_an_unhealthy_text_layer_change_nothing():
    md = TITLE + SHORT_HEADER
    assert write_page(_fixed(md, _rendered(), {})) == md
    bare = _rendered()
    bare.text_words = bare.words = None
    assert write_page(_fixed(md, bare)) == md
    garbled = [[(x, y, "Q" + t[::-1]) for x, y, t in row] for row in BODY]  # a broken font map
    assert write_page(_fixed(md, _rendered(garbled))) == md
    # most cells found, but under the health share asked for
    assert write_page(_fixed(md, _rendered(), {"regrid": True, "health": 1.01})) == md


def test_column_of_a_spanning_cell_and_of_a_word_between_columns():
    bands = {0: [(70, 90)], 1: [(150, 170)], 2: [(250, 280)], 3: [(350, 375)]}
    assert column_of(152, 168, 1, bands, 4) == 1
    assert column_of(200, 220, 1, bands, 4) is None  # over no column
    assert column_of(190, 240, 2, bands, 4) == 1  # centred over columns 1 and 2


def test_config_table_options():
    cfg = Config.load("configs/standin.yaml", ["table_options.regrid=true"])
    assert cfg.table_options == {"regrid": True}
    assert cfg.config_hash() != Config.load("configs/standin.yaml").config_hash()
    with pytest.raises(ValueError, match="unknown table option"):
        Config.load("configs/standin.yaml", ["table_options.rebuild=true"])


class _Answer:
    def __init__(self, markdown):
        self.markdown = markdown

    def get(self, call, blobs, meta=None):
        return ModelResult(key=call.key(), output={"markdown": self.markdown, "layout_page": None})


def test_pipeline_applies_table_options_only_when_set():
    texts = [(x, y, 10, t) for x, y, t in HEADER] + [(x, y, 10, t) for row in BODY for x, y, t in row]
    pdf = make_pdf([{"texts": texts}])
    base = Config(page_model=ocria.PUBLISHED, page_runtime="published", prompt_sha="published")
    md = TITLE + SHORT_HEADER
    assert write_document(parse(pdf, base, _Answer(md)).document) == md
    fixed = base.model_copy(update={"table_options": {"regrid": True}})
    assert FIXED_HEADER in write_document(parse(pdf, fixed, _Answer(md)).document)

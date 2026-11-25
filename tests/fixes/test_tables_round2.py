"""Round-two table options: leaf, absorb, split, upright, dedupe, complete_extent, unspan, join."""

from docparse.base.ocria_parse import parse_page
from docparse.fixes.tables import fix_tables, join_bracket_rows, looped_copies, right_align_leaf
from docparse.fixes.tables_align import TWord, lines_of, turned, upright_turn
from docparse.fixes.tables_html import parse_table
from docparse.pdf.render import render_pages
from docparse.write.markdown import write_page
from tests.pdfgen import make_pdf


def _page(texts):
    return render_pages(make_pdf([{"texts": [(x, y, 10, t) for x, y, t in texts]}]))[0]


def _bare():
    rendered = _page([(72, 700, "x")])
    rendered.text_words = rendered.words = None
    return rendered


def _fixed(md, rendered, options):
    page = parse_page(md)
    fix_tables(page, rendered, {"regrid": True, **options})
    return write_page(page)


# --- leaf: a short last header row stands over the last columns (no page geometry needed) -----------------------

LEAF = (
    "<table><tr><th rowspan='2'></th><th colspan='2'>Group A</th><th colspan='2'>Group B</th></tr>"
    "<tr><th>Mean</th><th>SD</th><th>Mean</th><th>SD</th></tr>"
    "<tr><td>Age</td><td>1, 2</td><td>3.7</td><td>1.1</td><td>4.2</td><td>0.7</td></tr>"
    "<tr><td>Weight</td><td>2</td><td>3.4</td><td>1.0</td><td>4.0</td><td>0.8</td></tr></table>"
)


def test_a_short_leaf_header_row_is_moved_over_the_last_columns_on_a_scan():
    out = _fixed(LEAF, _bare(), {"leaf": True, "leaf_mode": 3})
    assert "<tr><th></th><th>Mean</th><th>SD</th><th>Mean</th><th>SD</th></tr>" in out
    assert _fixed(LEAF, _bare(), {}) == LEAF  # off by default


def test_leaf_is_left_alone_when_a_one_column_header_ends_the_row_above():
    md = LEAF.replace("<th colspan='2'>Group B</th>", "<th colspan='2'>Group B</th><th>Change</th>")
    table = parse_table(md)
    assert not right_align_leaf(table, 3)  # the leaf row's gap may be under "Change", at the end


# --- absorb: a unit printed under a header that spans down joins that header ------------------------------------

ABSORB_TEXTS = [
    (72, 700, "Note"),
    (130, 700, "Assets"),
    (300, 700, "GROUP"),
    (130, 688, "(DKK"),
    (160, 688, "1,000)"),
    (280, 688, "2025"),
    (340, 688, "2024"),
    (72, 670, "7"),
    (130, 670, "Software"),
    (280, 670, "1,809"),
    (340, 670, "1,771"),
    (72, 655, "8"),
    (130, 655, "Aircraft"),
    (280, 655, "207,680"),
    (340, 655, "237,358"),
]
ABSORB = (
    "<table><tr><th rowspan='2'>Note</th><th rowspan='2'>Assets</th><th colspan='2'>GROUP</th></tr>"
    "<tr><th>(DKK 1,000)</th><th>2025</th><th>2024</th></tr>"
    "<tr><td>7</td><td>Software</td><td>1,809</td><td>1,771</td></tr>"
    "<tr><td>8</td><td>Aircraft</td><td>207,680</td><td>237,358</td></tr></table>"
)


def test_a_unit_under_a_header_spanning_down_joins_it():
    out = _fixed(ABSORB, _page(ABSORB_TEXTS), {"absorb": True})
    assert "<th rowspan='2'>Assets (DKK 1,000)</th>" in out
    assert "<tr><th>2025</th><th>2024</th></tr>" in out
    assert _fixed(ABSORB, _page(ABSORB_TEXTS), {}) == ABSORB


# --- split: two numbers on either side of a gutter are two columns ----------------------------------------------

SPLIT_TEXTS = [
    (72, 700, "Quarter"),
    (150, 700, "GMV"),
    (72, 685, "Q3"),
    (150, 685, "PLN"),
    (175, 685, "15,458m"),
    (240, 685, "+9.9%"),
    (270, 685, "YoY"),
    (72, 670, "9M"),
    (150, 670, "PLN"),
    (175, 670, "45,573m"),
    (240, 670, "+10.0%"),
    (275, 670, "YoY"),
]
SPLIT = (
    "<table><tr><th>Quarter</th><th>GMV</th></tr>"
    "<tr><td>Q3</td><td>PLN 15,458m +9.9% YoY</td></tr>"
    "<tr><td>9M</td><td>PLN 45,573m +10.0% YoY</td></tr></table>"
)


def test_two_values_with_a_gutter_between_them_become_two_columns():
    out = _fixed(SPLIT, _page(SPLIT_TEXTS), {"split": True})
    assert '<th colspan="2">GMV</th>' in out
    assert "<tr><td>Q3</td><td>PLN 15,458m</td><td>+9.9% YoY</td></tr>" in out
    assert _fixed(SPLIT, _page(SPLIT_TEXTS), {}) == SPLIT


def test_a_currency_code_beside_its_amount_is_not_split():
    texts = [(x, y, t.replace("+9.9%", "EUR").replace("+10.0%", "EUR")) for x, y, t in SPLIT_TEXTS]
    texts = [(x, y, t) for x, y, t in texts if t not in ("15,458m", "45,573m", "PLN")]
    md = (
        "<table><tr><th>Quarter</th><th>GMV</th></tr><tr><td>Q3</td><td>EUR YoY</td></tr>"
        "<tr><td>9M</td><td>EUR YoY</td></tr></table>"
    )
    assert _fixed(md, _page(texts), {"split": True}) == md


# --- upright: the text layer of a page printed sideways or upside down is turned to read upright -----------------


def _words(rows, step=40):
    """Upright words: rows of texts, left to right, written in reading order."""
    out, i = [], 0
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            x0, y0 = 72 + c * step, 100 + r * 15
            out.append(TWord(i, text, x0, y0, x0 + 6 * len(text), y0 + 10, text.lower()))
            i += 1
    return out


def test_the_turn_that_sets_a_page_upright_is_found_from_the_text_layer():
    rows = [["alpha", "beta", "gamma", "delta"] for _ in range(6)]
    words = _words(rows)
    assert upright_turn(words, 612, 792) == 0
    for angle, back in ((180, 180), (90, 270), (270, 90)):
        sideways = turned(words, 612, 792, angle)
        w, h = (792, 612) if angle in (90, 270) else (612, 792)
        assert upright_turn(sideways, w, h) == back
        again = turned(sideways, w, h, back)
        assert [[x.text for x in line] for line in lines_of(again)] == rows


# --- dedupe: a table the read wrote again and again ---------------------------------------------------------------


def test_copies_of_a_looped_table_are_dropped_but_repeated_templates_stay():
    table = "<table><tr><th>Sector</th><th>Weighting</th></tr><tr><td>Industrials</td><td>102.30%</td></tr></table>"
    looped = "\n\n".join(["Intro"] + [f"The following table:\n{table}" for _ in range(60)])
    page = parse_page(looped)
    copies = looped_copies(page)
    tables = [r.id for r in page.regions if r.kind == "table"]
    assert len(tables) == 60 and copies == tables[1:]
    few = parse_page("\n\n".join([table] * 4))  # a form printing one blank table four times is no loop
    assert looped_copies(few) == []


# --- complete_extent: a cut-off table among column groups side by side --------------------------------------------


def test_a_cut_off_table_is_completed_inside_its_own_column_group():
    left = [(72, 700 - 15 * i, f"{i}") for i in range(6)] + [(120, 700 - 15 * i, f"{i}.5") for i in range(6)]
    right = [(300, 700 - 15 * i, f"{i + 6}") for i in range(6)] + [(350, 700 - 15 * i, f"{i + 6}.5") for i in range(6)]
    rendered = _page(left + right)
    cut = (
        "<table><tr><td>0</td><td>0.5</td></tr><tr><td>1</td><td>1.5</td></tr><tr><td>2</td><td>2.5</td></tr><tr><td>3"
    )
    md = "Head\n\n" + cut
    assert _fixed(md, rendered, {"complete": True}) == md  # the right group's words do not fit: nothing added
    out = _fixed(md, rendered, {"complete": True, "complete_extent": True})
    assert out.endswith(
        "<tr><td>3</td><td>3.5</td></tr><tr><td>4</td><td>4.5</td></tr><tr><td>5</td><td>5.5</td></tr></table>"
    )


# --- unspan (not kept): a value spanning columns whose words sit in one of them ----------------------------------


def test_a_spanning_value_printed_in_one_column_moves_to_that_column():
    texts = [(72, 700, "Name"), (150, 700, "MIN"), (200, 700, "TYP"), (250, 700, "MAX")]
    texts += [(72, 685, "Gain"), (150, 685, "-1"), (200, 685, "0.3"), (250, 685, "1")]
    texts += [(72, 670, "Bits"), (150, 670, "12")]
    md = (
        "<table><tr><th>Name</th><th>MIN</th><th>TYP</th><th>MAX</th></tr>"
        "<tr><td>Gain</td><td>-1</td><td>0.3</td><td>1</td></tr>"
        "<tr><td>Bits</td><td colspan='3'>12</td></tr></table>"
    )
    out = _fixed(md, _page(texts), {"unspan": True, "unspan_mode": 2})
    assert "<tr><td>Bits</td><td>12</td><td></td><td></td></tr>" in out


# --- join (rejected; kept off): a row of bracketed second lines joins the row above -------------------------------


def test_bracketed_continuation_rows_join_the_row_above():
    t = parse_table(
        "<table><tr><th></th><th>N</th><th>RR</th></tr><tr><td>Visits</td><td>40</td><td>1.8</td></tr>"
        "<tr><td></td><td>(82)</td><td>[1.5–2.0]</td></tr><tr><td>Tests</td><td>33</td><td>11.0</td></tr></table>"
    )
    assert join_bracket_rows(t)
    t.changed = True
    assert t.render() == (
        "<table><tr><th></th><th>N</th><th>RR</th></tr><tr><td>Visits</td><td>40 (82)</td><td>1.8 [1.5–2.0]</td></tr>"
        "<tr><td>Tests</td><td>33</td><td>11.0</td></tr></table>"
    )

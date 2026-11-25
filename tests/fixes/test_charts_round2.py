import pytest

from docparse.base.ocria_parse import parse_page
from docparse.fixes.charts import correct_tables
from docparse.fixes.charts_geometry import find_axes
from docparse.fixes.layout_refine import PtWord, words_in_points
from docparse.pdf.graphics import load_shapes
from docparse.pdf.render import render_pages
from docparse.write.markdown import write_page
from tests.pdfgen import make_pdf

KEPT = {"signs": False, "upright": False}  # the kept configuration's axis rules


def test_tick_labels_with_lost_leading_digits_take_their_value_from_the_sequence():
    # "80 0 0 0 0 30 20 10 0" down a column: the 7, 6, 5 and 4 glyphs have no text mapping
    texts = ("80", "0", "0", "0", "0", "30", "20", "10", "0")
    col = [PtWord(t, 40 + 3 * (len(t) == 1), 300 + 10 * i, 50, 305 + 10 * i) for i, t in enumerate(texts)]
    assert all(len(a.values) <= 3 for a in find_axes(col, **KEPT))
    (axis,) = find_axes(col, lost=True, **KEPT)
    assert axis.values == [80, 70, 60, 50, 40, 30, 20, 10, 0]
    assert axis.value(302.5 + 10 * 2) == pytest.approx(60)


def test_lost_minus_signs_and_other_numbers_are_not_lost_digits():
    # "20 10 0 10 20": a minus sign alone is not a lost digit (population pyramids print both sides positive)
    col = [PtWord(t, 40, 300 + 13 * i, 50, 306 + 13 * i) for i, t in enumerate(("20", "10", "0", "10", "20"))]
    assert all(min(a.values) >= 0 for a in find_axes(col, lost=True, **KEPT))
    # a label that is not the tail of its value in the sequence leaves the column alone
    odd = [PtWord(t, 40, 300 + 10 * i, 50, 305 + 10 * i) for i, t in enumerate(("80", "5", "60", "50", "40"))]
    assert all(70 not in a.values for a in find_axes(odd, lost=True, **KEPT))


def test_rotated_category_labels_split_into_pieces_place_the_categories():
    from docparse.fixes.charts_cats import rotated_positions

    box = (90, 50, 400, 200)  # the plot; rotated labels just under it, read bottom to top
    words = []
    for x, pieces in ((100, ["Swit", "erland"]), (150, ["United", "States"]), (200, ["at", "ia"]), (250, ["Spain"])):
        y = 260.0
        for p in pieces:  # each piece a narrow, tall box; the next piece above it
            words.append(PtWord(p, x, y - 4 * len(p), x + 5, y))
            y -= 4 * len(p) + 2
    words.append(PtWord("Source:", 90, 270, 120, 276))  # horizontal text under the labels
    cats = ["Switzerland", "United States", "Latvia", "Spain"]
    assert rotated_positions(cats, words, box) == [102.5, 152.5, 202.5, 252.5]
    assert rotated_positions(["Spain", "Latvia", "United States", "Switzerland"], words, box) is None
    assert rotated_positions(["Spain", "Latvia", "United States", "Switzerland"], words, box, ordered=False) == [
        252.5,
        202.5,
        152.5,
        102.5,
    ]


def _two_series_pdf():
    red, blue = {"A": 12, "B": 30, "C": 20, "D": 26}, {"A": 10, "B": 27, "C": 18, "D": 31}
    draw = ["0.8 0.8 0.8 RG 0.5 w"]
    draw += [f"70 {y} m 400 {y} l S" for y in (100, 150, 200, 250, 300)]  # gridlines at 0, 10, .. 40
    for i, k in enumerate(red):
        draw.append(f"0.9 0.1 0.1 rg {100 + 70 * i} 100 20 {5 * red[k]} re f")
        draw.append(f"0.1 0.1 0.9 rg {122 + 70 * i} 100 20 {5 * blue[k]} re f")
    draw += ["0.9 0.1 0.1 rg 420 320 8 8 re f", "0.1 0.1 0.9 rg 420 300 8 8 re f"]  # legend keys
    texts = [(52, y - 1.5, 8, str(v)) for v, y in zip((0, 10, 20, 30, 40), (100, 150, 200, 250, 300), strict=True)]
    texts += [(115 + 70 * i, 88, 8, k) for i, k in enumerate(red)]
    texts += [(432, 321, 8, "Red"), (432, 301, 8, "Blue")]
    return make_pdf([{"draw": draw, "texts": texts}])


def _corrected(md, pdf, opts):
    rendered = render_pages(pdf)[0]
    page = parse_page(md)
    page.width, page.height = rendered.width_pt, rendered.height_pt
    correct_tables(page, words_in_points(rendered), load_shapes(pdf, 0), opts)
    return write_page(page)


def test_a_binding_against_the_legend_takes_the_series_drawn_in_the_named_colour():
    # the reading put Blue's numbers under Red and Red's under Blue
    table = (
        "<table><tr><th>Category</th><th>Red</th><th>Blue</th></tr><tr><th>A</th><td>10</td><td>12</td></tr>"
        "<tr><th>B</th><td>28</td><td>29</td></tr><tr><th>C</th><td>18</td><td>21</td></tr>"
        "<tr><th>D</th><td>30</td><td>25</td></tr></table>"
    )
    pdf = _two_series_pdf()
    plain = _corrected(table, pdf, {"measure": True, "legend": True})
    assert "<th>A</th><td>10</td><td>12</td>" in plain  # the values bound Red to the blue bars
    fixed = _corrected(table, pdf, {"measure": True, "legend": True, "legend_override": True})
    assert "<th>A</th><td>12</td><td>10</td>" in fixed and "<th>D</th><td>26</td><td>31</td>" in fixed


def test_a_series_bound_in_another_panel_moves_to_the_panel_titled_with_its_group():
    panels = {
        "North region": (52, 100, {"Red": [12, 30, 20, 26], "Blue": [10, 27, 18, 31]}),
        "South region": (302, 350, {"Red": [22, 8, 34, 14], "Blue": [6, 20, 30, 12]}),
    }
    rgb = {"Red": "0.9 0.1 0.1", "Blue": "0.1 0.1 0.9"}
    draw, texts = [], []
    for title, (x_axis, x_bars, series) in panels.items():
        texts += [(x_axis, 100 + 50 * k - 1.5, 8, str(10 * k)) for k in range(5)]
        texts.append((x_bars + 20, 320, 8, title))
        for j, (name, vals) in enumerate(series.items()):
            draw += [f"{rgb[name]} rg {x_bars + 45 * i + 12 * j} 100 10 {5 * v} re f" for i, v in enumerate(vals)]
    draw += ["0.9 0.1 0.1 rg 150 350 8 8 re f", "0.1 0.1 0.9 rg 250 350 8 8 re f"]
    texts += [(162, 351, 8, "Red"), (262, 351, 8, "Blue")]
    pdf = make_pdf([{"draw": draw, "texts": texts}])
    rows = ""
    for group, reading in (("North region", panels["South region"][2]), ("South region", panels["North region"][2])):
        for i, cat in enumerate("ABCD"):  # each panel read with the other panel's numbers, off by one
            head = f'<th rowspan="4">{group}</th>' if i == 0 else ""
            rows += f"<tr>{head}<th>{cat}</th><td>{reading['Red'][i] + 1}</td><td>{reading['Blue'][i] + 1}</td></tr>"
    table = f"<table><tr><th>Region</th><th>Category</th><th>Red</th><th>Blue</th></tr>{rows}</table>"
    opts = {"measure": True, "panels": True, "legend": True, "titles": True, "legend_override": True}
    plain = _corrected(table, pdf, opts)
    assert "<th>A</th><td>22</td><td>6</td>" in plain  # North read in the South panel
    moved = _corrected(table, pdf, {**opts, "panel_override": True})
    assert '<th rowspan="4">North region</th><th>A</th><td>12</td><td>10</td>' in moved
    assert "<th>D</th><td>14</td><td>12</td>" in moved

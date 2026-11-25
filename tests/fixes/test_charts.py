import pytest

from docparse.base.ocria_parse import parse_page
from docparse.core.config import Config
from docparse.fixes.charts import correct_tables, table_units
from docparse.fixes.charts_geometry import find_axes, find_marks, snap_axis, tick_value
from docparse.fixes.charts_table import format_number, number_of, parse_cells, replace_spans
from docparse.fixes.layout_refine import PtWord, words_in_points
from docparse.models.adapters import ocria
from docparse.models.calls import ModelResult
from docparse.pdf.graphics import load_shapes
from docparse.pdf.render import render_pages
from docparse.pipeline import parse
from docparse.write.markdown import write_document, write_page
from tests.pdfgen import make_pdf

VALUES = {"A": 12, "B": 27, "C": 35, "D": 8}  # bar heights: 5 points per unit from the zero line at y=100


def _chart_pdf(extra_texts=()):
    draw = ["0.8 0.8 0.8 RG 0.5 w"]
    draw += [f"70 {y} m 400 {y} l S" for y in (100, 150, 200, 250, 300)]  # gridlines at 0, 10, .. 40
    for i, v in enumerate(VALUES.values()):
        draw.append(f"0.1 0.3 0.8 rg {100 + 70 * i} 100 30 {5 * v} re f")
    draw.append("0.1 0.3 0.8 rg 420 300 8 8 re f")  # legend key
    texts = [(52, y - 1.5, 8, str(v)) for v, y in zip((0, 10, 20, 30, 40), (100, 150, 200, 250, 300), strict=True)]
    texts += [(112 + 70 * i, 88, 8, k) for i, k in enumerate(VALUES)]
    texts += [(432, 301, 8, "Sales"), *extra_texts]
    return make_pdf([{"draw": draw, "texts": texts}])


TABLE = (
    "<table><tr><th>Category</th><th>Sales</th></tr><tr><th>A</th><td>11</td></tr><tr><th>B</th><td>28</td></tr>"
    "<tr><th>C</th><td>34</td></tr><tr><th>D</th><td>9</td></tr></table>"
)


def _page(md, pdf):
    rendered = render_pages(pdf)[0]
    page = parse_page(md)
    page.width, page.height = rendered.width_pt, rendered.height_pt
    return page, words_in_points(rendered), load_shapes(pdf, 0)


def test_tick_values_and_numbers_in_cells():
    assert [tick_value(t) for t in ("−20%", "1,000", "$5", "0.25", "abc", "2024")] == [-20, 1000, 5, 0.25, None, 2024]
    raw = "<table><tr><th>x</th><td>−3.5%</td><td>$1,234</td><td>n/a</td><td>1–2</td><td><b>4</b></td></tr></table>"
    cells = parse_cells(raw)
    nums = [number_of(raw, c) for c in cells]
    assert nums[0] is None and nums[3] is None and nums[4] is None and nums[5] is None
    assert nums[1].value == -3.5 and nums[1].minus == "−" and nums[1].decimals == 1
    assert nums[2].value == 1234 and nums[2].comma and nums[2].currency == "$"
    assert format_number(-2.04, 1, nums[1]) == "−2.0" and format_number(2345.6, 0, nums[2]) == "$2,346"
    edited = replace_spans(raw, [(nums[1].span, "−2.0")])
    assert edited == raw.replace("−3.5%", "−2.0%")


def test_table_units_read_both_ways():
    down, across = table_units(TABLE)
    assert [u.name for u in down] == ["Sales"] and down[0].cats == ["A", "B", "C", "D"]
    assert down[0].values == [11, 28, 34, 9]
    assert [u.name for u in across] == ["A", "B", "C", "D"] and across[0].cats == ["Sales"]


def test_axes_from_tick_labels_linear_log_and_stray_numbers():
    col = [PtWord(str(v), 40, 500 - 20 * i, 50, 506 - 20 * i) for i, v in enumerate((0, 10, 20, 30))]
    stray = PtWord("17", 40, 380, 50, 386)  # a figure number aligned with the ticks, off their spacing
    (axis,) = find_axes([*col, stray])
    assert axis.orient == "v" and axis.values == [30, 20, 10, 0] and axis.value(503) == pytest.approx(0)
    logs = [PtWord(f"{10**k}", 40, 500 - 30 * k, 60, 506 - 30 * k) for k in range(2, 6)]
    (log_axis,) = find_axes(logs)
    assert log_axis.log and log_axis.value(503 - 30 * 3) == pytest.approx(1000)


def test_labels_beside_their_gridlines_snap_to_them():
    pdf = _chart_pdf()
    page, words, shapes = _page(TABLE, pdf)
    marks = find_marks(shapes, page.width, page.height)
    (axis,) = [a for a in find_axes(words) if a.orient == "v"]
    snapped = snap_axis(axis, marks)
    assert snapped.coord(0) == pytest.approx(792 - 100, abs=0.01)
    assert snapped.coord(40) == pytest.approx(792 - 300, abs=0.01)


def test_bar_values_replace_the_reading_and_nothing_else_changes():
    md = f"# Chart\n\n{TABLE}\n\nSource: test"
    page, words, shapes = _page(md, _chart_pdf())
    assert correct_tables(page, words, shapes, {"measure": True}) == 4
    out = write_page(page)
    assert out == md.replace(">11<", ">12<").replace(">28<", ">27<").replace(">34<", ">35<").replace(">9<", ">8<")
    assert page.regions[1].chosen().reader == "fix.charts.vector"


def test_printed_values_and_unrelated_tables_are_kept():
    printed = _chart_pdf(extra_texts=[(182, 240, 7, "28")])  # a data label printing the model's reading
    page, words, shapes = _page(TABLE, printed)
    correct_tables(page, words, shapes, {"measure": True})
    assert ">28<" in write_page(page) and ">12<" in write_page(page)
    other = TABLE.replace(">11<", ">900<").replace(">28<", ">100<").replace(">34<", ">750<").replace(">9<", ">20<")
    page, words, shapes = _page(other, _chart_pdf())
    assert correct_tables(page, words, shapes, {"measure": True}) == 0
    assert write_page(page) == other and len(page.regions[0].candidates) == 1


def test_config_option_and_pipeline_round_trip():
    cfg = Config.load("configs/standin.yaml", ["chart_options.measure=true"])
    assert cfg.chart_options == {"measure": True}
    with pytest.raises(ValueError, match="unknown chart option"):
        Config.load("configs/standin.yaml", ["chart_options.guess=true"])

    class Client:
        def get(self, call, blobs, meta=None):
            return ModelResult(key=call.key(), output={"markdown": TABLE})

    pdf = _chart_pdf()
    base = Config(page_model=ocria.PUBLISHED, page_runtime="published", prompt_sha="published")
    assert write_document(parse(pdf, base, Client()).document) == TABLE
    fixed = base.model_copy(update={"chart_options": {"measure": True}})
    assert ">12<" in write_document(parse(pdf, fixed, Client()).document)

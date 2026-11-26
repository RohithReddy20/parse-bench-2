import pytest

from docparse.base.ocria_parse import parse_page
from docparse.fixes.chart_hints import extract_chart_hint, format_chart_hint
from docparse.fixes.charts import correct_tables
from docparse.fixes.layout_refine import PtWord
from docparse.pdf.graphics import Shape

PAGE_SIZE = (612, 792)
BLUE = (20, 80, 180)


def _ticks(x0, x1, step):
    return [PtWord(str(i * step), x0, 297 - i * 50, x1, 303 - i * 50) for i in range(5)]


def _line(points):
    return Shape(subpaths=(tuple(points),), closed=(False,), fill=None, stroke=BLUE, width=1.0)


def _two_axis_hint(right_step, legend_label=""):
    words = _ticks(30, 52, 10) + _ticks(390, 412, right_step)
    shapes = [_line([(80, 240), (160, 175), (240, 125)])]
    if legend_label:
        shapes.append(_line([(88, 70), (102, 70)]))
        words.append(PtWord(legend_label, 106, 67, 186, 73))
    return extract_chart_hint(words, shapes, page_size=PAGE_SIZE)


def test_dual_axis_hint_preserves_both_scales_and_labels_the_readings():
    hint = _two_axis_hint(right_step=100)

    assert [(a["side"], a["range"]) for a in hint["axes"]] == [
        ("left", [0, 40]),
        ("right", [0, 400]),
    ]
    assert [(s["axis_index"], [p["value"] for p in s["points"]]) for s in hint["series"]] == [
        (0, [12, 25, 35]),
        (1, [120, 250, 350]),
    ]
    formatted = format_chart_hint(hint)
    assert "Axis 1 (left, vertical, linear)" in formatted
    assert "Axis 2 (right, vertical, linear)" in formatted
    assert "left axis 1" in formatted
    assert "right axis 2" in formatted


def test_explicit_legend_axis_keeps_only_the_series_read_on_that_scale():
    hint = _two_axis_hint(right_step=100, legend_label="Rate (right axis)")

    assert len(hint["axes"]) == 2
    (series,) = hint["series"]
    assert series["axis_index"] == 1
    assert "Rate (right axis) (#1450b4, line, right axis 2)" in format_chart_hint(hint)


def test_mirrored_axis_and_its_series_are_deduplicated():
    hint = _two_axis_hint(right_step=10)

    (axis,) = hint["axes"]
    (series,) = hint["series"]
    assert axis["side"] == "left"
    assert series["axis_index"] == 0


def test_fifty_point_series_keeps_every_measurement():
    shape = _line((80 + 5 * i, 290 - 3 * i) for i in range(50))
    hint = extract_chart_hint(_ticks(30, 52, 10), [shape], page_size=PAGE_SIZE)

    (series,) = hint["series"]
    assert [p["x"] for p in series["points"]] == list(range(1, 51))
    assert [p["value"] for p in series["points"]] == pytest.approx([2 + 0.6 * i for i in range(50)])
    assert "50=31.4" in format_chart_hint(hint)


@pytest.mark.parametrize(
    ("label", "y"),
    [
        pytest.param("+5", 270, id="opposite-sign-in-plot"),
        pytest.param("+5", 450, id="opposite-sign-off-plot"),
        pytest.param("-5", 450, id="same-sign-off-plot"),
        pytest.param("-4", 270, id="opposite-sign-within-snap-tolerance"),
    ],
)
def test_printed_value_guard_requires_matching_sign_and_bound_plot(label, y):
    words = _ticks(30, 52, 10) + [PtWord(label, 154, y - 3, 166, y + 3)]
    shape = _line([(80, 240), (160, 275), (240, 125), (320, 260)])
    draft = (
        "<table><tr><th>Category</th><th>Value</th></tr>"
        "<tr><th>A</th><td>12</td></tr><tr><th>B</th><td>-5</td></tr>"
        "<tr><th>C</th><td>35</td></tr><tr><th>D</th><td>8</td></tr></table>"
    )
    page = parse_page(draft)
    page.width, page.height = PAGE_SIZE

    assert correct_tables(page, words, [shape], {"snap": True, "snap_tol": 0.25}) == 1
    assert page.regions[0].chosen().raw == draft.replace("<td>-5</td>", "<td>5</td>")

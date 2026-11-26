"""Unit tests for vector chart hint extraction and formatting."""

from __future__ import annotations

import pytest

from docparse.fixes.chart_hints import (
    MAX_POINTS_PER_SERIES,
    MAX_SERIES,
    extract_chart_hint,
    format_chart_hint,
)
from docparse.fixes.layout_refine import PtWord, words_in_points
from docparse.pdf.graphics import Shape, load_shapes
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf, simple_pdf

PAGE_H = 792.0


def _bar_pdf(extra_texts=()):
    """Single-series vertical bar chart (0..40, 5pt/unit from PDF y=100) with legend and title."""
    values = {"A": 12, "B": 27, "C": 35, "D": 8}
    draw = ["0.8 0.8 0.8 RG 0.5 w"]
    draw += [f"70 {y} m 400 {y} l S" for y in (100, 150, 200, 250, 300)]
    for i, v in enumerate(values.values()):
        draw.append(f"0.1 0.3 0.8 rg {100 + 70 * i} 100 30 {5 * v} re f")
    draw.append("0.1 0.3 0.8 rg 420 300 8 8 re f")
    texts = [(70, 330, 11, "Quarterly Sales Overview")]
    texts += [(52, y - 1.5, 8, str(v)) for v, y in zip((0, 10, 20, 30, 40), (100, 150, 200, 250, 300), strict=True)]
    texts += [(112 + 70 * i, 86, 8, k) for i, k in enumerate(values)]
    texts += [(432, 301, 8, "Sales"), *extra_texts]
    return make_pdf([{"draw": draw, "texts": texts}])


def _grouped_bar_pdf():
    """Two-series grouped vertical bar chart (Red and Blue side-by-side per category)."""
    red = {"Q1": 12, "Q2": 30, "Q3": 20, "Q4": 26}
    blue = {"Q1": 10, "Q2": 27, "Q3": 18, "Q4": 31}
    draw = ["0.8 0.8 0.8 RG 0.5 w"]
    draw += [f"70 {y} m 400 {y} l S" for y in (100, 150, 200, 250, 300)]
    for i, k in enumerate(red):
        draw.append(f"0.9 0.1 0.1 rg {100 + 70 * i} 100 20 {5 * red[k]} re f")
        draw.append(f"0.1 0.1 0.9 rg {122 + 70 * i} 100 20 {5 * blue[k]} re f")
    draw += ["0.9 0.1 0.1 rg 420 320 8 8 re f", "0.1 0.1 0.9 rg 420 300 8 8 re f"]
    texts = [(52, y - 1.5, 8, str(v)) for v, y in zip((0, 10, 20, 30, 40), (100, 150, 200, 250, 300), strict=True)]
    texts += [(116 + 70 * i, 86, 8, k) for i, k in enumerate(red)]
    texts += [(432, 321, 8, "Red"), (432, 301, 8, "Blue")]
    return make_pdf([{"draw": draw, "texts": texts}])


def _stacked_bar_pdf():
    """Two-series stacked vertical bar chart (bars overlapping in x)."""
    low = {"2021": 15, "2022": 20, "2023": 10}
    high = {"2021": 10, "2022": 15, "2023": 25}
    draw = ["0.8 0.8 0.8 RG 0.5 w"]
    draw += [f"70 {y} m 350 {y} l S" for y in (100, 150, 200, 250, 300)]
    for i, yr in enumerate(low):
        x = 110 + 80 * i
        h1 = 5 * low[yr]
        h2 = 5 * high[yr]
        draw.append(f"0.1 0.5 0.2 rg {x} 100 32 {h1} re f")
        draw.append(f"0.9 0.6 0.1 rg {x} {100 + h1} 32 {h2} re f")
    draw += ["0.1 0.5 0.2 rg 370 300 8 8 re f", "0.9 0.6 0.1 rg 370 280 8 8 re f"]
    texts = [(52, y - 1.5, 8, str(v)) for v, y in zip((0, 10, 20, 30, 40), (100, 150, 200, 250, 300), strict=True)]
    texts += [(118 + 80 * i, 84, 8, yr) for i, yr in enumerate(low)]
    texts += [(382, 301, 8, "Base"), (382, 281, 8, "Bonus")]
    return make_pdf([{"draw": draw, "texts": texts}])


def _horizontal_bar_pdf():
    """Horizontal bar chart (values 0..40 along bottom x-axis, categories on left)."""
    vals = [("Alpha", 15), ("Beta", 30), ("Gamma", 22)]
    draw = ["0.8 0.8 0.8 RG 0.5 w"]
    draw += [f"{x} 120 m {x} 280 l S" for x in (140, 190, 240, 290, 340)]  # 0, 10, 20, 30, 40 (5pt/unit)
    for i, (_, v) in enumerate(vals):
        y = 240 - 45 * i
        draw.append(f"0.2 0.4 0.7 rg 140 {y} {5 * v} 20 re f")
    texts = [
        (x - (2.2 if v < 10 else 4.4), 104, 8, str(v))
        for v, x in zip((0, 10, 20, 30, 40), (140, 190, 240, 290, 340), strict=True)
    ]
    texts += [(90, 247 - 45 * i, 8, name) for i, (name, _) in enumerate(vals)]
    return make_pdf([{"draw": draw, "texts": texts}])


def _line_and_marker_pdf():
    """Line series (with vertex dots) plus a separate marker-only scatter series over years 2020..2023."""
    line_vals = [(2020, 10), (2021, 25), (2022, 18), (2023, 36)]
    dot_vals = [(2020, 5), (2021, 14), (2022, 30), (2023, 22)]
    draw = ["0.8 0.8 0.8 RG 0.5 w"]
    draw += [f"70 {y} m 350 {y} l S" for y in (100, 150, 200, 250, 300)]
    xs = {yr: 100 + 70 * i for i, yr in enumerate((2020, 2021, 2022, 2023))}
    # Blue stroked line + vertex dots of same colour (must not duplicate as separate marker series)
    path = [f"0.0 0.4 0.8 RG 1.5 w {xs[line_vals[0][0]]} {100 + 5 * line_vals[0][1]} m"]
    for yr, v in line_vals[1:]:
        path.append(f"{xs[yr]} {100 + 5 * v} l")
    path.append("S")
    draw.append(" ".join(path))
    for yr, v in line_vals:
        draw.append(f"0.0 0.4 0.8 rg {xs[yr] - 2} {100 + 5 * v - 2} 4 4 re f")
    # Orange standalone markers (no connecting line)
    for yr, v in dot_vals:
        draw.append(f"0.9 0.4 0.0 rg {xs[yr] - 2.5} {100 + 5 * v - 2.5} 5 5 re f")
    # Legend keys: short blue stroke + orange square (both right-aligned at x=386)
    draw.append("0.0 0.4 0.8 RG 1.5 w 370 300 m 386 300 l S")
    draw.append("0.9 0.4 0.0 rg 380 278 6 6 re f")

    texts = [(52, y - 1.5, 8, str(v)) for v, y in zip((0, 10, 20, 30, 40), (100, 150, 200, 250, 300), strict=True)]
    texts += [(xs[yr] - 9, 84, 8, str(yr)) for yr in xs]
    texts += [(392, 298, 8, "Trend"), (392, 279, 8, "Target")]
    return make_pdf([{"draw": draw, "texts": texts}])


def _two_charts_pdf():
    """Two independent charts side by side on one page to test bbox scoping."""
    draw = []
    texts = []
    # Left chart (x ~ 40..250, PDF y ~ 100..300 => display y ~ 492..692)
    draw += ["0.8 0.8 0.8 RG 0.5 w"] + [f"65 {y} m 240 {y} l S" for y in (100, 150, 200, 250)]
    for i, v in enumerate((10, 20, 25)):
        draw.append(f"0.1 0.2 0.8 rg {85 + 50 * i} 100 20 {5 * v} re f")
    draw.append("0.1 0.2 0.8 rg 85 275 8 8 re f")
    texts += [(48, y - 1.5, 8, str(v)) for v, y in zip((0, 10, 20, 30), (100, 150, 200, 250), strict=True)]
    texts += [(97, 276, 8, "LeftSeries"), (70, 305, 10, "Left Chart Title")]

    # Right chart (x ~ 320..540, PDF y ~ 100..300)
    draw += ["0.8 0.8 0.8 RG 0.5 w"] + [f"355 {y} m 530 {y} l S" for y in (100, 150, 200, 250)]
    for i, v in enumerate((40, 80, 100)):
        draw.append(f"0.8 0.2 0.1 rg {375 + 50 * i} 100 20 {1.25 * v} re f")
    draw.append("0.8 0.2 0.1 rg 375 275 8 8 re f")
    texts += [(332, y - 1.5, 8, str(v)) for v, y in zip((0, 40, 80, 120), (100, 150, 200, 250), strict=True)]
    texts += [(387, 276, 8, "RightSeries"), (360, 305, 10, "Right Chart Title")]
    return make_pdf([{"draw": draw, "texts": texts}])


def test_single_vertical_bar_chart_rendered_and_words():
    pdf = _bar_pdf(extra_texts=[(182, 240, 7, "27")])
    rendered = render_pages(pdf, dpi=72)[0]
    shapes = load_shapes(pdf, 0)

    hint_from_page = extract_chart_hint(rendered, shapes)
    hint_from_words = extract_chart_hint(words_in_points(rendered), shapes, page_size=(rendered.width_pt, rendered.height_pt))
    assert hint_from_page == hint_from_words

    assert len(hint_from_page["axes"]) == 1
    ax = hint_from_page["axes"][0]
    assert ax["orientation"] == "vertical"
    assert ax["log"] is False
    assert ax["range"] == [0.0, 40.0]
    assert ax["step"] == pytest.approx(10.0)
    assert ax["ticks"] == [("0", 0.0), ("10", 10.0), ("20", 20.0), ("30", 30.0), ("40", 40.0)]

    assert hint_from_page["legend"] == [{"color": "#1a4dcc", "rgb": [26, 77, 204], "label": "Sales"}]

    assert len(hint_from_page["series"]) == 1
    s = hint_from_page["series"][0]
    assert s["type"] == "bar"
    assert s["layout"] == "single"
    assert s["label"] == "Sales"
    assert [(p["x"], p["value"]) for p in s["points"]] == [("A", 12.0), ("B", 27.0), ("C", 35.0), ("D", 8.0)]

    # In-plot printed label captured without duplicating axis ticks.
    printed_texts = [item["text"] for item in hint_from_page["printed_labels"]]
    assert "27" in printed_texts
    assert "40" not in printed_texts
    assert any("Quarterly Sales Overview" in line for line in hint_from_page["nearby_text"])

    formatted = format_chart_hint(hint_from_page)
    assert formatted.startswith("[Vector Chart Hint]")
    assert "Sales" in formatted
    assert "A=12, B=27, C=35, D=8" in formatted


def test_grouped_bar_chart_detects_layout_and_both_series():
    pdf = _grouped_bar_pdf()
    rendered = render_pages(pdf, dpi=72)[0]
    hint = extract_chart_hint(rendered, load_shapes(pdf, 0))

    assert {item["label"] for item in hint["legend"]} == {"Red", "Blue"}
    by_label = {s["label"]: s for s in hint["series"]}
    assert set(by_label) == {"Red", "Blue"}
    assert by_label["Red"]["layout"] == "grouped"
    assert by_label["Blue"]["layout"] == "grouped"
    assert [p["value"] for p in by_label["Red"]["points"]] == [12.0, 30.0, 20.0, 26.0]
    assert [p["value"] for p in by_label["Blue"]["points"]] == [10.0, 27.0, 18.0, 31.0]
    assert [p["x"] for p in by_label["Red"]["points"]] == ["Q1", "Q2", "Q3", "Q4"]

    formatted = format_chart_hint(hint)
    assert "grouped" in formatted
    assert "Q1=12" in formatted and "Q4=31" in formatted


def test_stacked_bar_chart_detects_overlap_and_segment_bounds():
    pdf = _stacked_bar_pdf()
    rendered = render_pages(pdf, dpi=72)[0]
    hint = extract_chart_hint(rendered, load_shapes(pdf, 0))

    by_label = {s["label"]: s for s in hint["series"]}
    assert set(by_label) == {"Base", "Bonus"}
    assert by_label["Base"]["layout"] == "stacked"
    assert by_label["Bonus"]["layout"] == "stacked"
    assert [(p["x"], p["value"], p["base"], p["top"]) for p in by_label["Base"]["points"]] == [
        (2021.0, 15.0, 0.0, 15.0),
        (2022.0, 20.0, 0.0, 20.0),
        (2023.0, 10.0, 0.0, 10.0),
    ]
    assert [(p["x"], p["value"], p["base"], p["top"]) for p in by_label["Bonus"]["points"]] == [
        (2021.0, 10.0, 15.0, 25.0),
        (2022.0, 15.0, 20.0, 35.0),
        (2023.0, 25.0, 10.0, 35.0),
    ]

    formatted = format_chart_hint(hint)
    assert "stacked" in formatted
    assert "2023=25 (10..35)" in formatted


def test_horizontal_bar_chart_measures_values_and_left_labels():
    pdf = _horizontal_bar_pdf()
    rendered = render_pages(pdf, dpi=72)[0]
    hint = extract_chart_hint(rendered, load_shapes(pdf, 0))

    assert len(hint["axes"]) == 1
    assert hint["axes"][0]["orientation"] == "horizontal"
    assert len(hint["series"]) == 1
    s = hint["series"][0]
    assert s["orientation"] == "horizontal"
    assert s["type"] == "bar"
    assert [(p["x"], p["value"]) for p in s["points"]] == [("Alpha", 15.0), ("Beta", 30.0), ("Gamma", 22.0)]


def test_line_series_with_year_axis_and_standalone_markers():
    pdf = _line_and_marker_pdf()
    rendered = render_pages(pdf, dpi=72)[0]
    hint = extract_chart_hint(rendered, load_shapes(pdf, 0))

    assert len(hint["axes"]) == 2
    assert {a["orientation"] for a in hint["axes"]} == {"vertical", "horizontal"}

    by_label = {s["label"]: s for s in hint["series"]}
    # Blue line's own vertex dots must NOT emit a duplicate marker series.
    assert len(hint["series"]) == 2
    assert by_label["Trend"]["type"] == "line"
    assert [(p["x"], p["value"]) for p in by_label["Trend"]["points"]] == [
        (2020.0, 10.0),
        (2021.0, 25.0),
        (2022.0, 18.0),
        (2023.0, 36.0),
    ]
    assert by_label["Target"]["type"] == "marker"
    assert [(p["x"], p["value"]) for p in by_label["Target"]["points"]] == [
        (2020.0, 5.0),
        (2021.0, 14.0),
        (2022.0, 30.0),
        (2023.0, 22.0),
    ]


def test_log_scale_axis_evaluates_line_values():
    words = [PtWord(str(10**k), 40.0, 300.0 - 40.0 * k, 65.0, 308.0 - 40.0 * k) for k in range(1, 5)]
    # y=264 -> 10, y=224 -> 100, y=184 -> 1000, y=144 -> 10000
    line_shape = Shape(
        subpaths=(((100.0, 264.0), (160.0, 224.0), (220.0, 184.0), (280.0, 144.0)),),
        closed=(False,),
        fill=None,
        stroke=(20, 100, 200),
        width=1.5,
    )
    hint = extract_chart_hint(words, [line_shape], page_size=(612.0, 792.0))
    assert len(hint["axes"]) == 1
    assert hint["axes"][0]["log"] is True
    assert hint["axes"][0]["step"] is None
    assert len(hint["series"]) == 1
    vals = [p["value"] for p in hint["series"][0]["points"]]
    assert vals == pytest.approx([10.0, 100.0, 1000.0, 10000.0], rel=0.02)
    assert "log" in format_chart_hint(hint)


def test_bbox_scopes_extraction_to_target_chart_only():
    pdf = _two_charts_pdf()
    rendered = render_pages(pdf, dpi=72)[0]
    shapes = load_shapes(pdf, 0)

    # Display y = 792 - pdf_y; PDF y in [100..300] -> display y in [492..692].
    left_hint = extract_chart_hint(rendered, shapes, bbox=(40.0, 480.0, 260.0, 710.0))
    right_hint = extract_chart_hint(rendered, shapes, bbox=(320.0, 480.0, 550.0, 710.0))

    assert len(left_hint["axes"]) == 1 and left_hint["axes"][0]["range"] == [0.0, 30.0]
    assert [item["label"] for item in left_hint["legend"]] == ["LeftSeries"]
    assert [p["value"] for p in left_hint["series"][0]["points"]] == [10.0, 20.0, 25.0]
    assert any("Left Chart Title" in ln for ln in left_hint["nearby_text"])
    assert not any("Right Chart Title" in ln for ln in left_hint["nearby_text"])

    assert len(right_hint["axes"]) == 1 and right_hint["axes"][0]["range"] == [0.0, 120.0]
    assert [item["label"] for item in right_hint["legend"]] == ["RightSeries"]
    assert [p["value"] for p in right_hint["series"][0]["points"]] == [40.0, 80.0, 100.0]
    assert any("Right Chart Title" in ln for ln in right_hint["nearby_text"])
    assert not any("Left Chart Title" in ln for ln in right_hint["nearby_text"])


def test_caps_series_points_and_max_chars():
    # Vertical axis 0..100 (y=300..100 -> 2pt/unit)
    words = [PtWord(str(v), 30.0, 300.0 - 2.0 * v - 4.0, 52.0, 300.0 - 2.0 * v + 4.0) for v in range(0, 101, 20)]
    shapes = []
    # 16 colored lines, each with 80 points
    for s_idx in range(16):
        pts = tuple((80.0 + 4.0 * i, 290.0 - (i % 40) * 3.0 - s_idx * 2.0) for i in range(80))
        shapes.append(
            Shape(
                subpaths=(pts,),
                closed=(False,),
                fill=None,
                stroke=((s_idx * 37) % 240, (s_idx * 73) % 240, (s_idx * 113) % 240),
                width=1.0,
            )
        )
    hint = extract_chart_hint(words, shapes, page_size=(612.0, 792.0))
    assert len(hint["series"]) == MAX_SERIES
    assert hint["omitted_series"] == 16 - MAX_SERIES
    for s in hint["series"]:
        assert len(s["points"]) == MAX_POINTS_PER_SERIES
        assert s["total_points"] == 80

    for limit in (120, 400, 1200, 2500):
        out = format_chart_hint(hint, max_chars=limit)
        assert 0 < len(out) <= limit
        assert not out.rstrip().endswith(":")  # never leaves a dangling section header


def test_empty_raster_and_non_chart_pages_return_empty_hint():
    # 1. Completely empty inputs
    empty = extract_chart_hint([], [])
    assert empty == {"axes": [], "legend": [], "series": [], "printed_labels": [], "nearby_text": []}
    assert format_chart_hint(empty) == ""

    # 2. Plain text PDF with no shapes (or raster page with no vector paths)
    pdf = simple_pdf("Annual Report Summary 2025")
    rendered = render_pages(pdf, dpi=72)[0]
    plain_hint = extract_chart_hint(rendered, load_shapes(pdf, 0))
    assert format_chart_hint(plain_hint) == ""

    # 3. Single decorative bar with no axes and no legend
    deco_pdf = make_pdf([{"draw": ["0.1 0.2 0.5 rg 50 700 500 10 re f"], "texts": [(50, 720, 12, "Page Header")]}])
    deco_rendered = render_pages(deco_pdf, dpi=72)[0]
    deco_hint = extract_chart_hint(deco_rendered, load_shapes(deco_pdf, 0))
    assert format_chart_hint(deco_hint) == ""

    # 4. Edge-case inputs to format_chart_hint
    assert format_chart_hint({}) == ""
    assert format_chart_hint({"axes": [], "legend": [], "series": [], "nearby_text": ["Just text"]}) == ""
    assert format_chart_hint({"axes": [{"orientation": "vertical", "range": [0, 10], "ticks": []}], "legend": [], "series": []}, max_chars=0) == ""

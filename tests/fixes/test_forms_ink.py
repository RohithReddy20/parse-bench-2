"""Ink-ratio check-box verifier (`docparse.fixes.forms_ink`) on synthetic renders and hand-built PDFs."""

import pytest
from PIL import Image, ImageDraw

from docparse.fixes.forms import apply_edits
from docparse.fixes.forms_ink import (
    _box_ink_stats,
    box_ink_ratio,
    checkbox_candidates,
    classify_box_ink,
    ink_checkbox_edits,
    repair_checkbox_ink,
    vector_checkbox_rects,
)
from docparse.pdf.render import RenderedPage, Word, render_pages
from tests.pdfgen import make_pdf

DPI = 288
PX = DPI / 72  # 4 pixels per point
BLACK = (0, 0, 0)
BOX_A, BOX_B = (20.0, 20.0, 30.0, 30.0), (60.0, 20.0, 70.0, 30.0)  # 10 pt boxes, top-down points
BOX_C = (20.0, 36.0, 30.0, 46.0)  # a third, unlabelled box below BOX_A
LABELS = [("Yes", (34.0, 21.0, 48.0, 29.0)), ("No", (74.0, 21.0, 84.0, 29.0))]  # each just right of its box
BOX_JA, BOX_NEIN = (31.0, 20.0, 41.0, 30.0), (69.0, 20.0, 79.0, 30.0)
RIGHT_LABELS = [("ja", (20.0, 21.0, 28.0, 29.0)), ("nein", (50.0, 21.0, 66.0, 29.0))]  # each just left of its box


def _px(box):
    return tuple(v * PX for v in box)


def _canvas(shade=None):
    """A 100 x 50 pt white page; with `shade`, a grey band of that luminance behind both boxes."""
    img = Image.new("RGB", (round(100 * PX), round(50 * PX)), (255, 255, 255))
    if shade is not None:
        ImageDraw.Draw(img).rectangle([0, 60, img.width - 1, 140], fill=(shade, shade, shade))
    return img


def _draw(img, box, mark=None):
    """A 1 pt black outline; mark: None (empty), "x", "tick", "speck" (7 tiny dots) or "solid" (a ■ bullet)."""
    x0, y0, x1, y1 = (round(v) for v in _px(box))
    d = ImageDraw.Draw(img)
    if mark == "solid":
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=BLACK)
        return
    d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=BLACK, width=4)
    if mark == "x":
        d.line([x0 + 6, y0 + 6, x1 - 7, y1 - 7], fill=BLACK, width=4)
        d.line([x0 + 6, y1 - 7, x1 - 7, y0 + 6], fill=BLACK, width=4)
    elif mark == "tick":
        d.line([(x0 + 8, y0 + 22), (x0 + 16, y0 + 30), (x0 + 32, y0 + 10)], fill=BLACK, width=4)
    elif mark == "speck":  # 7 dots of 2 x 2 px in the 22 x 22 px interior: ink ~0.058
        for k in range(7):
            d.rectangle([x0 + 11 + 3 * k, y0 + 12 + 2 * k, x0 + 12 + 3 * k, y0 + 13 + 2 * k], fill=BLACK)


def _shade(img, box, lum):
    """Fill a drawn box inside its outline row by row with grey `lum(row)`: a shaded or coloured section box."""
    x0, y0, x1, y1 = (round(v) for v in _px(box))
    d = ImageDraw.Draw(img)
    for r in range(y1 - y0 - 8):
        v = int(lum(r))
        d.line([(x0 + 4, y0 + 4 + r), (x1 - 5, y0 + 4 + r)], fill=(v, v, v))


def _page(img, labels=LABELS):
    words = [Word(text, list(_px(box))) for text, box in labels]
    return RenderedPage(image=img, dpi=DPI, width_pt=100, height_pt=50, text_words=words)


def _fix(md, img, boxes=(BOX_A, BOX_B), labels=LABELS, **opts):
    return apply_edits(md, ink_checkbox_edits(md, _page(img, labels), b"", boxes=list(boxes), **opts))


@pytest.mark.parametrize(
    ("ink", "want"),
    [
        (0.0, False),
        (0.025, False),
        (0.06, None),
        (0.119, None),
        (0.12, True),
        (0.5, True),
        (0.58, True),
        (0.62, None),
        (0.82, None),
    ],
)
def test_classify_box_ink_is_tri_state(ink, want):
    assert classify_box_ink(ink) is want


@pytest.mark.parametrize(
    ("ink", "std", "want"), [(0.3, 40.0, True), (0.3, 28.0, True), (0.3, 27.9, None), (0.01, 0.0, False)]
)
def test_classify_box_ink_needs_a_contrasted_interior(ink, std, want):
    assert classify_box_ink(ink, std) is want


def test_empty_box_unchecks_a_false_mark():
    img = _canvas()
    _draw(img, BOX_A)
    _draw(img, BOX_B)
    assert box_ink_ratio(img, _px(BOX_A)) == 0.0
    assert _fix("[x] Yes [ ] No", img) == "[ ] Yes [ ] No"


@pytest.mark.parametrize("mark", ["x", "tick"])
def test_ticked_box_checks_a_missed_mark(mark):
    img = _canvas()
    _draw(img, BOX_A)
    _draw(img, BOX_B, mark)
    ink, std = _box_ink_stats(img, _px(BOX_B))
    assert std >= 30 and classify_box_ink(ink, std) is True
    assert _fix("[ ] Yes [ ] No", img) == "[ ] Yes [x] No"


def test_grey_band_calibrates_on_the_local_background():
    img = _canvas(shade=150)  # a fixed "darker than 200" rule would read every box interior here as ink
    _draw(img, BOX_A)
    _draw(img, BOX_B, "x")
    assert box_ink_ratio(img, _px(BOX_A)) == 0.0
    assert _fix("[ ] Yes [x] No", img) == "[ ] Yes [x] No"
    assert _fix("[x] Yes [ ] No", img) == "[ ] Yes [x] No"


def test_solid_bullet_is_never_read_as_a_tick():
    img = _canvas()
    _draw(img, BOX_A, "solid")
    _draw(img, BOX_B)
    ink = box_ink_ratio(img, _px(BOX_A))
    assert ink > 0.85 and classify_box_ink(ink) is None
    assert _fix("[ ] Yes [ ] No", img) == "[ ] Yes [ ] No"


def test_shaded_boxes_are_never_read_as_ticks():
    img = _canvas()
    # BOX_A: a dark solid fill; BOX_B: a light gradient whose darker third crosses the ink threshold
    for box, lum in ((BOX_A, lambda r: 120), (BOX_B, lambda r: 240 - 1.5 * r)):
        _draw(img, box)
        _shade(img, box, lum)
    dark, (ink, std) = _box_ink_stats(img, _px(BOX_A)), _box_ink_stats(img, _px(BOX_B))
    assert dark[0] > 0.58 and classify_box_ink(*dark) is None
    # by ink alone the gradient reads as a tick; its flat interior does not
    assert classify_box_ink(ink) is True and std < 28 and classify_box_ink(ink, std) is None
    assert _fix("[ ] Yes [ ] No", img) == "[ ] Yes [ ] No"
    assert _fix("[x] Yes [ ] No", img) == "[x] Yes [ ] No"


@pytest.mark.parametrize(
    ("glyph", "want"), [("6", "[ ] Yes"), ("01", "[ ] Yes"), ("X", "[x] Yes"), (None, "[x] Yes")]
)
def test_box_around_a_printed_field_number_is_not_a_check_box(glyph, want):
    img = _canvas()
    _draw(img, BOX_A, "x")  # the printed number's ink reads like a tick
    _draw(img, BOX_B)
    _draw(img, BOX_C)
    labels = LABELS + ([(glyph, (23.0, 22.0, 27.0, 28.0))] if glyph else [])  # centred in BOX_A
    assert _fix("[ ] Yes [ ] No", img, boxes=(BOX_A, BOX_B, BOX_C), labels=labels) == f"{want} [ ] No"


@pytest.mark.parametrize(
    ("md", "want"),
    [
        ("ja [ ] nein [ ]", "ja [ ] nein [x]"),
        ("**ja** [ ] **nein**: [ ]", "**ja** [ ] **nein**: [x]"),
        ("ja [x] nein [ ]", "ja [ ] nein [x]"),
    ],
)
def test_box_right_of_its_label_sets_the_mark_after_it(md, want):
    img = _canvas()
    _draw(img, BOX_JA)
    _draw(img, BOX_NEIN, "x")
    assert _fix(md, img, boxes=(BOX_JA, BOX_NEIN), labels=RIGHT_LABELS) == want


def test_touching_comb_boxes_are_all_excluded_but_a_touching_pair_is_kept():
    comb = [(100.0 + 10 * k, 50.0, 110.0 + 10 * k, 60.0) for k in range(5)]
    pair = [(200.0, 50.0, 210.0, 60.0), (210.0, 50.0, 220.0, 60.0)]
    assert checkbox_candidates([*comb, BOX_A, BOX_B, *pair]) == [BOX_A, BOX_B, *pair]


def test_double_strokes_merge_and_cell_grids_and_large_boxes_drop():
    inner = (21.0, 21.0, 29.0, 29.0)  # the inner outline of a double-stroked BOX_A
    grid = [(100.0 + 30 * c, 100.0 + 25 * r, 110.0 + 30 * c, 110.0 + 25 * r) for c in range(4) for r in range(3)]
    cell = (200.0, 20.0, 240.0, 40.0)
    assert checkbox_candidates([inner, BOX_A, *grid, cell, BOX_B]) == [BOX_A, BOX_B]


def test_faint_speckle_is_ambiguous_and_keeps_the_read():
    img = _canvas()
    _draw(img, BOX_A, "speck")
    _draw(img, BOX_B)
    ink = box_ink_ratio(img, _px(BOX_A))
    assert 0.04 < ink < 0.08 and classify_box_ink(ink) is None
    assert _fix("[x] Yes [ ] No", img) == "[x] Yes [ ] No"
    assert _fix("[ ] Yes [ ] No", img) == "[ ] Yes [ ] No"


def test_needs_two_boxes_and_leaves_acroform_widgets_alone():
    img = _canvas()
    _draw(img, BOX_A)
    _draw(img, BOX_B, "x")
    md = "[x] Yes [ ] No"
    assert _fix(md, img, boxes=[BOX_A]) == md
    assert _fix(md, img, skip_boxes=[(*BOX_A, True)]) == "[x] Yes [x] No"


def test_vector_checkbox_rects_from_a_pdf_merge_double_strokes_and_drop_combs():
    comb = [(300 + 11 * k, 500, 10, 10) for k in range(6)]  # a 6-character field, boxes 1 pt apart
    rects = [(100, 600, 9, 9), (200, 600, 10, 10), (201, 601, 8, 8), *comb, (400, 600, 60, 20)]
    got = sorted(vector_checkbox_rects(make_pdf([{"rects": rects}]), 0))
    assert got == [pytest.approx((100, 183, 109, 192)), pytest.approx((200, 182, 210, 192))]
    assert vector_checkbox_rects(b"\x89PNG\r\n\x1a\n" + b"\0" * 32, 0) == []


def test_repair_checkbox_ink_end_to_end_is_idempotent():
    x_mark = "0.6 w 106.5 700.5 m 112.5 706.5 l S 106.5 706.5 m 112.5 700.5 l S"  # a thin X inside the first box
    page = {
        "texts": [(120, 700, 11, "Yes"), (220, 700, 11, "No")],
        "rects": [(105, 699, 9, 9), (205, 699, 9, 9)],
        "draw": x_mark,
    }
    data = make_pdf([page])
    rendered = render_pages(data)[0]
    once = repair_checkbox_ink("[ ] Yes [x] No", rendered, data, 0)
    assert once == "[x] Yes [ ] No"
    assert repair_checkbox_ink(once, rendered, data, 0) == once

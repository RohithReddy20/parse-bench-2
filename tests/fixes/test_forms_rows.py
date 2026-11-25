from docparse.fixes.forms import repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def _fix(md: str, printed, typed, options) -> str:
    data = make_pdf([{"texts": printed, "overlay": typed}])
    return repair(md, render_pages(data)[0], overlay_fonts(data), options)


LINE = [(40, 700, 10, "8"), (60, 700, 10, "Advertising"), (300, 700, 10, "8")]
VALUE = [(330, 700, 10, "1,597")]


def test_a_line_number_cell_joins_its_caption_and_its_repeat_at_the_box_goes():
    md = "<table><tr><td><strong>8</strong></td><td>Advertising</td><td><strong>8</strong></td><td>1,597</td></tr></table>"
    out = "<table><tr><td></td><td><strong>8</strong> Advertising</td><td></td><td>1,597</td></tr></table>"
    assert _fix(md, LINE, VALUE, {"ids": True}) == out
    assert _fix(md, LINE, VALUE, {"combs": True}) == md  # off unless asked


def test_a_typed_number_is_a_value_not_a_line_number():
    printed = [(40, 700, 10, "Anzahl"), (200, 700, 10, "Material")]
    typed = [(120, 700, 10, "8"), (280, 700, 10, "Stahl")]
    md = "<table><tr><td>Anzahl</td><td>8</td><td>Material</td><td>Stahl</td></tr></table>"
    assert _fix(md, printed, typed, {"ids": True}) == md


def test_grids_with_captioned_value_columns_or_several_slots_per_caption_are_left_alone():
    printed = [(300, 720, 10, "Plus-value"), *LINE[:2]]
    head = "<table><tr><th></th><th></th><th>Plus-value</th></tr>"
    md = head + "<tr><td>8</td><td>Advertising</td><td>1,597</td></tr></table>"
    assert _fix(md, printed, VALUE, {"ids": True, "codes": True}) == md
    md = "<table><tr><td>8</td><td>Advertising</td><td>1,597</td><td></td><td></td></tr></table>"
    assert _fix(md, LINE[:2], VALUE, {"ids": True}) == md


def test_a_box_code_between_a_caption_and_its_typed_value_moves_to_the_caption():
    printed = [(40, 700, 10, "Salaries"), (250, 700, 10, "9060")]
    typed = [(330, 700, 10, "2,149.10")]
    md = "<table><tr><td>Salaries</td><td>9060</td><td>2,149.10</td></tr></table>"
    out = "<table><tr><td>9060 Salaries</td><td></td><td>2,149.10</td></tr></table>"
    assert _fix(md, printed, typed, {"codes": True}) == out
    printed = [(40, 700, 10, "Salaries"), (250, 700, 10, "9060"), (330, 700, 10, "Wages")]
    md = "<table><tr><td>Salaries</td><td>9060</td><td>Wages</td></tr></table>"  # no typed value after the code
    assert _fix(md, printed, typed, {"codes": True}) == md


def test_character_box_outlines_before_a_typed_value_are_dropped():
    printed = [(40, 720, 10, "Total rents"), (40, 700, 10, "\xa3")]
    typed = [(120, 700, 10, "7060")]
    md = "**14** **Total rents**\n\xa3 [ ] [ ] [ ] 7060\n\n[ ] Yes"
    assert _fix(md, printed, typed, {"outlines": True}) == "**14** **Total rents**\n\xa3 7060\n\n[ ] Yes"


def test_a_grid_of_check_boxes_under_option_captions_becomes_one_line_per_row():
    from docparse.fixes.forms_grids import box_grid_lines

    md = (
        '<table><tr><th rowspan="2">How would you rate it?</th><th colspan="3">Most days</th></tr>'
        "<tr><th>Good</th><th>Fair</th><th>Poor</th></tr>"
        "<tr><td>1. Walk</td><td>[ ]</td><td>[x]</td><td>[ ]</td></tr>"
        "<tr><td>2. Read</td><td>☐</td><td>☐</td><td>☒</td></tr></table>"
    )
    out = "How would you rate it? Most days\n1. Walk: [ ] Good [x] Fair [ ] Poor\n2. Read: [ ] Good [ ] Fair [x] Poor"
    assert box_grid_lines(md) == out
    plain = "<table><tr><th>A</th><th>B</th></tr><tr><td>1. Walk</td><td>12</td></tr></table>"
    assert box_grid_lines(plain) is None  # values, not boxes
    printed = [(40, 700, 10, "1."), (52, 700, 10, "Walk"), (200, 720, 10, "Good"), (260, 720, 10, "Fair")]
    md = "<table><tr><th></th><th>Good</th><th>Fair</th></tr><tr><td>1. Walk</td><td>[ ]</td><td>[x]</td></tr></table>"
    assert _fix(md, printed, [(262, 700, 10, "X")], {"grids": True}) == "1. Walk: [ ] Good [x] Fair"

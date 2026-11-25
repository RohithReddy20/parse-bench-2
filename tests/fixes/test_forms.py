import pytest

from docparse.core.config import Config
from docparse.fixes.forms import fills, page_words, read_form, repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf

FORM = [(72, 700, 11, "Family name"), (72, 660, 11, "Postcode"), (72, 620, 11, "Date of birth")]
COMB = [(200 + 18 * i, 660, 11, ch) for i, ch in enumerate("B37PW")]
DATE = [(200 + 18 * i + (24 if i >= 2 else 0) + (24 if i >= 4 else 0), 620, 11, ch) for i, ch in enumerate("06121930")]
FILL = [(200, 700, 11, "TRIVEDI"), *COMB, *DATE]


def _page(texts=FORM, overlay=FILL):
    data = make_pdf([{"texts": texts, "overlay": overlay}])
    return render_pages(data)[0], overlay_fonts(data)


def test_typed_values_are_grouped_into_fills():
    rendered, fonts = _page()
    printed, overlay = page_words(rendered, fonts)
    assert [f.text for f in fills(overlay, printed)] == ["TRIVEDI", "B37PW", "06", "12", "1930"]


def test_a_comb_spelled_glyph_by_glyph_is_written_as_one_value():
    rendered, fonts = _page()
    md = "**Family name**: TRIVEDI\n\n**Postcode**: B 3 7 P W\n\n**Date of birth**: 0 6 1 2 1 9 3 0"
    out = repair(md, rendered, fonts, {"combs": True})
    assert out == "**Family name**: TRIVEDI\n\n**Postcode**: B37PW\n\n**Date of birth**: 06 12 1930"


def test_comb_glyphs_in_bold_runs_tags_or_cells_keep_the_markup_balanced():
    rendered, fonts = _page()
    md = "**Postcode**: **B** **3** **7** **P** **W**\n\n<td>0</td><td>6</td><td>x</td>"
    out = '**Postcode**: **B37PW**\n\n<td colspan="2">06</td><td>x</td>'
    assert repair(md, rendered, fonts, {"combs": True}) == out
    md = "Postcode <u>B</u> <u>3</u> [ ] <u>7</u> [ ] <u>P</u> <u>W</u>"  # character boxes read as check boxes
    assert repair(md, rendered, fonts, {"combs": True}) == "Postcode <u>B37PW</u>"
    md = "Postcode: B 3 | 7 P W"  # a pipe is a column: not joined
    assert repair(md, rendered, fonts, {"combs": True}) == md


def test_a_printed_separator_inside_a_comb_stays():
    typed = [(200 + 18 * i, 660, 11, ch) for i, ch in enumerate("17PW")]
    rendered, fonts = _page([(72, 660, 11, "Amount"), (211, 658, 11, ", ")], typed)
    out = repair("Amount: 1 , 7 P W", rendered, fonts, {"combs": True})
    assert out == "Amount: 1,7PW"


def test_dot_leaders_and_typewriter_spaces_do_not_split_a_typed_value():
    leader = [(150, 700, 11, ". " * 30)]
    typed = [(160, 700, 11, "County"), (206.6, 700, 11, "Court")]  # Courier: a space as wide as a glyph
    rendered, fonts = _page([(72, 700, 11, "In the"), *leader], typed)
    printed, overlay = page_words(rendered, fonts)
    assert [f.text for f in fills(overlay, printed)] == ["County Court"]


def test_pages_without_an_overlay_or_options_come_back_unchanged():
    md = "**Postcode**: B 3 7 P W"
    rendered, fonts = _page()
    assert repair(md, rendered, fonts, {}) is md
    plain = make_pdf([{"texts": FORM + FILL}])
    assert repair(md, render_pages(plain)[0], overlay_fonts(plain), {"combs": True}) is md


def test_fills_are_placed_after_their_label_even_when_values_repeat():
    rendered, fonts = _page(
        [(72, 700, 11, "Mother"), (72, 660, 11, "Father")], [(200, 700, 11, "SMITH"), (200, 660, 11, "SMITH")]
    )
    printed, overlay = page_words(rendered, fonts)
    md = "Mother: SMITH\nFather: SMITH"
    page = read_form(md, printed, overlay)
    assert [f.words[0].cy < 100 for f in page.fills] == [True, False]  # fill 0 is the mother's
    assert page.placed == {0: (1, 1), 1: (3, 3)}  # tokens: Mother SMITH Father SMITH


def test_form_options_are_validated():
    assert Config.load("configs/standin.yaml", ["form_options.combs=true"]).form_options == {"combs": True}
    with pytest.raises(ValueError, match="unknown form option"):
        Config.load("configs/standin.yaml", ["form_options.guess=true"])

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


def test_comb_glyphs_across_cells_or_bold_runs_are_handled_safely():
    rendered, fonts = _page()
    md = "**Postcode**: **B** **3** **7** **P** **W**\n\n<td>0</td><td>6</td>"
    assert repair(md, rendered, fonts, {"combs": True}) == "**Postcode**: **B37PW**\n\n<td>0</td><td>6</td>"


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

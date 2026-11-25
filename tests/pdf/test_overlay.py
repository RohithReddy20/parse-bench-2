from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf

FORM = [(72, 700, 11, "Family name"), (72, 660, 11, "Given names")]
FILL = [(200, 700, 11, "TRIVEDI"), (200, 660, 11, "LAYLA ELIZABETH")]


def test_text_added_on_top_of_a_finished_page_is_the_overlay():
    assert overlay_fonts(make_pdf([{"texts": FORM, "overlay": FILL}])) == frozenset({"Courier"})


def test_a_page_written_in_one_go_has_no_overlay():
    assert overlay_fonts(make_pdf([{"texts": FORM + FILL}])) == frozenset()
    assert overlay_fonts(b"not a pdf") == frozenset()


def test_words_carry_their_font_so_overlay_words_can_be_told_apart():
    data = make_pdf([{"texts": FORM, "overlay": FILL}])
    words = render_pages(data)[0].text_words
    fonts = overlay_fonts(data)
    assert [w.text for w in words if w.font in fonts] == ["TRIVEDI", "LAYLA", "ELIZABETH"]
    assert [w.text for w in words if w.font not in fonts] == ["Family", "name", "Given", "names"]

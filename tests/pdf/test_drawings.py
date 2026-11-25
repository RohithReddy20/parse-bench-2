from docparse.pdf.drawings import horizontal_lines
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf

# Helvetica 12 pt at (72, 700): "Hello" runs from x 72 to 99, "world" from 103 to 131; the baseline is y 700
TEXT = [(72, 700, 12, "Hello world")]


def _words(draw: str, decorations: bool = True):
    (page,) = render_pages(make_pdf([{"texts": TEXT, "draw": draw}]), decorations=decorations)
    return {w.text: w.marks for w in page.text_words}


def test_a_stroked_line_through_a_word_strikes_it():
    assert _words("0.8 w 102 703.5 m 131 703.5 l S") == {"Hello": ".....", "world": "sssss"}


def test_a_thin_filled_rectangle_through_a_word_strikes_it():
    assert _words("102 703.1 29 0.8 re f") == {"Hello": ".....", "world": "sssss"}


def test_a_line_just_below_the_baseline_underlines():
    assert _words("0.8 w 72 698 m 99 698 l S") == {"Hello": "uuuuu", "world": "....."}


def test_a_rule_running_far_beyond_the_text_marks_nothing():
    assert _words("0.8 w 36 703.5 m 576 703.5 l S") == {"Hello": ".....", "world": "....."}


def test_a_thick_bar_is_not_a_line():
    assert _words("102 699 29 8 re f") == {"Hello": ".....", "world": "....."}


def test_marks_are_only_computed_on_request():
    assert _words("0.8 w 102 703.5 m 131 703.5 l S", decorations=False) == {"Hello": "", "world": ""}


def test_horizontal_lines_in_page_space():
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(make_pdf([{"texts": TEXT, "draw": "0.8 w 102 703.5 m 131 703.5 l S 50 50 10 200 re f"}]))
    (line,) = horizontal_lines(pdf[0])  # the tall rectangle is not a line
    assert abs(line.x0 - 102) < 1 and abs(line.x1 - 131) < 1 and abs(line.y - 703.5) < 0.5 and line.thick < 1.5
    pdf.close()

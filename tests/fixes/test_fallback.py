from docparse.fixes.fallback import looped, reading_recall, text_dump
from docparse.fixes.layout_refine import words_in_points
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf

LINES = [(72, 700, 11, "Family name"), (72, 686, 11, "Given names"), (72, 640, 11, "Date of birth")]


def _words():
    return words_in_points(render_pages(make_pdf([{"texts": LINES}]))[0])


def test_recall_is_the_share_of_pdf_tokens_the_reading_holds():
    words = _words()
    assert reading_recall("Family name\nGiven names\n**Date of birth**", words) == 1.0
    assert reading_recall("Family name", words) == 2 / 7
    assert reading_recall("name name name", words) == 1 / 7  # a repeated word counts once per PDF occurrence
    assert reading_recall("", words) == 0.0
    assert reading_recall("anything", []) == 1.0  # no text layer: nothing to compare against


def test_text_dump_is_one_line_per_printed_line_with_gaps_as_paragraphs():
    assert text_dump(_words()) == "Family name\nGiven names\n\nDate of birth"


def test_looped_reads_compress_far_better_than_real_pages():
    assert looped("<td></td>" * 2000)
    assert not looped("<td></td>" * 100)  # too short to judge
    prose = " ".join(f"word{i} value{i * 7} and some more text number {i}" for i in range(400))
    assert not looped(prose)

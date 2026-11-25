import pytest

from docparse.base.ocria_parse import parse_page
from docparse.core.config import Config
from docparse.fixes.text import refine_text
from docparse.pdf.render import render_pages
from docparse.write.markdown import write_page
from tests.pdfgen import make_pdf

LINE = "the quick brown foxes jumps over the lazy dogs"


def _rendered(text: str = LINE, **page):
    pdf = make_pdf([{"texts": [(72, 700, 12, text)] if text else [], **page}])
    return render_pages(pdf, layer=True)[0]


def _fixed(md: str, rendered, options=None) -> str:
    page = parse_page(md)
    refine_text(page, rendered, options or {"respell": True})
    return write_page(page)


def test_a_misread_word_takes_the_text_layer_spelling():
    assert _fixed("# Fox\n\nthe quick brown foxas jumps over the lazy dogs", _rendered()) == (
        "# Fox\n\nthe quick brown foxes jumps over the lazy dogs"
    )


def test_a_misread_number_of_the_same_length_is_corrected():
    rendered = _rendered("they paid 1234 dollars to the firm last year")
    assert _fixed("they paid 1284 dollars to the firm last year", rendered) == (
        "they paid 1234 dollars to the firm last year"
    )
    assert _fixed("they paid 12345 dollars to the firm last year", rendered) == (
        "they paid 12345 dollars to the firm last year"
    )  # a different length is another number, not a misread digit


def test_words_without_context_or_far_from_the_layer_stay():
    rendered = _rendered()
    assert _fixed("quick brown foxas", rendered) == "quick brown foxas"  # under two matching words on one side
    assert _fixed("the quick brown wolves jumps over the lazy dogs", rendered) == (
        "the quick brown wolves jumps over the lazy dogs"
    )


def test_ligature_and_case_artefacts_of_the_layer_are_not_copied():
    rendered = _rendered("the oAcial report says this was the case")
    assert _fixed("the official report says this was the case", rendered) == (
        "the official report says this was the case"
    )
    rendered = _rendered("the ocial report says this was the case")
    assert _fixed("the official report says this was the case", rendered) == (
        "the official report says this was the case"
    )


def test_an_invisible_text_layer_is_not_trusted():
    hidden = f"BT 3 Tr /F1 12 Tf 72 700 Td ({LINE}) Tj ET"
    rendered = _rendered("", under=hidden)
    assert [w.text for w in rendered.text_words][:2] == ["the", "quick"]
    assert _fixed("the quick brown foxas jumps over the lazy dogs", rendered) == (
        "the quick brown foxas jumps over the lazy dogs"
    )


def test_untouched_regions_keep_their_base_candidate():
    page = parse_page("Intro line\n\nthe quick brown foxas jumps over the lazy dogs")
    refine_text(page, _rendered(), {"respell": True})
    intro, body = page.regions
    assert len(intro.candidates) == 1 and body.chosen().reader == "fix.text"


def test_text_options_are_validated():
    cfg = Config.load("configs/standin.yaml", ["text_options.respell=true"])
    assert cfg.text_options == {"respell": True}
    with pytest.raises(ValueError, match="unknown text option"):
        Config.load("configs/standin.yaml", ["text_options.guess=true"])

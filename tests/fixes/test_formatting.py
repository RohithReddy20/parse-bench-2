import pytest

from docparse.base.ocria_parse import parse_page
from docparse.core.config import Config
from docparse.fixes.formatting import refine_format
from docparse.pdf.render import render_pages
from docparse.write.markdown import write_page
from tests.pdfgen import make_pdf

STRUCK_WORLD = "0.8 w 102 703.5 m 131 703.5 l S"  # through "world" of "Hello world" at (72, 700), 12 pt


def _rendered(draw: str = STRUCK_WORLD, text: str = "Hello world"):
    return render_pages(make_pdf([{"texts": [(72, 700, 12, text)], "draw": draw}]), decorations=True)[0]


def _fixed(md: str, options: dict, rendered=None) -> str:
    page = parse_page(md)
    refine_format(page, rendered, options)
    return write_page(page)


def test_strikes_inside_html_cells_become_tags():
    md = "Intro ~~kept~~ as is\n\n<table><tr><td>~~old~~ new</td><td>~~a b~~</td></tr></table>\n"
    assert _fixed(md, {"cell_strike": True}) == (
        "Intro ~~kept~~ as is\n\n<table><tr><td><s>old</s> new</td><td><s>a b</s></td></tr></table>\n"
    )


def test_untouched_blocks_keep_their_base_candidate():
    page = parse_page("Intro ~~kept~~\n\n<table><tr><td>~~old~~</td></tr></table>")
    refine_format(page, None, {"cell_strike": True})
    intro, table = page.regions
    assert len(intro.candidates) == 1 and intro.selected == intro.candidates[0].id
    assert len(table.candidates) == 2 and table.chosen().reader == "fix.formatting"
    assert table.candidates[0].raw == "<table><tr><td>~~old~~</td></tr></table>"


def test_a_struck_pdf_word_is_struck_in_the_reading():
    assert _fixed("# Title\n\nHello world", {"strike": True}, _rendered()) == "# Title\n\nHello ~~world~~"


def test_a_struck_run_is_one_span_and_stays_on_its_line():
    rendered = _rendered("0.8 w 72 703.5 m 131 703.5 l S")
    assert _fixed("Hello, world!", {"strike": True}, rendered) == "~~Hello, world~~!"
    assert _fixed("Hello\nworld", {"strike": True}, rendered) == "~~Hello~~\n~~world~~"


def test_struck_words_in_a_table_cell_get_tags():
    md = "<table><tr><td>Hello world</td></tr></table>"
    assert _fixed(md, {"strike": True}, _rendered()) == "<table><tr><td>Hello <s>world</s></td></tr></table>"


def test_existing_or_unpaired_strikes_are_left_alone():
    rendered = _rendered()
    assert _fixed("Hello ~~world~~", {"strike": True}, rendered) == "Hello ~~world~~"
    assert _fixed("~~Hello world", {"strike": True}, rendered) == "~~Hello world"


def test_nothing_changes_without_marks_or_options():
    md = "Hello world"
    assert _fixed(md, {"strike": True}, _rendered(draw="")) == md
    assert _fixed(md, {}, _rendered()) == md
    assert _fixed(md, {"strike": True}, None) == md


def test_a_list_marker_wrapped_in_emphasis_becomes_list_syntax():
    md = "**Fact Check**\n**1.** a\n**2.** b\n\n__a)__ first\n\nSee **1.** above"
    assert _fixed(md, {"list_marker": True}) == "**Fact Check**\n1. a\n2. b\n\na) first\n\nSee **1.** above"


def _styled(texts):
    return render_pages(make_pdf([{"texts": texts}]), styles=True)[0]


def test_words_know_their_bold_share():
    rp = _styled([(72, 700, 12, "plain"), (72, 680, 12, "heavy", "FB")])
    assert {w.text: w.bold for w in rp.text_words} == {"plain": 0.0, "heavy": 1.0}


def test_bold_the_pdf_prints_in_a_regular_face_is_removed():
    rp = _styled([(72, 700, 12, "Hello world"), (72, 680, 12, "Note", "FB")])
    assert _fixed("**Hello world**\n\n**Note**", {"unbold": True}, rp) == "Hello world\n\n**Note**"
    assert _fixed("**Hello** world", {"unbold": True}, rp) == "Hello world"


def test_bold_stays_when_the_page_has_no_bold_face_or_in_headings():
    rp = _styled([(72, 700, 12, "Hello world")])  # no bold face anywhere: the fonts say nothing about bold
    assert _fixed("**Hello world**", {"unbold": True}, rp) == "**Hello world**"
    rp = _styled([(72, 700, 12, "Hello world"), (72, 680, 12, "Note", "FB")])
    assert _fixed("# **Hello world**", {"unbold": True}, rp) == "# **Hello world**"


def test_the_pipeline_applies_the_options_after_the_page_read():
    from docparse.models.adapters import ocria
    from docparse.models.calls import ModelResult
    from docparse.pipeline import parse
    from docparse.write.markdown import write_document

    class Client:
        def get(self, call, blobs, meta=None):
            return ModelResult(key=call.key(), output={"markdown": "Hello world"})

    pdf = make_pdf([{"texts": [(72, 700, 12, "Hello world")], "draw": STRUCK_WORLD}])
    config = Config(page_model=ocria.PUBLISHED, page_runtime="published", prompt_sha="published")
    assert write_document(parse(pdf, config, Client()).document) == "Hello world"
    config = config.model_copy(update={"format_options": {"strike": True}})
    assert write_document(parse(pdf, config, Client()).document) == "Hello ~~world~~"


def test_format_options_are_validated():
    cfg = Config.load("configs/standin.yaml", ["format_options.strike=true", "format_options.cell_strike=true"])
    assert cfg.format_options == {"strike": True, "cell_strike": True}
    with pytest.raises(ValueError, match="unknown format option"):
        Config.load("configs/standin.yaml", ["format_options.blink=true"])

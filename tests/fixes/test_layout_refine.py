import pytest

from docparse.core.config import Config
from docparse.core.ir import Box, LayoutItem
from docparse.fixes.layout_refine import assign_words, lines_of, refine, words_in_points
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf

P1 = ["First paragraph line one of the report", "first paragraph line two continues here"]
P2 = ["Second paragraph starts after a clear gap", "second paragraph line two ends the block"]


def _rendered():
    texts = [(72, 700, 11, P1[0]), (72, 686, 11, P1[1]), (72, 640, 11, P2[0]), (72, 626, 11, P2[1])]
    return render_pages(make_pdf([{"texts": texts}]))[0]


def _item(type_="Text", box=(60, 80, 420, 175), md="model text"):
    return LayoutItem(type=type_, label="text", md=md, bbox=Box(x0=box[0], y0=box[1], x1=box[2], y1=box[3]))


def test_no_options_or_no_text_layer_changes_nothing():
    rendered = _rendered()
    items = [_item()]
    assert refine(items, rendered, {}) is items
    rendered.words = None
    assert refine(items, rendered, {"tighten": True, "native_text": True, "split_gap": 1.0}) is items


def test_tighten_shrinks_a_padded_box_to_its_words():
    rendered = _rendered()
    (item,) = refine([_item()], rendered, {"tighten": True})
    words = words_in_points(rendered)
    assert item.bbox.x0 == pytest.approx(min(w.x0 for w in words)) and item.bbox.x0 > 60
    assert item.bbox.y1 == pytest.approx(max(w.y1 for w in words)) and item.bbox.y1 < 175
    assert item.md == "model text"  # tightening alone keeps the model's text


def test_pictures_tables_and_wordless_items_are_untouched():
    rendered = _rendered()
    picture, table, empty = _item("Picture"), _item("Table"), _item(box=(450, 400, 560, 500))
    out = refine([picture, table, empty], rendered, {"tighten": True, "native_text": True, "split_gap": 1.0})
    assert out == [picture, table, empty]


def test_native_text_is_the_pdf_words_in_reading_order():
    (item,) = refine([_item()], _rendered(), {"native_text": True})
    assert item.md == "\n".join(P1 + P2)
    assert item.bbox == _item().bbox


def test_split_at_a_paragraph_gap():
    rendered = _rendered()
    first, second = refine([_item()], rendered, {"split_gap": 1.0})
    assert first.md == "\n".join(P1) and second.md == "\n".join(P2)
    assert first.bbox.y1 < second.bbox.y0 and first.type == second.type == "Text"
    (single,) = refine([_item(box=(60, 80, 420, 112))], rendered, {"split_gap": 1.0})  # one paragraph only
    assert single.md == "model text"  # an item that is not split keeps its text
    (heading,) = refine([_item("Section-header")], rendered, {"split_gap": 1.0})  # headings are never split
    assert heading.type == "Section-header"
    assert len(refine([_item()], rendered, {"split_gap": 6.0})) == 1  # the gap is under the limit


def test_a_word_goes_to_the_smallest_item_holding_it():
    rendered = _rendered()
    big, small = _item(), _item(box=(60, 80, 420, 112))
    owned = assign_words([big, small], words_in_points(rendered))
    assert len(owned[1]) == len(" ".join(P1).split()) and len(owned[0]) == len(" ".join(P2).split())
    assert [len(line) for line in lines_of(owned[1])] == [len(P1[0].split()), len(P1[1].split())]


def test_config_overrides_and_unknown_option(tmp_path):
    cfg = Config.load(
        "configs/standin.yaml", ["layout_options.tighten=true", "layout_options.split_gap=1.5", "layout=none"]
    )
    assert cfg.layout_options == {"tighten": True, "split_gap": 1.5} and cfg.layout == "none"
    assert cfg.config_hash() != Config.load("configs/standin.yaml").config_hash()
    with pytest.raises(ValueError, match="unknown layout option"):
        Config.load("configs/standin.yaml", ["layout_options.padding=2"])
    with pytest.raises(ValueError, match="key=value"):
        Config.load("configs/standin.yaml", ["nonsense"])

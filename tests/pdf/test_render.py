import io

import pytest
from PIL import Image

from docparse.core.errors import ParseError
from docparse.pdf.render import MAX_PIXELS, pixel_hash, png_bytes, render_pages
from tests.pdfgen import make_pdf, simple_pdf


def _close(size, expected) -> bool:
    """PDFium rounds the pixel size up, so a render can be one pixel larger than points * scale."""
    return all(0 <= a - b <= 1.001 for a, b in zip(size, expected, strict=True))


def test_two_pages_at_300_dpi():
    pages = render_pages(
        make_pdf([{"texts": [(72, 700, 12, "One")]}, {"size": (300, 400), "texts": [(20, 300, 10, "Two")]}])
    )
    assert len(pages) == 2
    assert (pages[0].width_pt, pages[0].height_pt, pages[0].dpi) == (612, 792, 300)
    assert _close(pages[0].image.size, (2550, 3300)) and pages[0].image.mode == "RGB"
    assert _close(pages[1].image.size, (1250, 1667))
    assert [w.text for w in pages[0].words] == ["One"] and [w.text for w in pages[1].words] == ["Two"]


def test_word_boxes_are_in_image_pixels():
    (page,) = render_pages(simple_pdf("Hello world"))
    hello, world = page.words
    assert (hello.text, world.text) == ("Hello", "world")
    # text origin (72, 700) pt in a 792 pt page -> x = 300 px, baseline y = (792 - 700) * 300 / 72 = 383 px
    assert 295 <= hello.box[0] <= 310 and hello.box[1] < 383 <= hello.box[3] + 15
    assert hello.box[2] < world.box[0]


@pytest.mark.parametrize("rot", [90, 180, 270])
def test_rotated_page_is_rendered_as_displayed_and_words_stay_inside(rot):
    (page,) = render_pages(simple_pdf("Rotated text", rotate=rot))
    assert page.rotation == rot
    expected = (792, 612) if rot in (90, 270) else (612, 792)
    assert (page.width_pt, page.height_pt) == expected
    assert _close(page.image.size, (expected[0] * 300 / 72, expected[1] * 300 / 72))
    assert page.words
    for w in page.words:
        assert 0 <= w.box[0] < w.box[2] <= page.image.width and 0 <= w.box[1] < w.box[3] <= page.image.height


def test_cropbox_smaller_than_mediabox():
    (page,) = render_pages(simple_pdf("Cropped", crop=(36, 36, 576, 756)))
    assert (page.width_pt, page.height_pt) == (540, 720)
    (word,) = page.words
    # the origin moves with the crop: x = (72 - 36) * 300 / 72 = 150 px
    assert 145 <= word.box[0] <= 160
    assert word.box[3] <= page.image.height


def test_large_format_sheet_renders_below_300_dpi():
    (page,) = render_pages(simple_pdf("Big sheet", size=(5000, 5000)))
    assert page.dpi < 300
    assert page.image.width * page.image.height <= MAX_PIXELS * 1.001
    assert page.scale == pytest.approx(72 / page.dpi)
    (w0, w1) = page.words
    assert w0.box[2] * page.scale < 5000  # words convert to points with the page's own scale
    assert w0.box[0] * page.scale == pytest.approx(72, abs=2)


def test_pixel_hash_is_stable_and_content_sensitive():
    a1 = render_pages(simple_pdf("Same"))[0].image
    a2 = render_pages(simple_pdf("Same"))[0].image
    b = render_pages(simple_pdf("Different"))[0].image
    assert pixel_hash(a1) == pixel_hash(a2) != pixel_hash(b)
    assert len(pixel_hash(a1)) == 64
    assert Image.open(io.BytesIO(png_bytes(a1))).size == a1.size


def test_image_input_is_one_page_without_words():
    buf = io.BytesIO()
    Image.new("L", (600, 300), 255).save(buf, format="PNG")
    (page,) = render_pages(buf.getvalue())
    assert page.words is None and page.image.mode == "RGB"
    assert (page.width_pt, page.height_pt) == (144, 72)


@pytest.mark.parametrize("data", [b"", b"garbage bytes", b"%PDF-1.4\nnot really a pdf", b"\x89PNG\r\n\x1a\ntruncated"])
def test_unreadable_bytes_raise_parse_error(data):
    with pytest.raises(ParseError):
        render_pages(data)


def test_page_limit():
    with pytest.raises(ParseError, match="limit"):
        render_pages(make_pdf([{} for _ in range(3)]), max_pages=2)

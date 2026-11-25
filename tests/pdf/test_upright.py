from PIL import Image

from docparse.core.ir import Box
from docparse.pdf.render import RenderedPage, Word
from docparse.pdf.upright import turn_page, unturn_box


def _page():
    image = Image.new("RGB", (200, 100), "white")
    image.putpixel((10, 20), (0, 0, 0))
    word = Word(text="x", box=[10.0, 20.0, 30.0, 26.0])
    return RenderedPage(image=image, dpi=72, width_pt=200, height_pt=100, words=[word], text_words=[word])


def test_turning_moves_image_words_and_size_together():
    page = _page()
    quarter = turn_page(page, 90)  # clockwise: the left edge becomes the top edge
    assert quarter.image.size == (100, 200) and (quarter.width_pt, quarter.height_pt) == (100, 200)
    assert quarter.image.getpixel((100 - 1 - 20, 10)) == (0, 0, 0)
    assert quarter.words[0].box == [100 - 26.0, 10.0, 100 - 20.0, 30.0] == quarter.text_words[0].box
    half = turn_page(page, 180)
    assert half.image.getpixel((200 - 1 - 10, 100 - 1 - 20)) == (0, 0, 0)
    assert half.words[0].box == [170.0, 74.0, 190.0, 80.0]
    assert turn_page(page, 0) is page and page.words[0].box == [10.0, 20.0, 30.0, 26.0]  # the original is untouched


def test_a_box_on_the_turned_page_comes_back_to_the_stored_page():
    page = _page()
    for angle in (90, 180, 270):
        turned = turn_page(page, angle)
        x0, y0, x1, y1 = turned.words[0].box
        back = unturn_box(Box(x0=x0, y0=y0, x1=x1, y1=y1), angle, turned.width_pt, turned.height_pt)
        assert (back.x0, back.y0, back.x1, back.y1) == (10.0, 20.0, 30.0, 26.0)

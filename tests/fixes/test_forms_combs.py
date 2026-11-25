from docparse.core.config import Config
from docparse.fixes.forms import repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def _page(texts, overlay):
    data = make_pdf([{"texts": texts, "overlay": overlay}])
    return render_pages(data)[0], overlay_fonts(data)


def test_bracketed_combs_and_empty_comb_divider_noise_are_repaired():
    texts = [
        (72, 700, 11, "Phone"),
        (72, 580, 11, "Date"),
    ]
    overlay = [
        *[(160 + 12 * i, 700, 11, ch) for i, ch in enumerate("15684536825")],
    ]
    rendered, fonts = _page(texts, overlay)
    md = "**Phone**: [1][5][6][8][4][5][3][6][8][2][5]\n\n**Date :** 1__1__1__1__1__1__1__1"
    out = repair(md, rendered, fonts, {"combs": True, "comb_noise": True})
    assert "**Phone**: 15684536825" in out
    assert "**Date :** [___]" in out


def test_mini_value_table_below_caption_is_flattened_onto_the_caption():
    texts = [
        (72, 700, 11, "Name"),
        (72, 640, 11, "Net profit"),
        (260, 620, 11, "3.83"),
        (290, 620, 11, "£"),
        (72, 580, 11, "Adjustment"),
        (260, 560, 11, "3.82"),
        (290, 560, 11, "£"),
    ]
    overlay = [
        *[(160 + 14 * i, 680, 11, ch) for i, ch in enumerate("HEITOR")],
        *[(270 + 14 * i, 680, 11, ch) for i, ch in enumerate("SANTOS")],
        (310, 620, 11, "78,595"),
    ]
    rendered, fonts = _page(texts, overlay)
    md = (
        "**Name:**\n"
        "<table><tr><td>H</td><td>E</td><td>I</td><td>T</td><td>O</td><td>R</td>"
        "<td></td><td>S</td><td>A</td><td>N</td><td>T</td><td>O</td><td>S</td></tr>"
        "<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td>"
        "<td></td><td></td><td></td><td></td><td></td><td></td></tr></table>\n\n"
        "Net profit\n"
        "<table><tr><th>3.83</th><th>£</th><th>78,595</th></tr></table>\n\n"
        "Adjustment\n"
        "<table><tr><th>3.82</th><th>£</th><th></th></tr></table>"
    )
    out = repair(md, rendered, fonts, {"combs": True, "mini_tables": True})
    assert "**Name:** HEITOR SANTOS" in out
    assert "3.83 Net profit: £ 78,595" in out
    assert "3.82 Adjustment: [___]" in out
    assert "<table>" not in out


def test_options_are_validated_and_off_by_default():
    cfg = Config.load("configs/standin.yaml", ["form_options.comb_noise=true", "form_options.mini_tables=true"])
    assert cfg.form_options == {"comb_noise": True, "mini_tables": True}
    rendered, fonts = _page([(72, 700, 11, "Phone")], [(160 + 12 * i, 700, 11, ch) for i, ch in enumerate("12345")])
    md = "**Phone**: [1][2][3][4][5]"
    assert repair(md, rendered, fonts, {"combs": True}) == md

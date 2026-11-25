from docparse.core.config import Config
from docparse.fixes.forms import repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def _page(texts, overlay, rects=None):
    spec = {"texts": texts, "overlay": overlay}
    if rects:
        spec["rects"] = rects
    data = make_pdf([spec])
    return data, render_pages(data)[0], overlay_fonts(data)


def test_table_clean_merges_id_col_trailing_code_and_strips_bilingual_tail():
    texts = [
        (72, 740, 11, "5.7"),
        (120, 740, 11, "Total mileage"),
        (72, 710, 11, "5.8"),
        (120, 710, 11, "Mileage relief"),
        (72, 660, 11, "Heat"),
        (300, 660, 11, "7A"),
        (72, 620, 11, "Emirate"),
    ]
    overlay = [(220, 740, 11, "100"), (220, 660, 11, "724.48"), (220, 620, 11, "Sharjah")]
    data, rendered, fonts = _page(texts, overlay)
    md = (
        "<table>"
        "<tr><td>5.7</td><td>Total mileage</td><td>100</td></tr>"
        "<tr><td>5.8</td><td>Mileage relief</td><td></td></tr>"
        "</table>\n\n"
        "<table><tr><td>Heat</td><td>724.48</td><td>7A</td></tr></table>\n\n"
        "<table><tr><td>Emirate</td><td>Sharjah<br/>الإمارة</td></tr></table>"
    )
    out = repair(md, rendered, fonts, {"table_clean": True}, pdf_bytes=data, page_index=0)
    assert '<td colspan="2">5.7 Total mileage</td>' in out
    assert '<td colspan="2">5.8 Mileage relief</td>' in out
    assert "7A Heat: 724.48" in out
    assert "<td>Sharjah</td>" in out


def test_table_clean_reads_horizontal_checkbox_strip_with_adjacent_ticks():
    # Drawn 12x12 boxes directly below RA, RB, RC, RD; adjacent ticks in RB and RC!
    texts = [(100, 720, 10, "RA"), (120, 720, 10, "RB"), (140, 720, 10, "RC"), (160, 720, 10, "RD")]
    rects = [(98, 700, 12, 12), (118, 700, 12, 12), (138, 700, 12, 12), (158, 700, 12, 12)]
    overlay = [(121, 702, 10, "X"), (141, 702, 10, "X")]
    data, rendered, fonts = _page(texts, overlay, rects=rects)
    md = (
        "<table>"
        "<tr><th>RA</th><th>RB</th><th>RC</th><th>RD</th></tr>"
        "<tr><td></td><td><strong>X</strong></td><td><strong>X</strong></td><td></td></tr>"
        "</table>"
    )
    out = repair(md, rendered, fonts, {"table_clean": True}, pdf_bytes=data, page_index=0)
    assert "<td>[ ] RA</td><td>[x] RB</td><td>[x] RC</td><td>[ ] RD</td>" in out


def test_wave2_option_is_validated():
    cfg = Config.load("configs/standin.yaml", ["form_options.table_clean=true"])
    assert cfg.form_options == {"table_clean": True}

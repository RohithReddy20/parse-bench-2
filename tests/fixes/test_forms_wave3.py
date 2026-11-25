from docparse.core.config import Config
from docparse.fixes.forms import repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def _page(texts, overlay):
    data = make_pdf([{"texts": texts, "overlay": overlay}])
    return data, render_pages(data)[0], overlay_fonts(data)


def test_field_notes_strips_table_examples_merges_paren_and_trailing_code_rows_and_moves_blank_code():
    texts = [
        (72, 740, 11, "Namenszusatz (Beispiel: Freifrau, Graf)"),
        (72, 710, 11, "22. Chassis No. (Affix Pencil print)"),
        (72, 680, 11, "Enter the amount from line 33"),
        (420, 680, 11, "34"),
        (72, 650, 11, "Recettes brutes sans abattement 4BE"),
    ]
    overlay = [(300, 710, 11, "KYJ08XL8YRBUH451X"), (320, 680, 11, "40,346.80")]
    data, rendered, fonts = _page(texts, overlay)
    md = (
        "<table>"
        "<tr><td><strong>Namenszusatz (Beispiel: Freifrau, Graf)</strong></td><td></td></tr>"
        "</table>\n\n"
        "<table>"
        "<tr><td>22. Chassis No. (Affix Pencil print)</td><td>KYJ08XL8YRBUH451X</td></tr>"
        "</table>\n\n"
        "<table>"
        "<tr><td>Enter the amount from line 33</td><td></td><td></td><td>40,346.80</td><td><strong>34</strong></td></tr>"
        "</table>\n\n"
        "Recettes brutes sans abattement *n’excédant pas 15 000 €* 4BE: [___]"
    )
    out = repair(md, rendered, fonts, {"field_notes": True}, pdf_bytes=data, page_index=0)
    assert "<td><strong>Namenszusatz</strong></td>" in out
    assert "22. Chassis No. (Affix Pencil print): KYJ08XL8YRBUH451X" in out
    assert "34 Enter the amount from line 33: 40,346.80" in out
    assert "4BE Recettes brutes sans abattement: [___]" in out


def test_field_notes_option_is_validated():
    assert Config.load("configs/standin.yaml", ["form_options.field_notes=true"]).form_options == {"field_notes": True}

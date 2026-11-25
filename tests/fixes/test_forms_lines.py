from docparse.core.config import Config
from docparse.fixes.forms import repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def _page(texts, overlay):
    data = make_pdf([{"texts": texts, "overlay": overlay}])
    return render_pages(data)[0], overlay_fonts(data)


def test_field_lines_moves_printed_code_cleans_blank_hints_and_splits_furniture():
    texts = [
        (72, 740, 11, "einfache Entfernung:"),
        (220, 740, 11, "111"),
        (290, 740, 11, "km"),
        (72, 710, 11, "Arbeitslohn:"),
        (220, 710, 11, "127"),
        (290, 710, 11, "EUR"),
        (72, 680, 11, "Telephone Number (999) 999-9999:"),
        (72, 650, 11, "Namenszusatz (Beispiel: Freifrau, Graf):"),
        (72, 620, 11, "Dividendes:"),
        (220, 620, 11, "(2TS)"),
        (72, 590, 11, "Value:"),
        (220, 590, 11, "AUD"),
        (72, 560, 11, "Total people:"),
        (240, 560, 11, "Give details below"),
        (72, 530, 11, "Date of birth:"),
        (240, 530, 11, "(Day Month Year)"),
        (72, 500, 11, "Income:"),
        (72, 470, 11, "Tax:"),
    ]
    overlay = [(250, 740, 11, "25"), (190, 560, 11, "3"), (170, 530, 11, "06 12 1930"), (220, 470, 11, "255.00")]
    rendered, fonts = _page(texts, overlay)
    md = (
        "einfache Entfernung: 111 25 km\n"
        "Arbeitslohn: 127 [___] EUR\n"
        "Telephone Number (999) 999-9999: [___]\n"
        "Namenszusatz (Beispiel: Freifrau, Graf): [___]\n"
        "Dividendes: [___] (2TS)\n"
        "Value: AUD\n"
        "Total people: 3 ▶ Give details below\n"
        "Date of birth: 06 12 1930 (Day Month Year)\n"
        "Income: [___], [___], [___]. [___]（元） [x] None\n"
        "Tax: [___], [___], 255.00"
    )
    out = repair(md, rendered, fonts, {"field_lines": True})
    assert "111 einfache Entfernung: 25" in out
    assert "127 Arbeitslohn: [___]" in out
    assert "Telephone Number: [___]" in out
    assert "Namenszusatz: [___]" in out
    assert "Dividendes: [___]\n" in out
    assert "Value: [___]\n" in out
    assert "Total people: 3\n ▶ Give details below" in out
    assert "Date of birth: 06 12 1930 (Day Month Year)" in out
    assert "Income: [___] [x] None" in out
    assert "Tax: 255.00" in out


def test_field_lines_option_is_validated():
    assert Config.load("configs/standin.yaml", ["form_options.field_lines=true"]).form_options == {"field_lines": True}

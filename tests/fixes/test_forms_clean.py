from docparse.core.config import Config
from docparse.fixes.forms import repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def _page(texts, overlay):
    data = make_pdf([{"texts": texts, "overlay": overlay}])
    return data, render_pages(data)[0], overlay_fonts(data)


def test_field_clean_repairs_divider_combs_eq_footnotes_and_date_units():
    texts = [
        (72, 750, 11, "Code postal :"),
        (72, 725, 11, "Date de naissance :"),
        (72, 695, 11, "Formula 12 25"),
        (72, 635, 11, "vom Tag Monat Jahr bis Tag Monat Jahr"),
    ]
    overlay = [
        *[(200 + 12 * i, 750, 11, ch) for i, ch in enumerate("51")],
        *[(235 + 12 * i, 750, 11, ch) for i, ch in enumerate("100")],
        *[(200 + 12 * i, 725, 11, ch) for i, ch in enumerate("1510")],
        *[(260 + 12 * i, 725, 11, ch) for i, ch in enumerate("1997")],
        (280, 695, 11, "45,369.46"),
    ]
    data, rendered, fonts = _page(texts, overlay)
    md = (
        "**Code postal :** 1<u>5</u>1<u>1</u>1<u>0</u>1<u>0</u>\n\n"
        "**Date de naissance :** 1 | 5 | 1 | 0 | 1 | 9 | 9 | 7\n\n"
        "<table><tr><td>Formula <u>12</u></td><td></td><td>= 45,369.46 25</td></tr></table>\n\n"
        "**vom** **Tag** **Monat** **Jahr** **bis** **Tag** **Monat** **Jahr**"
    )
    out = repair(md, rendered, fonts, {"field_clean": True}, pdf_bytes=data, page_index=0)
    assert "**Code postal :** 51100" in out
    assert "**Date de naissance :** 15101997" in out
    assert "25 Formula: 45,369.46" in out
    assert "**vom** [___]\n**Tag** **Monat** **Jahr** **bis** [___]\n**Tag** **Monat** **Jahr**" in out


def test_field_clean_option_is_validated():
    assert Config.load("configs/standin.yaml", ["form_options.field_clean=true"]).form_options == {"field_clean": True}
    assert Config.load("configs/standin.yaml", ["form_options.blank_hints=true"]).form_options == {"blank_hints": True}


def test_blank_hints_strips_printed_hints_on_blank_fields():
    texts = [
        (72, 750, 11, "16.1 Trading income allowance - read the notes"),
        (72, 730, 11, "5b Spouse prior name. See instructions."),
        (72, 700, 11, "Fax number (999) 999-9999:"),
        (72, 670, 11, "924 Profits (imprime 2561)"),
        (72, 640, 11, "VL3 IMPOSTA DOVUTA ovvero ,00"),
    ]
    overlay = [(300, 610, 11, "FILLED")]
    data, rendered, fonts = _page(texts, overlay)
    md = (
        "**16.1 Trading income allowance** – read the notes\n£ 0 0\n\n"
        "**5b Spouse prior name.** See instructions.\n[___]\n\n"
        "**Fax number (999) 999-9999:** [___]\n\n"
        "**924 Profits** (*imprime 2561*) [___]\n\n"
        "<table><tr><td>VL3 <strong>IMPOSTA DOVUTA</strong><br/>ovvero</td><td>,00</td></tr></table>\n\n"
        "**Other:** FILLED"
    )
    out = repair(md, rendered, fonts, {"blank_hints": True}, pdf_bytes=data, page_index=0)
    assert "**16.1 Trading income allowance**\n£ 0 0" in out
    assert "**5b Spouse prior name.**\n[___]" in out
    assert "**Fax number:** [___]" in out
    assert "**924 Profits** [___]" in out
    assert "ovvero" not in out
    assert "<td>[___]</td>" in out

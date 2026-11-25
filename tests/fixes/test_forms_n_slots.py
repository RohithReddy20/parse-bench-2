from docparse.core.config import Config
from docparse.fixes.forms import repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def _page(texts, overlay):
    data = make_pdf([{"texts": texts, "overlay": overlay}])
    return data, render_pages(data)[0], overlay_fonts(data)


def test_field_slots_handles_cap_over_val_legend_phone_and_date_hint():
    texts = [
        (72, 740, 11, "Full name"),
        (220, 740, 11, "Date of expiry"),
        (220, 725, 11, "Month"),
        (300, 725, 11, "Year"),
        (72, 680, 11, "Religion change: 1 = Leave, 2 = Switch, 3 = Join"),
        (72, 640, 11, "Country code Area code Number"),
        (72, 600, 11, "Date of birth YYYY-MM-DD"),
    ]
    overlay = [
        (72, 725, 11, "JANELLE WEBER"),
        (260, 725, 11, "08"),
        (340, 725, 11, "2027"),
        (72, 620, 11, "(61) (02) 9876 5432"),
        (250, 600, 11, "1985-04-12"),
    ]
    data, rendered, fonts = _page(texts, overlay)
    md = (
        "<table>"
        '<tr><th>Full name</th><th colspan="2">Date of expiry</th></tr>'
        "<tr><td>JANELLE WEBER</td><td>Month: 08</td><td>Year: 2027</td></tr>"
        "</table>\n\n"
        "Religion change: 1 = Leave, 2 = Switch, 3 = Join\n\n"
        "Country code: [___] Area code: [___] Number: [___]\n"
        "( 61 ) ( 02 ) 9876 5432\n\n"
        "Date of birth: YYYY-MM-DD 1985-04-12"
    )
    out = repair(md, rendered, fonts, {"field_slots": True}, pdf_bytes=data, page_index=0)
    assert "<td>Full name: JANELLE WEBER</td>" in out
    assert "<td>Date of expiry Month: 08</td>" in out
    assert "<td>Date of expiry Year: 2027</td>" in out
    assert "Religion change: [___]\n1 = Leave, 2 = Switch, 3 = Join" in out
    assert "Country code: 61\nArea code: 02\nNumber: 9876 5432" in out
    assert "Date of birth: 1985-04-12" in out
    assert "YYYY-MM-DD" not in out


def test_field_slots_skips_signature_tables_and_paren_date_hints():
    texts = [
        (72, 740, 11, "Signature of applicant"),
        (250, 740, 11, "Date"),
        (72, 680, 11, "Date of birth (YYYY-MM-DD)"),
    ]
    overlay = [(72, 725, 11, "Jane Doe"), (250, 725, 11, "2025-01-01"), (260, 680, 11, "1990-02-03")]
    data, rendered, fonts = _page(texts, overlay)
    md = (
        "<table>"
        "<tr><th>Signature of applicant</th><th>Date</th></tr>"
        "<tr><td>Jane Doe</td><td>2025-01-01</td></tr>"
        "</table>\n\n"
        "Date of birth (YYYY-MM-DD): 1990-02-03"
    )
    out = repair(md, rendered, fonts, {"field_slots": True}, pdf_bytes=data, page_index=0)
    assert out == md


def test_field_slots_option_is_validated():
    assert Config.load("configs/standin.yaml", ["form_options.field_slots=true"]).form_options == {"field_slots": True}

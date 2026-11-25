from docparse.core.config import Config
from docparse.fixes.forms import repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def _page(texts, overlay):
    data = make_pdf([{"texts": texts, "overlay": overlay}])
    return render_pages(data)[0], overlay_fonts(data)


def test_kv_table_rows_move_printed_codes_to_caption_and_keep_overlay_or_blank():
    texts = [
        (72, 700, 11, "Line 119 plus line 120"),
        (300, 700, 11, "40400"),
        (350, 700, 11, "="),
        (500, 700, 11, "121"),
        (72, 660, 11, "Federal tax on split income"),
        (300, 660, 11, "40424"),
        (350, 660, 11, "+"),
        (500, 660, 11, "120"),
    ]
    overlay = [(400, 700, 11, "5,452.47")]
    rendered, fonts = _page(texts, overlay)
    md = (
        "<table>"
        "<tr><td>Line 119 plus line 120</td><td>40400</td><td><strong>=</strong></td>"
        "<td>5,452.47</td><td></td><td><strong>121</strong></td></tr>"
        "<tr><td>Federal tax on split income</td><td><strong>40424</strong></td><td><strong>+</strong></td>"
        "<td></td><td></td><td><strong>• 120</strong></td></tr>"
        "</table>"
    )
    out = repair(md, rendered, fonts, {"table_rows": True})
    assert "40400 121 Line 119 plus line 120: 5,452.47" in out
    assert "40424 120 Federal tax on split income: [___]" in out


def test_single_value_column_table_with_header_merges_reference_code():
    texts = [
        (72, 740, 11, "Description"),
        (320, 740, 11, "Reference"),
        (420, 740, 11, "Value"),
        (72, 700, 11, "Current value of your family home"),
        (320, 700, 11, "A"),
        (72, 660, 11, "Current value of other property"),
        (320, 660, 11, "B"),
    ]
    overlay = [(420, 700, 11, "130,073.93"), (420, 660, 11, "121,509.06")]
    rendered, fonts = _page(texts, overlay)
    md = (
        "<table>"
        "<tr><th>Description</th><th>Reference</th><th>Value</th></tr>"
        "<tr><td>Current value of your family home</td><td>A</td><td>130,073.93</td></tr>"
        "<tr><td>Current value of other property</td><td>B</td><td>121,509.06</td></tr>"
        "</table>"
    )
    out = repair(md, rendered, fonts, {"table_rows": True})
    assert "A Current value of your family home: 130,073.93" in out
    assert "B Current value of other property: 121,509.06" in out


def test_multi_value_column_table_is_left_alone():
    texts = [
        (250, 740, 11, "Capital allowances"),
        (420, 740, 11, "Balancing charges"),
        (72, 700, 11, "Vehicle allowance"),
        (250, 700, 11, "3.14"),
        (300, 700, 11, "£"),
        (420, 700, 11, "3.15"),
        (460, 700, 11, "£"),
    ]
    overlay = [(320, 700, 11, "500"), (480, 700, 11, "100")]
    rendered, fonts = _page(texts, overlay)
    md = (
        '<table><tr><th></th><th colspan="2">Capital allowances</th><th colspan="2">Balancing charges</th></tr>'
        "<tr><td>Vehicle allowance</td><td>3.14</td><td>£ 500</td><td>3.15</td><td>£ 100</td></tr></table>"
    )
    assert repair(md, rendered, fonts, {"table_rows": True}) == md


def test_two_row_caption_over_value_grid_pairs_each_caption_with_cell_below():
    texts = [
        (72, 700, 11, "Cardholder name"),
        (240, 700, 11, "Card number"),
        (420, 700, 11, "Date of expiry"),
    ]
    overlay = [(72, 680, 11, "JANELLE WEBER"), (240, 680, 11, "6957 4266")]
    rendered, fonts = _page(texts, overlay)
    md = (
        "<table>"
        "<tr><td>Cardholder name</td><td>Card number</td><td>Date of expiry</td></tr>"
        "<tr><td>JANELLE WEBER</td><td>6957 4266</td><td></td></tr>"
        "</table>"
    )
    out = repair(md, rendered, fonts, {"table_rows": True})
    assert "Cardholder name: JANELLE WEBER" in out
    assert "Card number: 6957 4266" in out
    assert "Date of expiry: [___]" in out


def test_blank_printed_code_cents_and_repeated_empty_template_cells_are_cleaned():
    texts = [
        (72, 740, 11, "RP1"),
        (120, 740, 11, "Spese sanitarie"),
        (280, 740, 11, "1"),
        (310, 740, 11, ",00"),
        (380, 740, 11, "2"),
        (430, 740, 11, ",00"),
        (72, 700, 11, "Importo rata"),
        (180, 700, 11, "3"),
        (200, 700, 11, ",00"),
        (72, 660, 11, "NIK/NPWP:"),
        (72, 645, 11, "NAMA:"),
        (72, 620, 11, "NIK/NPWP:"),
        (72, 605, 11, "NAMA:"),
        (72, 580, 11, "NIK/NPWP:"),
        (72, 565, 11, "NAMA:"),
    ]
    overlay = [(400, 740, 11, "2581")]
    rendered, fonts = _page(texts, overlay)
    md = (
        "<table>"
        "<tr><td>RP1</td><td>Spese sanitarie</td><td>1 ,00</td><td>2 2581 ,00</td></tr>"
        '<tr><td>RP2</td><td colspan="3">Importo rata 3 ,00</td></tr>'
        "<tr><td>NIK/NPWP:<br/>NAMA:</td><td></td><td></td><td></td></tr>"
        "<tr><td>NIK/NPWP:<br/>NAMA:</td><td></td><td></td><td></td></tr>"
        "<tr><td>NIK/NPWP:<br/>NAMA:</td><td></td><td></td><td></td></tr>"
        "</table>"
    )
    out = repair(md, rendered, fonts, {"table_rows": True})
    assert "<td>1 ,00</td>" not in out
    assert "<td>2581</td>" in out
    assert "3 Importo rata: [___]" in out
    assert "NIK/NPWP:<br/>NAMA:" not in out


def test_table_rows_option_is_validated():
    assert Config.load("configs/standin.yaml", ["form_options.table_rows=true"]).form_options == {"table_rows": True}

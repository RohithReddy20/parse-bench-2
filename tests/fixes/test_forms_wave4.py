from docparse.core.config import Config
from docparse.fixes.forms import repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def _page(texts, overlay):
    data = make_pdf([{"texts": texts, "overlay": overlay}])
    return data, render_pages(data)[0], overlay_fonts(data)


def test_table_grids_repairs_box_num_cells_col_num_headers_examples_and_dmy_watermarks():
    texts = [
        (72, 740, 11, "Cost of sales 3.30 3.46 Other costs 3.31 Disallowable Total"),
        (72, 710, 11, "NAME DOB JOB (1) (2) (3)"),
        (72, 670, 11, "2.2 Job title for example - Carer"),
        (72, 630, 11, "Date d d m m y y y y"),
    ]
    overlay = [
        (240, 740, 11, "164"),
        (320, 740, 11, "86,576"),
        (72, 695, 11, "ALICE"),
        (160, 695, 11, "22-04-2020"),
        (260, 695, 11, "STUDENT"),
        (240, 670, 11, "Welder"),
        (140, 630, 11, "27"),
        (170, 630, 11, "09"),
        (200, 630, 11, "2026"),
    ]
    data, rendered, fonts = _page(texts, overlay)
    md = (
        "<table>"
        "<tr><td></td><td><strong>Disallowable</strong></td><td><strong>Total</strong></td></tr>"
        "<tr><td>Cost of sales<br/>See notes</td><td><strong>3.30</strong> £ 164</td><td><strong>3.46</strong> £ 86,576</td></tr>"
        "<tr><td>Other costs</td><td><strong>3.31</strong> £ [ ]</td><td></td></tr>"
        "</table>\n\n"
        "<table>"
        "<tr><th>NAME</th><th>DOB</th><th>JOB</th></tr>"
        "<tr><th>(1)</th><th>(2)</th><th>(3)</th></tr>"
        "<tr><td>ALICE</td><td>22-04-2020</td><td>STUDENT</td></tr>"
        "</table>\n\n"
        "<table>"
        "<tr><td>2.2 <strong>Job title</strong><br/>for example - Carer</td><td>Welder</td></tr>"
        "</table>\n\n"
        "**Date**: d 2 d 7 m 0 m 9 y 2 y 0 y 2 y 6"
    )
    out = repair(md, rendered, fonts, {"table_grids": True}, pdf_bytes=data, page_index=0)
    assert "3.30 Cost of sales — Disallowable: 164" in out
    assert "3.46 Cost of sales — Total: 86,576" in out
    assert "3.31 Other costs — Disallowable: [___]" in out
    assert "<th>(1) NAME</th><th>(2) DOB</th><th>(3) JOB</th>" in out
    assert "<tr><th>(1)</th>" not in out
    assert "for example" not in out
    assert "**Date**: 27 09 2026" in out


def test_table_grids_leaves_code_only_cells_untouched():
    texts = [(72, 740, 11, "Equipment 3.16 3.17")]
    overlay = [(240, 740, 11, "977")]
    data, rendered, fonts = _page(texts, overlay)
    md = "<table><tr><td>Equipment</td><td><strong>3.16</strong></td><td>£ 977</td><td><strong>3.17</strong></td><td>£</td></tr></table>"
    out = repair(md, rendered, fonts, {"table_grids": True}, pdf_bytes=data, page_index=0)
    assert out == md


def test_table_grids_option_is_validated():
    assert Config.load("configs/standin.yaml", ["form_options.table_grids=true"]).form_options == {"table_grids": True}

from docparse.core.config import Config
from docparse.fixes.forms import repair
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def test_n_blank_forms_repairs_box_num_code_slots_inline_splits_and_units():
    data = make_pdf([{"texts": [(72, 740, 11, "Blank form")]}])
    rendered = render_pages(data)[0]
    md = (
        "<table>"
        "<tr><td><strong>Tax A</strong><br/>note</td><td><strong>141.</strong><br/>zł</td><td><strong>142.</strong><br/>,</td></tr>"
        "<tr><td><strong>Tax B</strong></td><td><strong>143.</strong><br/>,</td><td></td></tr>"
        "</table>\n\n"
        "<table>"
        "<tr><td><strong>Gross pay</strong></td><td><strong>110</strong></td><td>[___]</td><td>,—</td><td><strong>111</strong></td><td>[___]</td></tr>"
        "<tr><td><strong>Income tax</strong></td><td><strong>140</strong></td><td>[___]</td><td></td><td><strong>141</strong></td><td>[___]</td></tr>"
        "<tr><td><strong>Surcharge</strong></td><td><strong>150</strong></td><td>[___]</td><td></td><td><strong>151</strong></td><td>[___]</td></tr>"
        "</table>\n\n"
        "**First Name:** [___] **Middle Name:** [___] **Last Name:** [___]\n"
        "**Age:** [___] Years **City:** [___] **State:** [___] **ZIP:** [___]\n"
        "**Phone:** [___] **Email:** [___] **PIN:** [___] **District:** [___]"
    )
    out = repair(md, rendered, frozenset(), {"n_blank_forms": True}, pdf_bytes=data, page_index=0)
    assert "<td>141. Tax A: [___]</td>" in out
    assert "<td>142. Tax A: [___]</td>" in out
    assert "<td>143. Tax B: [___]</td>" in out
    assert "<td>110 Gross pay: [___]</td>" in out
    assert "<td>111 Gross pay: [___]</td>" in out
    assert "**First Name:** [___]\n**Middle Name:** [___]\n**Last Name:** [___]" in out
    assert "**Age:** [___]\n**City:** [___]" in out
    assert "Years" not in out


def test_n_blank_forms_skips_overlay_pages_and_pages_with_few_blanks():
    data = make_pdf([{"texts": [(72, 740, 11, "Name")], "overlay": [(140, 740, 11, "Alice")]}])
    rendered = render_pages(data)[0]
    md = "**First Name:** [___] **Middle Name:** [___]"
    # Overlay page -> untouched by n_blank_forms
    assert repair(md, rendered, frozenset({"F1"}), {"n_blank_forms": True}, pdf_bytes=data, page_index=0) == md
    # Non-overlay page with < 10 blanks -> untouched
    assert repair(md, rendered, frozenset(), {"n_blank_forms": True}, pdf_bytes=data, page_index=0) == md


def test_n_blank_forms_option_is_validated():
    assert Config.load("configs/standin.yaml", ["form_options.n_blank_forms=true"]).form_options == {
        "n_blank_forms": True
    }

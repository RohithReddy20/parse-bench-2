from docparse.core.config import Config
from docparse.fixes.forms import apply_edits
from docparse.fixes.forms_acro import acro_checkbox_edits, acro_id_edits, repair_acroform
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def test_acro_id_edits_merges_row_id_and_expands_sub_letters():
    md = (
        "<table>"
        "<tr><td><strong>1</strong></td><td>Wages, salaries, tips</td><td><strong>1</strong></td><td></td></tr>"
        "<tr><td><strong>2a</strong></td><td>Tax-exempt interest</td><td><strong>2a</strong></td>"
        "<td><strong>b</strong> Taxable interest</td><td><strong>2b</strong></td><td></td></tr>"
        '<tr><td><strong>8</strong></td><td>Other income:</td><td></td><td rowspan="3"></td></tr>'
        "<tr><td><strong>a</strong></td><td>Net operating loss</td><td><strong>8a</strong></td></tr>"
        "<tr><td><strong>b</strong></td><td>Gambling</td><td><strong>8b</strong></td></tr>"
        "</table>"
    )
    out = apply_edits(md, acro_id_edits(md))
    assert '<td colspan="2">1 Wages, salaries, tips</td>' in out
    assert '<td colspan="2">2a Tax-exempt interest</td>' in out
    assert "<td><strong>2b</strong> Taxable interest</td>" in out
    assert '<td colspan="2">8a Net operating loss</td>' in out
    assert '<td colspan="2">8b Gambling</td>' in out


def test_acro_checkbox_edits_inserts_boxes_beside_aligned_words():
    data = make_pdf([{"texts": [(100, 700, 11, "Attendance"), (260, 700, 11, "Demographics")]}])
    rendered = render_pages(data)[0]
    # Top-down coords of words are around y=80..95 (page height 792 -> 792-711 = 81)
    y0 = rendered.text_words[0].box[1] * rendered.scale
    y1 = rendered.text_words[0].box[3] * rendered.scale
    x0_a = rendered.text_words[0].box[0] * rendered.scale
    x0_b = rendered.text_words[1].box[0] * rendered.scale
    cboxes = [(x0_a - 16, y0, x0_a - 6, y1, False), (x0_b - 16, y0, x0_b - 6, y1, True)]
    md = "<table><tr><td>Attendance</td><td>Demographics</td></tr></table>"
    out = apply_edits(md, acro_checkbox_edits(md, rendered, cboxes))
    assert "<td>[ ] Attendance</td><td>[x] Demographics</td>" in out


def test_repair_acroform_leaves_non_acroform_pdf_unchanged():
    data = make_pdf([{"texts": [(72, 700, 11, "1 Wages")]}])
    rendered = render_pages(data)[0]
    md = "<table><tr><td>1</td><td>Wages</td><td></td></tr><tr><td>2</td><td>Tax</td><td></td></tr></table>"
    assert repair_acroform(md, rendered, data, 0) == md


def test_acroform_option_is_validated():
    assert Config.load("configs/standin.yaml", ["form_options.acroform=true"]).form_options == {"acroform": True}
    assert Config.load("configs/standin.yaml", ["form_options.acro_clean=true"]).form_options == {"acro_clean": True}


def test_acro_clean_strips_blank_row_indices_splits_notes_and_fills_yes_no_grids():
    from docparse.fixes.forms_acro import acro_clean_markdown

    md = (
        "<table><tr><th>Name</th><th>Phone</th></tr>"
        "<tr><td>1) [___]</td><td></td></tr>"
        "<tr><td>2. [___]</td><td></td></tr></table>\n\n"
        "<table><tr><th></th><th>Yes</th><th>No</th></tr>"
        "<tr><td>1. Question A</td><td></td><td></td></tr>"
        "<tr><td>2. Question B</td><td></td><td></td></tr></table>\n\n"
        "**46. Net tax**: [___] **This is the amount you owe. 46.** [___]\n"
        "**Authorized Signature**: [___] (Owner, Partner)\n"
        "ending , 20 [___] See separate instructions.\n"
        "<table><tr><td><strong>LAST 4:</strong><br/><strong>XXX-XX-</strong> [___]</td></tr></table>"
    )
    out = acro_clean_markdown(md, has_boxes=True)
    assert "<tr><td>[___]</td><td></td></tr>" in out
    assert "1) [___]" not in out
    assert "2. [___]" not in out
    assert "<td>[ ] Yes</td><td>[ ] No</td>" in out
    assert "**46. Net tax**: [___]\n**This is the amount you owe. 46.** [___]" in out
    assert "**Authorized Signature**: [___]\n(Owner, Partner)" in out
    assert "20 [___]\nSee separate instructions." in out
    assert "<strong>LAST 4:</strong><br/>[___]" in out

"""AcroForm v2: checkbox marks corrected to widget state, and missing text-field values written beside labels."""

from docparse.fixes.forms import apply_edits
from docparse.fixes.forms_acro import (
    acro_checkbox_edits,
    acro_text_value_edits,
    acroform_text_widgets,
    acroform_widgets,
    repair_acroform,
)
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def _pdf_string(s: str) -> str:
    if s.isascii():
        return "(" + s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"
    return "<FEFF" + s.encode("utf-16-be").hex().upper() + ">"


def _acro_pdf(texts: list[tuple[float, float, float, str]], fields: list[dict]) -> bytes:
    """One 612x792 page with Helvetica `texts` `(x, y, size, text)` and AcroForm widget `fields`:
    {"kind": "tx" | "cb", "rect": (x0, y0, x1, y1) in PDF user space, "value": str, "checked": bool,
    "F": annotation flags, "Ff": field flags}."""
    first_field = 8
    refs = " ".join(f"{first_field + i} 0 R" for i in range(len(fields)))
    content = "\n".join(f"BT /F1 {size} Tf {x} {y} Td {_pdf_string(text)} Tj ET" for x, y, size, text in texts).encode(
        "latin-1"
    )
    on_ap = b"0 0 m 10 10 l S"
    objects = [
        f"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [{refs}] >> >>".encode(),
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents 5 0 R /Annots [{refs}] >>"
        ).encode(),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /XObject /Subtype /Form /BBox [0 0 10 10] /Length 0 >>\nstream\n\nendstream",
        b"<< /Type /XObject /Subtype /Form /BBox [0 0 10 10] /Length %d >>\nstream\n" % len(on_ap)
        + on_ap
        + b"\nendstream",
    ]
    for i, f in enumerate(fields):
        rect = "[{} {} {} {}]".format(*f["rect"])
        common = (
            f"/Type /Annot /Subtype /Widget /T (f{i}) /Rect {rect} /F {f.get('F', 4)} /Ff {f.get('Ff', 0)} /P 4 0 R"
        )
        if f["kind"] == "cb":
            state = "/Yes" if f.get("checked") else "/Off"
            body = f"<< {common} /FT /Btn /V {state} /AS {state} /AP << /N << /Yes 7 0 R /Off 6 0 R >> >> >>"
        else:
            value = f" /V {_pdf_string(f['value'])}" if f.get("value") is not None else ""
            body = f"<< {common} /FT /Tx /DA (/F1 10 Tf 0 g){value} >>"
        objects.append(body.encode("latin-1"))
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def _word_boxes(rendered) -> dict[str, tuple[float, float, float, float]]:
    s = rendered.scale
    return {w.text: (w.box[0] * s, w.box[1] * s, w.box[2] * s, w.box[3] * s) for w in rendered.text_words}


def _box_left(box: tuple[float, float, float, float], checked: bool) -> tuple[float, float, float, float, bool]:
    x0, y0, _x1, y1 = box
    return (x0 - 16, y0, x0 - 6, y1, checked)


def _field_right(box: tuple[float, float, float, float], value: str) -> tuple[float, float, float, float, str]:
    _x0, y0, x1, y1 = box
    return (x1 + 8, y0 - 3, x1 + 180, y1 + 3, value)


# --- checkbox marks follow the authoritative widget state -------------------------------------------------


def test_checkbox_marks_flip_to_widget_state_on_separate_lines():
    rendered = render_pages(make_pdf([{"texts": [(100, 700, 11, "Married"), (100, 670, 11, "Single")]}]))[0]
    wb = _word_boxes(rendered)
    cboxes = [_box_left(wb["Married"], True), _box_left(wb["Single"], False)]
    md = "[ ] Married\n[x] Single\n"
    assert apply_edits(md, acro_checkbox_edits(md, rendered, cboxes)) == "[x] Married\n[ ] Single\n"


def test_checkbox_marks_flip_within_one_unit_and_keep_agreeing_marks():
    data = make_pdf([{"texts": [(100, 700, 11, "Married"), (220, 700, 11, "Single"), (340, 700, 11, "Widowed")]}])
    rendered = render_pages(data)[0]
    wb = _word_boxes(rendered)
    cboxes = [_box_left(wb["Married"], True), _box_left(wb["Single"], False), _box_left(wb["Widowed"], False)]
    md = "<table><tr><td>[ ] Married [x] Single [ ] Widowed</td></tr></table>"
    edits = acro_checkbox_edits(md, rendered, cboxes)
    assert len(edits) == 2  # the agreeing `[ ] Widowed` is untouched
    assert apply_edits(md, edits) == "<table><tr><td>[x] Married [ ] Single [ ] Widowed</td></tr></table>"


def test_checkbox_mark_far_from_word_or_without_widget_is_left_alone():
    rendered = render_pages(make_pdf([{"texts": [(100, 700, 11, "Married"), (100, 670, 11, "Single")]}]))[0]
    wb = _word_boxes(rendered)
    cboxes = [_box_left(wb["Married"], True)]
    md = "[ ] see the notes on page two Married\n[x] Single\n"  # the mark is not right before `Married`
    assert acro_checkbox_edits(md, rendered, cboxes) == []


# --- missing text-field values -----------------------------------------------------------------------------


def _label_page():
    data = make_pdf(
        [{"texts": [(72, 700, 11, "Name:"), (72, 670, 11, "City"), (300, 640, 11, "Employer"), (72, 610, 11, "Zip")]}]
    )
    rendered = render_pages(data)[0]
    wb = _word_boxes(rendered)
    ex0, _ey0, _ex1, ey1 = wb["Employer"]
    texts = [
        _field_right(wb["Name:"], "Jane Q. Doe"),
        _field_right(wb["City"], "Boston"),
        (ex0, ey1 + 4, ex0 + 200, ey1 + 20, "Acme Corp"),  # field below its label
        _field_right(wb["Zip"], "02139"),
    ]
    return rendered, texts


def test_missing_text_values_are_written_beside_their_labels():
    rendered, texts = _label_page()
    md = "Name:\nCity [___]\nEmployer\n<table><tr><td>Zip</td><td></td></tr></table>"
    out = apply_edits(md, acro_text_value_edits(md, rendered, texts))
    assert out == (
        "Name: Jane Q. Doe\nCity Boston\nEmployer: Acme Corp\n<table><tr><td>Zip: 02139</td><td></td></tr></table>"
    )


def test_present_text_values_are_never_duplicated():
    rendered, texts = _label_page()
    md = "Name: **JANE  Q. DOE**\nCity: boston\nEmployer\nAcme Corp\nZip 02139"
    assert acro_text_value_edits(md, rendered, texts) == []


def test_text_value_with_unaligned_label_is_skipped():
    rendered, texts = _label_page()
    md = "Something else entirely\n"
    assert acro_text_value_edits(md, rendered, texts[:1]) == []


# --- reading widgets from a real AcroForm PDF --------------------------------------------------------------


def _form_pdf() -> bytes:
    return _acro_pdf(
        [
            (72, 700, 11, "Name:"),
            (72, 670, 11, "PIN:"),
            (72, 640, 11, "Code:"),
            (84, 610, 11, "Married"),
            (84, 580, 11, "Single"),
        ],
        [
            {"kind": "tx", "rect": (130, 696, 330, 714), "value": "Jane Q. Doe"},
            {"kind": "tx", "rect": (130, 666, 330, 684), "value": "hunter2", "Ff": 1 << 13},  # password
            {"kind": "tx", "rect": (130, 636, 330, 654), "value": "secret1", "F": 2},  # hidden
            {"kind": "tx", "rect": (350, 636, 550, 654), "value": "secret2", "F": 32},  # no-view
            {"kind": "tx", "rect": (350, 696, 550, 714), "value": "____"},  # placeholder only
            {"kind": "tx", "rect": (350, 666, 550, 684)},  # no value
            {"kind": "tx", "rect": (350, 606, 550, 624), "value": "José Müller"},  # UTF-16 value
            {"kind": "cb", "rect": (70, 609, 80, 619), "checked": True},
            {"kind": "cb", "rect": (70, 579, 80, 589), "checked": False},
        ],
    )


def test_text_widgets_skip_password_hidden_noview_and_blank_fields():
    data = _form_pdf()
    tws = acroform_text_widgets(data, 0)
    assert [v for *_r, v in tws] == ["Jane Q. Doe", "José Müller"]
    x0, y0, x1, y1, _v = tws[0]
    assert (round(x0), round(y0), round(x1), round(y1)) == (130, 78, 330, 96)  # top-down page points
    n_int, cboxes = acroform_widgets(data, 0)
    assert n_int == 9
    assert [c[4] for c in cboxes] == [True, False]  # read from each widget's /AS appearance state


def test_repair_acroform_fills_missing_values_and_corrects_marks_end_to_end():
    data = _form_pdf()
    rendered = render_pages(data)[0]
    md = "Name:\nPIN:\nCode:\n[ ] Married\n[x] Single\n"
    out = repair_acroform(md, rendered, data, 0)
    assert out == "Name: Jane Q. Doe\nPIN:\nCode:\n[x] Married\n[ ] Single\n"  # no password/hidden values
    assert repair_acroform(out, rendered, data, 0) == out  # idempotent: present values are not repeated
    assert repair_acroform(md, rendered, data, 0, do_id_boxes=False) == md


def test_text_widgets_on_unreadable_input_are_empty():
    assert acroform_text_widgets(b"\x89PNG\r\n\x1a\n" + b"\0" * 64, 0) == []
    assert acroform_text_widgets(b"%PDF-1.4 not really a pdf", 0) == []
    assert acroform_text_widgets(_form_pdf(), 5) == []

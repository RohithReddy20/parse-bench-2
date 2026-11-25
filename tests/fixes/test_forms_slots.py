from docparse.fixes.forms import repair
from docparse.fixes.forms_slots import blank_line, colon_text, hint_text
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf

TYPED = [(400, 100, 10, "Smith")]  # a value typed elsewhere on the page: the page has an overlay


def _fix(md: str, printed, options, typed=TYPED) -> str:
    data = make_pdf([{"texts": printed, "overlay": typed}])
    return repair(md, render_pages(data)[0], overlay_fonts(data), options, data)


def test_printed_text_between_a_colon_and_a_blank_marker_becomes_a_hint_and_text_after_it_a_line():
    printed = [(40, 700, 10, "Behinderung:"), (140, 700, 10, "1"), (150, 700, 10, "="), (160, 700, 10, "Ja")]
    printed += [(200, 700, 10, "EUR")]
    md = "Behinderung: 1 = Ja [___] EUR"
    assert _fix(md, printed, {"blanks": True}) == "Behinderung (1 = Ja): [___]\nEUR"
    printed = [(40, 700, 10, "Arbeitslohn:"), (140, 700, 10, "127"), (200, 700, 10, "EUR")]
    assert _fix("Arbeitslohn: 127 [___]", printed, {"blanks": True}) == "127 Arbeitslohn: [___]"  # a box code
    md = "Arbeitslohn: 127 [___] EUR"
    assert _fix(md, printed, {"combs": True}) == md  # off unless asked
    data = make_pdf([{"texts": printed}])  # no overlay: not a filled form
    assert repair(md, render_pages(data)[0], overlay_fonts(data), {"blanks": True}, data) == md


def test_a_line_holding_typed_text_is_left_alone():
    printed = [(40, 700, 10, "Arbeitslohn:"), (140, 700, 10, "127")]
    md = "Arbeitslohn: 127 [___] Smith"
    assert _fix(md, printed, {"blanks": True}, typed=[(200, 700, 10, "Smith")]) == md


def test_blank_line_rules():
    assert blank_line("First Name: [___] Middle Name: [___]") == "First Name: [___]\nMiddle Name: [___]"
    assert blank_line("Day: [___] Month: [___] Year: [___]") == "Day: [___] Month: [___] Year: [___]"  # date parts
    assert blank_line("I am fluent in English and [___], which is") == "I am fluent in English and [___], which is"
    assert blank_line("Elle s'est terminée le : D15 [___]") == "D15 Elle s'est terminée le: [___]"
    assert blank_line("252 – dont intérêts: [___] (2TR)") == "252 – dont intérêts: [___]\n(2TR)"
    assert blank_line("* Namenszusatz (Beispiel: Freifrau, Graf): [___]") == (
        "* Namenszusatz (Beispiel Freifrau, Graf): [___]"  # a hint's own colon would cut the caption short
    )
    assert blank_line("支店名 Name of branch [___]") == "支店名 Name of branch: [___]"
    assert blank_line("**Last name(s)** [___]") == "**Last name(s)** [___]"  # one part: no colon to steal a caption
    assert blank_line("Will tickets be sold?: No [___]") == "Will tickets be sold?: No [___]"  # a choice is an answer
    # extras between the caption and its colon: a box code moves to the front, an italic hint is bracketed
    assert blank_line("* gain imposable *Après abattement*** **1TZ**: [___]") == (
        "* 1TZ gain imposable (Après abattement): [___]"
    )
    assert blank_line("**238 Intérêts inclus *lignes 233 + 237*****: [___]") == (
        "**238 Intérêts inclus** (lignes 233 + 237): [___]"
    )
    assert blank_line("Telephone Number (999) 999-9999: [___]") == "Telephone Number (999 999-9999): [___]"
    assert blank_line("**Code postal :** [___]") == "**Code postal :** [___]"


def test_a_cell_ending_in_box_decoration_or_a_worded_hint_gets_a_blank_marker():
    assert hint_text("Totale credito 5 ,00") == "Totale credito (5 ,00): [___]"
    assert hint_text("22 UCC at the end of the year (col. 7 minus col. 21) 9") == (
        "22 UCC at the end of the year (col. 7 minus col. 21 9): [___]"
    )
    assert hint_text("Branch number (5 digits)") == "Branch number (5 digits): [___]"
    assert hint_text("Geburtsort (falls Ausland: auch Staat)") == "Geburtsort (falls Ausland auch Staat): [___]"
    assert hint_text("IHT401") is None  # a code, not a decoration
    assert hint_text("Advertising") is None
    assert hint_text("Yes [ ] 5") is None  # a box


def test_cells_rewrites_printed_caption_cells_but_not_beside_boxes_or_over_printed_cells():
    printed = [(40, 700, 10, "Totale credito"), (140, 700, 10, "5"), (150, 700, 10, ",00")]
    md = "<table><tr><td>Totale credito 5 ,00</td><td></td></tr></table>"
    assert (
        _fix(md, printed, {"cells": True}) == "<table><tr><td>Totale credito (5 ,00): [___]</td><td></td></tr></table>"
    )
    boxed = "<table><tr><td>Totale credito 5 ,00</td><td>[ ]</td></tr></table>"
    assert _fix(boxed, printed, {"cells": True}) == boxed
    printed2 = [*printed, (40, 680, 10, "Due"), (80, 680, 10, "by"), (100, 680, 10, "April")]
    over = "<table><tr><td>Totale credito 5 ,00</td></tr><tr><td>Due by April</td></tr></table>"
    assert _fix(over, printed2, {"cells": True}) == over


def test_hints_on_lines_without_a_marker():
    assert colon_text("**Religion**: 1 = Austritt, 2 = Wechsel") == "**Religion** (1 = Austritt, 2 = Wechsel): [___]"
    assert colon_text("Answer: Yes") is None
    long = "Note: " + " ".join(["word"] * 13)
    assert colon_text(long) is None
    printed = [(40, 700, 10, "Religion:"), (140, 700, 10, "1"), (150, 700, 10, "="), (160, 700, 10, "Austritt")]
    assert _fix("Religion: 1 = Austritt", printed, {"hints": True}) == "Religion (1 = Austritt): [___]"


def test_blank_forms_needs_several_blank_markers_and_words_from_the_pdf():
    printed = [(40, 700, 10, "Behinderung:"), (140, 700, 10, "1"), (150, 700, 10, "="), (160, 700, 10, "Ja")]
    printed += [(40, 680, 10, "Name:"), (40, 660, 10, "Ort:")]
    data = make_pdf([{"texts": printed}])  # no overlay
    rp, fonts = render_pages(data)[0], overlay_fonts(data)
    md = "Behinderung: 1 = Ja [___]\nName: [___]\nOrt: [___]"
    on = {"blanks": True, "blank_forms": True}
    assert repair(md, rp, fonts, on, data) == "Behinderung (1 = Ja): [___]\nName: [___]\nOrt: [___]"
    assert repair(md, rp, fonts, {"blanks": True}, data) == md  # off unless asked
    two = "Behinderung: 1 = Ja [___]\nName: [___]"  # two markers: not taken for a form
    assert repair(two, rp, fonts, on, data) == two
    absent = "Behinderung: 9 = Nein [___]\nName: [___]\nOrt: [___]"  # the tail is not on the page
    assert repair(absent, rp, fonts, on, data) == absent

from docparse.fixes.forms import repair
from docparse.pdf.boxes import check_boxes
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf

# "Married? Yes [box] No [box]" on one line, and "[box] Email me" on the next
TEXTS = [(72, 700, 11, "Married?"), (150, 700, 11, "Yes"), (210, 700, 11, "No"), (100, 660, 11, "Email me")]
RECTS = [(175, 698, 11, 11), (228, 698, 11, 11), (82, 658, 11, 11)]


def _fix(md: str, ticks: list[tuple[float, float]], rects=RECTS) -> str:
    overlay = [(x, y, 10, "X") for x, y in ticks] + [(300, 600, 10, "SMITH")]
    data = make_pdf([{"texts": TEXTS, "rects": rects, "overlay": overlay}])
    return repair(md, render_pages(data)[0], overlay_fonts(data), {"boxes": True}, data, 0)


def test_drawn_boxes_are_found_once_each():
    data = make_pdf([{"texts": TEXTS, "rects": RECTS + [(176, 699, 9, 9)]}])  # an inner border is no new box
    boxes = [tuple(round(v) for v in b) for b in check_boxes(data)]  # outlines include the 1-point stroke
    assert boxes == [(174, 82, 187, 95), (227, 82, 240, 95), (81, 122, 94, 135)]


def test_a_tick_in_a_box_checks_that_option_and_an_empty_box_unchecks_it():
    md = "Married? Yes [x] No [ ]\n\n[x] Email me"
    assert _fix(md, [(231, 700)]) == "Married? Yes [ ] No [x]\n\n[ ] Email me"
    assert _fix(md, [(178, 700), (85, 660)]) == md


def test_stray_tick_text_beside_a_box_is_dropped_once_the_states_are_set():
    md = "Married? Yes [ ] **X** No [ ]"
    assert _fix(md, [(178, 700)]) == "Married? Yes [x] No [ ]"


def test_a_tick_without_a_drawn_box_still_checks_the_label_next_to_it():
    assert _fix("Married? Yes [ ] No [ ]", [(231, 700)], rects=[]) == "Married? Yes [ ] No [x]"
    # with no drawn boxes, an unticked option keeps the read's state
    assert _fix("Married? Yes [x] No [ ]", [(231, 700)], rects=[]) == "Married? Yes [x] No [x]"


def test_marks_whose_label_is_not_on_the_page_keep_their_state():
    md = "Divorced [x] Widowed [ ]"
    assert _fix(md, [(231, 700)]) == md

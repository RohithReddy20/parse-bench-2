from docparse.core.config import Config
from docparse.fixes.forms import repair
from docparse.pdf.overlay import overlay_fonts
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf


def test_box_grids_labels_checkbox_only_rating_cells():
    data = make_pdf(
        [
            {
                "texts": [(72, 700, 11, "Walking"), (220, 700, 11, "Good"), (300, 700, 11, "Fair")],
                "overlay": [(310, 680, 11, "X")],
            }
        ]
    )
    rendered, fonts = render_pages(data)[0], overlay_fonts(data)
    md = (
        "<table>"
        "<tr><th>Activity</th><th>Good</th><th>Fair</th><th>Poor</th></tr>"
        "<tr><td>Walking</td><td>[ ]</td><td>[x]</td><td>[ ]</td></tr>"
        "<tr><td>Standing</td><td>[x]</td><td>[ ]</td><td>[ ]</td></tr>"
        "</table>"
    )
    out = repair(md, rendered, fonts, {"box_grids": True}, pdf_bytes=data, page_index=0)
    assert "<td>[ ] Good</td><td>[x] Fair</td><td>[ ] Poor</td>" in out
    assert "<td>[x] Good</td><td>[ ] Fair</td><td>[ ] Poor</td>" in out


def test_box_grids_cleans_duplicate_box_and_splits_caption_before_inline_boxes():
    data = make_pdf([{"texts": [(72, 700, 11, "OAS"), (72, 660, 11, "Sexe")], "overlay": [(150, 700, 11, "X")]}])
    rendered, fonts = render_pages(data)[0], overlay_fonts(data)
    md = "[ ] OAS [x]\n\n**Sexe :** [ ] Masculin [x] Féminin"
    out = repair(md, rendered, fonts, {"box_grids": True}, pdf_bytes=data, page_index=0)
    assert "[x] OAS" in out
    assert "**Sexe :** \n[ ] Masculin [x] Féminin" in out


def test_bare_ticks_brackets_overlay_tick_and_marks_empty_drawn_sibling_box():
    # Drawn 10x10 boxes at x=180..190 (ticked) and x=280..290 (empty), y=700..710
    data = make_pdf(
        [
            {
                "texts": [(72, 700, 11, "Language:"), (198, 700, 11, "English"), (298, 700, 11, "French")],
                "overlay": [(182, 701, 11, "X")],
                "rects": [(180, 700, 10, 10), (280, 700, 10, 10)],
            }
        ]
    )
    rendered, fonts = render_pages(data)[0], overlay_fonts(data)
    md = "Language: X English French"
    out = repair(md, rendered, fonts, {"bare_ticks": True}, pdf_bytes=data, page_index=0)
    assert out == "Language: [x] English [ ] French"


def test_bare_ticks_does_not_touch_printed_x():
    data = make_pdf(
        [
            {
                "texts": [(72, 700, 11, "Type X English")],
                "overlay": [(300, 600, 11, "Alice")],
            }
        ]
    )
    rendered, fonts = render_pages(data)[0], overlay_fonts(data)
    md = "Type X English\nName: Alice"
    assert repair(md, rendered, fonts, {"bare_ticks": True}, pdf_bytes=data, page_index=0) == md


def test_box_grid_and_bare_tick_options_are_validated():
    cfg = Config.load("configs/standin.yaml", ["form_options.box_grids=true", "form_options.bare_ticks=true"])
    assert cfg.form_options == {"box_grids": True, "bare_ticks": True}

from docparse.core.config import Config
from docparse.core.ir import Box, LayoutItem
from docparse.fixes.layout_kdl import add_uncovered, apply_kdl, regions_in_points, split_coarse, words_as_items
from docparse.fixes.layout_refine import words_in_points
from docparse.models.adapters.heron import Detection
from docparse.pdf.render import render_pages
from tests.pdfgen import make_pdf

# two short paragraphs, one above the other, and a caption-like line further down
LINES = [(72, 700, 11, "Alpha paragraph text here"), (72, 640, 11, "Beta paragraph text here")]
LINES += [(72, 560, 11, "Gamma caption line")]


def _rendered():
    return render_pages(make_pdf([{"texts": LINES}]))[0]


def _item(type_, box, md="", label="text"):
    return LayoutItem(type=type_, label=label, md=md, bbox=Box(x0=box[0], y0=box[1], x1=box[2], y1=box[3]))


def _det(label, box_pt, rendered):
    k = 1.0 / rendered.scale  # detections are in page pixels
    return Detection(label=label, score=1.0, box=tuple(v * k for v in box_pt))


def _boxes(rendered):
    """Point boxes around each line of LINES (y grows downward in the IR)."""
    words = words_in_points(rendered)
    out = []
    for *_, text in LINES:
        first = text.split()[0]
        w = next(w for w in words if w.text == first)
        out.append((w.x0 - 2, w.y0 - 2, w.x0 + 160, w.y1 + 2))
    return out


def test_regions_in_points_scale_pixel_boxes():
    rendered = _rendered()
    (r,) = regions_in_points([_det("Text", (10, 20, 110, 60), rendered)], rendered)
    assert r.label == "Text" and abs(r.box.x0 - 10) < 1e-6 and abs(r.box.y1 - 60) < 1e-6


def test_words_as_items_reads_each_region_from_the_text_layer():
    rendered = _rendered()
    a, b, c = _boxes(rendered)
    dets = [_det("Title", a, rendered), _det("Text", b, rendered), _det("Chart", c, rendered)]
    dets.append(_det("Text", (400, 300, 500, 350), rendered))  # holds no words: dropped
    items = words_as_items(regions_in_points(dets, rendered), words_in_points(rendered))
    assert [(i.type, i.label) for i in items] == [("Title", "doc_title"), ("Text", "text"), ("Picture", "image")]
    assert items[0].md == "Alpha paragraph text here" and items[2].md == ""


def test_add_uncovered_adds_only_regions_no_item_overlaps():
    rendered = _rendered()
    a, b, c = _boxes(rendered)
    items = [_item("Text", a, "Alpha paragraph text here")]
    dets = [_det("Text", a, rendered), _det("Text", b, rendered), _det("Picture", (300, 300, 500, 400), rendered)]
    out = add_uncovered(items, regions_in_points(dets, rendered), words_in_points(rendered))
    assert out[0] is items[0] and len(out) == 3
    assert out[1].md == "Beta paragraph text here" and out[2].type == "Picture"


def test_split_coarse_replaces_a_table_holding_several_text_regions():
    rendered = _rendered()
    a, b, c = _boxes(rendered)
    table = _item("Table", (a[0] - 5, a[1] - 5, b[2] + 5, c[3] + 5), "| x |", label="table")
    other = _item("Text", (400, 50, 500, 60), "elsewhere")
    dets = [_det("Text", a, rendered), _det("Text", b, rendered), _det("Caption", c, rendered)]
    out = split_coarse([other, table], regions_in_points(dets, rendered), words_in_points(rendered))
    assert out[0] is other and [i.type for i in out[1:]] == ["Text", "Text", "Caption"]
    assert out[2].md == "Beta paragraph text here"
    # KDL itself sees a table there: the table stays
    dets.append(_det("Table", (table.bbox.x0, table.bbox.y0, table.bbox.x1, table.bbox.y1), rendered))
    assert split_coarse([other, table], regions_in_points(dets, rendered), words_in_points(rendered)) == [other, table]
    # one text region inside is not enough
    one = [_det("Text", a, rendered)]
    assert split_coarse([table], regions_in_points(one, rendered), words_in_points(rendered)) == [table]


def test_apply_kdl_changes_nothing_without_options_or_text_layer():
    rendered = _rendered()
    items = [_item("Text", (60, 80, 420, 175), "model text")]
    dets = [_det("Text", (300, 300, 500, 400), rendered)]
    assert apply_kdl(items, rendered, dets, {}, "") is items
    rendered.words = rendered.text_words = None
    assert apply_kdl(items, rendered, dets, {"kdl_add": True, "kdl_split": True, "kdl_words": True}, "") is items


def test_kdl_options_are_valid_config_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "page_model: {repo: r, revision: v, dtype: bf16}\npage_runtime: published\nprompt_sha: published\n"
        "layout_options: {kdl_add: true, kdl_split: true, kdl_replace: true, kdl_words: true}\n"
    )
    assert Config.load(path).layout_options["kdl_split"] is True

from docparse.models.adapters import heron, kdl
from docparse.models.adapters.ocria import OWN, build_page_call
from docparse.models.calls import ModelResult
from docparse.pdf.render import pixel_hash, render_pages
from tests.pdfgen import simple_pdf


def _box(x0, y0, x1, y1, label):
    return f"<|box_start|>{x0} {y0} {x1} {y1}<|box_end|><|ref_start|>{label}<|ref_end|>"


def test_layout_call_is_keyed_by_the_page_image_and_settings():
    image = render_pages(simple_pdf("Detect me"))[0].image
    call = kdl.layout_call(pixel_hash(image))
    assert call.inputs == [pixel_hash(image)] and call.prompt_id == "kdl-nano/layout"
    assert call.model.repo == "KDLAI/KDL-Frontier-Parser-nano-v2" and len(call.model.revision) == 40
    assert call.params["size"] == 1036 and call.params["max_new_tokens"] == 6000
    page_call, _ = build_page_call(image, OWN, "vllm-0.34.0", "s" * 64)
    assert call.inputs == page_call.inputs and call.key() not in (
        page_call.key(),
        heron.detect_call(call.inputs[0]).key(),
    )
    assert kdl.layout_call("0" * 64).key() != call.key()


def test_read_detections_scales_the_thousand_grid_to_page_pixels():
    text = (
        _box(100, 50, 900, 100, "title") + _box(100, 120, 500, 900, "text") + _box(0, 0, 0, 10, "text") + "<|im_end|>"
    )
    result = ModelResult(key="k", output={"text": text, "width": 2000, "height": 3000})
    title, body = kdl.read_detections(result)
    assert (title.label, title.score, title.box) == ("Title", 1.0, (200.0, 150.0, 1800.0, 300.0))
    assert body.label == "Text" and body.box == (200.0, 360.0, 1000.0, 2700.0)
    assert kdl.read_detections(ModelResult(key="k", output={})) == []


def test_containers_with_children_are_dropped_and_children_kept():
    text = (
        _box(100, 100, 900, 600, "image_block")
        + _box(120, 110, 880, 500, "image")
        + _box(120, 510, 880, 590, "image_caption")
        + _box(100, 650, 900, 900, "list")
        + _box(110, 660, 890, 700, "text")
        + _box(110, 710, 890, 750, "list_item")
        + _box(100, 920, 900, 980, "footer")
        + _box(950, 10, 990, 20, "unseen_label")
    )
    result = ModelResult(key="k", output={"text": text, "width": 1000, "height": 1000})
    assert [d.label for d in kdl.read_detections(result)] == [
        "Picture", "Caption", "List-item", "List-item", "Page-footer", "Text"
    ]  # fmt: skip
    raw = kdl.read_regions(result)
    assert [(r.raw, r.in_list) for r in raw][:4] == [
        ("image", False), ("image_caption", False), ("text", True), ("list_item", True)
    ]  # fmt: skip
    assert len(kdl.parse_regions(result.output["text"])) == 8


def test_a_lone_image_block_is_a_picture():
    result = ModelResult(key="k", output={"text": _box(10, 10, 500, 500, "image_block"), "width": 100, "height": 100})
    (d,) = kdl.read_detections(result)
    assert d.label == "Picture" and d.box == (1.0, 1.0, 50.0, 50.0)

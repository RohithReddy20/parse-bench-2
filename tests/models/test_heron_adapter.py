from docparse.models.adapters import heron
from docparse.models.adapters.ocria import OWN, build_page_call
from docparse.models.calls import ModelResult
from docparse.pdf.render import pixel_hash, render_pages
from tests.pdfgen import simple_pdf


def test_detect_call_is_keyed_by_the_page_image_and_settings():
    image = render_pages(simple_pdf("Detect me"))[0].image
    call = heron.detect_call(pixel_hash(image))
    assert call.inputs == [pixel_hash(image)] and call.params == {"threshold": 0.2, "max_side": 1600}
    assert call.model.repo == "docling-project/docling-layout-heron" and len(call.model.revision) == 40
    page_call, _ = build_page_call(image, OWN, "vllm-0.34.0", "s" * 64)
    assert call.inputs == page_call.inputs and call.key() != page_call.key()  # same blob, different call
    assert heron.detect_call("0" * 64).key() != call.key()


def test_read_detections_drops_degenerate_boxes():
    result = ModelResult(key="k", output={"detections": [
        {"label": "Text", "score": 0.91, "box": [10, 20, 110, 60]},
        {"label": "Picture", "score": 0.4, "box": [5, 5, 5, 50]},
    ]})  # fmt: skip
    (d,) = heron.read_detections(result)
    assert (d.label, d.score, d.box) == ("Text", 0.91, (10.0, 20.0, 110.0, 60.0))
    assert heron.read_detections(ModelResult(key="k", output={})) == []

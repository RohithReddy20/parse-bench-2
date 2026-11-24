"""The output contract (design §5.4): our records validate against the pinned harness's schema and are read by
the adapter the harness itself falls to, with no adapter of ours registered.

For any ParseOutput that carries `layout_pages`, the pinned harness selects `LlamaParseLayoutAdapter` by its
`matches` rule (it needs `llama-cloud` installed). That is the adapter OCRia-1's published layout records are
scored through, so it is the one ours must be read by too."""

from parse_bench.evaluation.layout_adapters.adapters import LlamaParseLayoutAdapter
from parse_bench.evaluation.layout_adapters.registry import create_layout_adapter_for_result
from parse_bench.schemas.parse_output import ParseLayoutPageIR
from parse_bench.schemas.pipeline_io import InferenceResult

from docparse.bench.record import result_record
from docparse.core.ir import Box, LayoutItem
from docparse.pipeline import parse
from tests.pdfgen import simple_pdf
from tests.test_pipeline import CONFIG, FakeClient

PUBLISHED_LABELS = {"text", "paragraph_title", "image", "header", "footer", "table", "doc_title"}


def _record():
    doc = parse(simple_pdf("Hello contract"), CONFIG, FakeClient()).document
    doc.pages[0].layout = [
        LayoutItem(
            type="Section-header", label="paragraph_title", md="# Page 1", bbox=Box(x0=72, y0=80, x1=300, y1=100)
        ),
        LayoutItem(type="Text", label="text", md="body 1", bbox=Box(x0=72, y0=110, x1=400, y1=130)),
        LayoutItem(type="Picture", label="image", md="", bbox=Box(x0=72, y0=200, x1=200, y1=300)),
        LayoutItem(
            type="Table",
            label="table",
            md="<table><tr><th>a</th></tr><tr><td>1</td></tr></table>",
            bbox=Box(x0=72, y0=320, x1=400, y1=400),
        ),
    ]
    return result_record("layout/case", "docparse_m0", doc)


def test_record_validates_as_inference_result():
    result = InferenceResult.model_validate(_record())
    assert result.output.pages[0].markdown.startswith("# Page 1")
    (layout,) = result.output.layout_pages
    assert isinstance(layout, ParseLayoutPageIR) and layout.page_number == 1
    assert (layout.width, layout.height) == (612.0, 792.0)
    for item in layout.items:
        assert item.bbox is not None and len(item.layout_segments) == 1 and item.layout_segments[0].label


def test_default_adapter_is_selected_and_reads_every_item():
    result = InferenceResult.model_validate(_record())
    adapter = create_layout_adapter_for_result(result)
    assert type(adapter) is LlamaParseLayoutAdapter
    output = adapter.to_layout_output(result)
    assert len(output.predictions) == 4
    assert all(0 <= p.bbox[0] < p.bbox[2] and 0 <= p.bbox[1] < p.bbox[3] for p in output.predictions)


def test_published_ocria_record_shape_gets_the_same_adapter():
    record = _record()
    record["pipeline_name"] = record["output"]["pipeline_name"] = "ocria_1"
    record["raw_output"] = {}
    assert type(create_layout_adapter_for_result(InferenceResult.model_validate(record))) is LlamaParseLayoutAdapter


def test_record_without_layout_is_not_claimed_by_a_vendor_adapter():
    record = _record()
    record["output"]["layout_pages"] = []
    adapter = create_layout_adapter_for_result(InferenceResult.model_validate(record))
    assert type(adapter).__name__ == "NormalizedLayoutOutputAdapter"


def test_segment_labels_are_canonical():
    from parse_bench.inference.layout_extraction import _canonical_label

    for label in PUBLISHED_LABELS:
        assert _canonical_label(label, label_version="v2") is not None

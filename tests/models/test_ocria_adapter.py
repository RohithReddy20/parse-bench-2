from docparse.models.adapters import ocria
from docparse.models.calls import ModelResult
from docparse.pdf.render import pixel_hash, render_pages
from evalkit import pins
from tests.pdfgen import simple_pdf


def _image(text="Hello"):
    return render_pages(simple_pdf(text))[0].image


def test_own_revision_matches_the_pin():
    assert ocria.OWN.revision == pins.OCRIA_REVISION and ocria.OWN.repo == pins.OCRIA_REPO
    assert ocria.PUBLISHED.revision == f"published-{pins.RESULTS_REVISION[:12]}"


def test_call_carries_the_harness_settings():
    image = _image()
    call, blobs = ocria.build_page_call(image, ocria.OWN, "vllm-0.31.0", "s" * 64)
    assert call.params == {"temperature": 0.0, "max_tokens": 24576, "enable_thinking": False}
    assert call.prompt_id == "ocria-1/page" and call.inputs == [pixel_hash(image)]
    assert list(blobs) == call.inputs and blobs[call.inputs[0]][:8] == b"\x89PNG\r\n\x1a\n"


def test_blobs_are_lazy():
    call, blobs = ocria.build_page_call(_image(), ocria.OWN, "vllm-0.31.0", "s" * 64)
    assert blobs._data is None  # nothing is encoded until a client needs the blob
    assert call.inputs[0] in blobs and len(blobs) == 1
    _ = blobs[call.inputs[0]]
    assert blobs._data is not None


def test_published_and_own_never_share_a_key():
    image = _image()
    own, _ = ocria.build_page_call(image, ocria.OWN, "vllm-0.31.0", "s" * 64)
    pub, _ = ocria.build_page_call(image, ocria.PUBLISHED, ocria.PUBLISHED_RUNTIME, "published")
    assert own.key() != pub.key()
    assert ocria.build_page_call(_image("Other"), ocria.OWN, "vllm-0.31.0", "s" * 64)[0].key() != own.key()


def test_read_raw_answer_is_postprocessed():
    raw = "```markdown\n<page_header>Head</page_header>\n\n<b>Bold</b> text\n```"
    read = ocria.read_page_result(ModelResult(key="k", output={"answer": raw, "finish_reason": "stop"}))
    assert read.raw_answer == raw and read.finish_reason == "stop"
    assert "**Bold** text" in read.markdown and "page_header" not in read.markdown and "Head" in read.markdown


def test_read_standin_markdown_is_used_as_is():
    layout = {"page_number": 1, "items": []}
    read = ocria.read_page_result(ModelResult(key="k", output={"markdown": "<b>kept</b>", "layout_page": layout}))
    assert read.markdown == "<b>kept</b>" and read.raw_answer is None and read.layout_page == layout


def test_empty_and_truncated_answers():
    assert ocria.read_page_result(ModelResult(key="k", output={"answer": ""})).markdown == ""
    assert ocria.read_page_result(ModelResult(key="k", output={"answer": None})).markdown == ""
    cut = ocria.read_page_result(
        ModelResult(key="k", output={"answer": "```markdown\n<table><tr><td>1", "finish_reason": "length"})
    )
    assert cut.finish_reason == "length" and "<table>" in cut.markdown

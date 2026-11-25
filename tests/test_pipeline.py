import io

import pytest
from PIL import Image

from docparse.bench.record import result_record
from docparse.core.config import Config
from docparse.core.errors import ParseError
from docparse.models.adapters import ocria
from docparse.models.cache import Cache
from docparse.models.calls import CacheMiss, ModelResult
from docparse.models.client import CacheOnlyClient, ReplayClient
from docparse.models.manifest import ManifestWriter, read_manifest
from docparse.pipeline import parse
from docparse.write.markdown import write_document
from tests.pdfgen import make_pdf, simple_pdf

CONFIG = Config(page_model=ocria.PUBLISHED, page_runtime="published", prompt_sha="published")


class FakeClient:
    """Answers every page call with markdown naming the page, plus a layout box in pixels-as-points."""

    def __init__(self, layout=True, answer=None):
        self.calls, self.metas, self.layout, self.answer = [], [], layout, answer

    def get(self, call, blobs, meta=None):
        self.calls.append(call)
        self.metas.append(meta)
        n = len(self.calls)
        if self.answer is not None:
            return ModelResult(key=call.key(), output={"answer": self.answer, "finish_reason": "length"})
        layout_page = None
        if self.layout:
            seg = {"x": 10.0, "y": 20.0, "w": 100.0, "h": 30.0, "label": "text"}
            layout_page = {
                "page_number": 1,
                "items": [{"type": "Text", "md": f"page {n}", "bbox": seg, "layout_segments": [seg]}],
            }
        return ModelResult(
            key=call.key(), output={"markdown": f"# Page {n}\n\nbody {n}", "layout_page": layout_page}, tokens_out=11
        )


def test_one_ir_page_per_pdf_page():
    pdf = make_pdf([{"texts": [(72, 700, 12, "A")]}, {"size": (300, 400), "texts": [(20, 300, 10, "B")]}])
    client = FakeClient()
    result = parse(pdf, CONFIG, client)
    doc = result.document
    assert result.pending == [] and doc.n_pages == 2 and len(client.calls) == 2
    assert [p.index for p in doc.pages] == [0, 1]
    assert (doc.pages[1].width, doc.pages[1].height) == (300, 400)
    assert write_document(doc) == "# Page 1\n\nbody 1\n\n# Page 2\n\nbody 2"
    assert doc.config_hash == CONFIG.config_hash() and len(doc.sha256) == 64
    assert doc.pages[0].cost.tokens_out == 11
    assert client.metas[1]["page"] == 1 and "doc" in client.metas[1]
    assert client.calls[0].model == ocria.PUBLISHED and client.calls[0].key() != client.calls[1].key()


def test_parse_sees_bytes_only():
    import inspect

    assert list(inspect.signature(parse).parameters) == ["pdf_bytes", "config", "client"]


def test_cache_miss_gives_pending_and_a_manifest(tmp_path):
    cache, manifest = Cache(tmp_path / "c"), ManifestWriter(tmp_path / "m")
    pdf = make_pdf([{"texts": [(72, 700, 12, "A")]}, {"texts": [(72, 700, 12, "B")]}])
    result = parse(pdf, CONFIG, CacheOnlyClient(cache, manifest))
    assert result.document is None and len(result.pending) == 2
    rows = list(read_manifest(tmp_path / "m"))
    assert [c.key() for c, _ in rows] == result.pending
    assert (tmp_path / "m" / "blobs" / rows[0][0].inputs[0]).read_bytes()[:4] == b"\x89PNG"
    with pytest.raises(CacheMiss):
        parse(pdf, CONFIG, ReplayClient(cache))


@pytest.mark.parametrize("data", [b"", b"not a pdf", b"%PDF-1.7 broken"])
def test_unreadable_bytes_raise_parse_error(data):
    with pytest.raises(ParseError):
        parse(data, CONFIG, FakeClient())


def test_image_upload_is_one_page():
    buf = io.BytesIO()
    Image.new("RGB", (850, 1100), "white").save(buf, format="JPEG")
    doc = parse(buf.getvalue(), CONFIG, FakeClient()).document
    assert doc.n_pages == 1 and doc.pages[0].width == pytest.approx(204)


@pytest.mark.parametrize(
    "answer",
    ["", "```markdown\n<table><tr><td>cut off", "```markdown\n# T\n\n```\nunclosed code", "<!-- only thinking"],
)
def test_truncated_or_empty_answer_still_parses_and_writes(answer):
    doc = parse(simple_pdf(), CONFIG, FakeClient(answer=answer)).document
    record = result_record("text/x", "p", doc)
    assert isinstance(record["output"]["markdown"], str)
    if "cut off" in answer:
        assert doc.pages[0].regions[-1].kind == "opaque"
        assert "cut off" in record["output"]["markdown"]


@pytest.mark.parametrize(
    "page", [{"rotate": 90}, {"rotate": 270}, {"crop": (36, 36, 576, 756)}, {"size": (5000, 5000)}]
)
def test_layout_boxes_stay_inside_the_displayed_page(page):
    doc = parse(simple_pdf(**page), CONFIG, FakeClient()).document
    p = doc.pages[0]
    assert p.layout and p.width > 0 and p.height > 0
    for item in p.layout:
        assert 0 <= item.bbox.x0 < item.bbox.x1 <= p.width and 0 <= item.bbox.y0 < item.bbox.y1 <= p.height
    if page.get("rotate"):
        assert (p.width, p.height) == (792, 612) and p.rotation == page["rotate"]
    if page.get("crop"):
        assert (p.width, p.height) == (540, 720)


def test_layout_none_config_drops_layout():
    cfg = CONFIG.model_copy(update={"layout": "none"})
    doc = parse(simple_pdf(), cfg, FakeClient()).document
    assert doc.pages[0].layout is None
    assert result_record("text/x", "p", doc)["output"]["layout_pages"] == []
    assert cfg.config_hash() != CONFIG.config_hash()


def test_config_load_and_unknown_fix(tmp_path):
    assert Config.load("configs/standin.yaml") == CONFIG
    bad = tmp_path / "bad.yaml"
    bad.write_text(open("configs/standin.yaml").read() + "fixes:\n  nonsense: true\n")
    with pytest.raises(ValueError, match="unknown fix"):
        Config.load(bad)
    assert set(CONFIG.fixes) == {"layout", "forms", "tables", "charts", "text", "formatting"} and not any(
        CONFIG.fixes.values()
    )


def test_record_is_deterministic():
    pdf = simple_pdf("Same bytes")
    a = result_record("text/a", "p", parse(pdf, CONFIG, FakeClient()).document)
    b = result_record("text/a", "p", parse(pdf, CONFIG, FakeClient()).document)
    assert a == b and a["raw_output"] == {"docparse": "2.0.0.dev0", "config_hash": CONFIG.config_hash()}


def test_ground_layout_asks_for_detections_and_builds_items(tmp_path):
    from docparse.fixes.layout import ocria_code_dir
    from docparse.models.adapters import heron

    if not (ocria_code_dir() / "ocria_grounding" / "layout.py").exists():
        pytest.skip("OCRia-1 grounding code not fetched")
    text = "Revenue grew by twelve percent compared with the same quarter of the previous year."
    pdf = make_pdf([{"texts": [(72, 660, 11, text)]}])
    cfg = CONFIG.model_copy(update={"layout": "ground"})

    cache, manifest = Cache(tmp_path / "c"), ManifestWriter(tmp_path / "m")
    planned = parse(pdf, cfg, CacheOnlyClient(cache, manifest))
    assert planned.document is None and len(planned.pending) == 2  # the page read and the detection
    page_call, det_call = (c for c, _ in read_manifest(tmp_path / "m"))
    assert det_call.model == heron.HERON and det_call.inputs == page_call.inputs
    assert len(list((tmp_path / "m" / "blobs").iterdir())) == 1  # one shared page image

    cache.put(ModelResult(key=page_call.key(), output={"markdown": text}), page_call)
    box = [250.0, 500.0, 2300.0, 600.0]
    cache.put(
        ModelResult(key=det_call.key(), output={"detections": [{"label": "Text", "score": 0.9, "box": box}]}), det_call
    )
    doc = parse(pdf, cfg, ReplayClient(cache)).document
    (item,) = doc.pages[0].layout
    assert item.type == "Text" and item.md == text
    assert item.bbox.x0 == pytest.approx(250 * 72 / 300) and item.bbox.y1 == pytest.approx(600 * 72 / 300)


def test_pages_without_a_text_layer_ask_for_attention_after_the_read(tmp_path):
    import io

    from PIL import Image, ImageDraw

    from docparse.fixes.layout import ocria_code_dir, unit_count

    if not (ocria_code_dir() / "ocria_grounding" / "layout.py").exists():
        pytest.skip("OCRia-1 grounding code not fetched")
    buf = io.BytesIO()
    image = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(image).rectangle((100, 38, 400, 58), fill="black")  # the ink the attention points at
    image.save(buf, format="PNG")
    text = "Scanned notice of the annual meeting"
    cfg = CONFIG.model_copy(update={"layout": "ground", "layout_attention": "no_text_layer"})
    cache, manifest = Cache(tmp_path / "c"), ManifestWriter(tmp_path / "m")
    assert len(parse(buf.getvalue(), cfg, CacheOnlyClient(cache, manifest)).pending) == 2  # read and detection
    page_call, det_call = (c for c, _ in read_manifest(tmp_path / "m"))
    cache.put(ModelResult(key=page_call.key(), output={"markdown": text}), page_call)
    box = [20.0, 20.0, 600.0, 80.0]
    cache.put(
        ModelResult(key=det_call.key(), output={"detections": [{"label": "Text", "score": 0.9, "box": box}]}), det_call
    )
    second = ManifestWriter(tmp_path / "m2")
    planned = parse(buf.getvalue(), cfg, CacheOnlyClient(cache, second))
    assert planned.document is None and len(planned.pending) == 1  # now the attention pass, with the answer
    (att_call,) = (c for c, _ in read_manifest(tmp_path / "m2"))
    assert att_call.prompt_id == "ocria-1/attention" and att_call.inputs[0] == page_call.inputs[0]
    assert (tmp_path / "m2" / "blobs" / att_call.inputs[1]).read_bytes() == text.encode()
    n = unit_count(text)
    out = {"grid": [15, 20], "idx": [[25]] * n, "val": [[1.0]] * n, "has": [True] * n}  # every word at one cell
    cache.put(ModelResult(key=att_call.key(), output=out), att_call)
    doc = parse(buf.getvalue(), cfg, ReplayClient(cache)).document
    assert doc is not None and [i.md for i in doc.pages[0].layout] == [text]
    # a page with a text layer never asks for attention
    pdf = make_pdf([{"texts": [(72, 660, 11, "Revenue grew by twelve percent compared with the previous year.")]}])
    assert len(parse(pdf, cfg, CacheOnlyClient(Cache(tmp_path / "c3"), ManifestWriter(tmp_path / "m3"))).pending) == 2
    # ... unless attention is asked for on every page (it places picture references)
    always = cfg.model_copy(update={"layout_attention": "always"})
    cache3 = Cache(tmp_path / "c4")
    parse(pdf, always, CacheOnlyClient(cache3, ManifestWriter(tmp_path / "m4")))
    for call, _meta in read_manifest(tmp_path / "m4"):
        out = {"detections": []} if call.prompt_id == "heron/detect" else {"markdown": "Revenue grew."}
        cache3.put(ModelResult(key=call.key(), output=out), call)
    again = parse(pdf, always, CacheOnlyClient(cache3, ManifestWriter(tmp_path / "m5")))
    assert len(again.pending) == 1 and [c.prompt_id for c, _ in read_manifest(tmp_path / "m5")] == ["ocria-1/attention"]


def test_kdl_options_ask_for_a_kdl_call_on_the_same_image_and_add_its_regions(tmp_path):
    from docparse.models.adapters import kdl

    text = "Revenue grew by twelve percent compared with the same quarter of the previous year."
    pdf = make_pdf([{"texts": [(72, 660, 11, text)]}])
    cfg = CONFIG.model_copy(update={"layout_options": {"kdl_add": True}})
    cache, manifest = Cache(tmp_path / "c"), ManifestWriter(tmp_path / "m")
    planned = parse(pdf, cfg, CacheOnlyClient(cache, manifest))
    assert planned.document is None and len(planned.pending) == 2  # the page read and the KDL layout call
    page_call, kdl_call = (c for c, _ in read_manifest(tmp_path / "m"))
    assert kdl_call.model == kdl.KDL and kdl_call.inputs == page_call.inputs

    seg = {"x": 400.0, "y": 20.0, "w": 100.0, "h": 30.0, "label": "text"}  # far from the text line
    layout_page = {"page_number": 1, "items": [{"type": "Text", "md": "x", "bbox": seg, "layout_segments": [seg]}]}
    cache.put(ModelResult(key=page_call.key(), output={"markdown": text, "layout_page": layout_page}), page_call)
    answer = "<|box_start|>090 160 900 190<|box_end|><|ref_start|>text<|ref_end|>"  # around the text line
    cache.put(ModelResult(key=kdl_call.key(), output={"text": answer, "width": 2550, "height": 3300}), kdl_call)
    doc = parse(pdf, cfg, ReplayClient(cache)).document
    assert [it.md for it in doc.pages[0].layout] == ["x", text]
    plain = parse(pdf, CONFIG, ReplayClient(cache)).document  # options off: no KDL call, same markdown
    assert write_document(plain) == write_document(doc) and [it.md for it in plain.pages[0].layout] == ["x"]

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

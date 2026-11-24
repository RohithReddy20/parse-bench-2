import json

from docparse.models.cache import Cache
from docparse.pdf.render import pixel_hash, render_pages
from evalkit.standin import import_published, published_call, source_file
from tests.pdfgen import make_pdf, simple_pdf


def _record(example_id, pages, layout_pages=()):
    return {"request": {"example_id": example_id},
            "output": {"markdown": "\n\n".join(pages), "pages": [{"page_index": i, "markdown": m} for i, m in enumerate(pages)],
                       "layout_pages": list(layout_pages)}}  # fmt: skip


def _setup(tmp_path):
    gt, pub = tmp_path / "gt", tmp_path / "pub"
    for sub in ("text", "headings", "layout"):
        (gt / sub).mkdir(parents=True)
    pub.mkdir()
    same = simple_pdf("Shared page")
    (gt / "text" / "a.pdf").write_bytes(same)
    (gt / "headings" / "a_copy.pdf").write_bytes(same)  # same pixels as text/a
    (gt / "text" / "two.pdf").write_bytes(
        make_pdf([{"texts": [(72, 700, 12, "P1")]}, {"texts": [(72, 700, 12, "P2")]}])
    )
    (gt / "layout" / "l.pdf").write_bytes(simple_pdf("Layout page"))
    layout_page = {"page_number": 1, "width": 612, "height": 792, "items": [{"type": "Text", "md": "Layout page"}]}
    (pub / "text.jsonl").write_text("\n".join(json.dumps(r) for r in [
        _record("text/a", ["# from text"]), _record("text/two", ["page one", "page two"]),
        _record("text/gone", ["no source file"])]))  # fmt: skip
    (pub / "headings.jsonl").write_text(json.dumps(_record("headings/a_copy", ["# from headings"])))
    (pub / "layout.jsonl").write_text(json.dumps(_record("layout/l", ["Layout page"], [layout_page])))
    return gt, pub, layout_page


def test_import_report_and_cache_contents(tmp_path):
    gt, pub, layout_page = _setup(tmp_path)
    cache = Cache(tmp_path / "cache")
    report = import_published(pub, gt, cache, workers=1)
    assert report.imported == 4  # text/a, two pages of text/two, layout/l
    assert report.collisions == [("text/a", "headings/a_copy")]
    assert report.missing_source == ["text/gone"]
    assert report.page_mismatch == [] and report.unreadable == []

    def read(pdf_path, page=0):
        image = render_pages(pdf_path.read_bytes())[page].image
        return cache.get(published_call(pixel_hash(image)).key()).output

    assert read(gt / "text" / "a.pdf")["markdown"] == "# from text"  # first write wins on the collision
    assert read(gt / "text" / "two.pdf", 1)["markdown"] == "page two"
    assert read(gt / "layout" / "l.pdf")["layout_page"] == layout_page
    assert read(gt / "text" / "a.pdf")["layout_page"] is None

    again = import_published(pub, gt, cache, workers=1)
    assert again.imported == 0 and again.cached == 4


def test_source_file_finds_images(tmp_path):
    (tmp_path / "text").mkdir()
    (tmp_path / "text" / "scan.jpg").write_bytes(b"x")
    assert source_file(tmp_path, "text", "scan").name == "scan.jpg"
    assert source_file(tmp_path, "text", "nope") is None

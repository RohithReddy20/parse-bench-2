import pytest

from docparse.core.hashing import canonical_json, sha256_hex
from docparse.core.ir import Box, Candidate, Document, LayoutItem, OpaqueContent, Page, Region, TableContent


def test_canonical_json_is_order_independent_and_compact():
    assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": "é"}]}) == '{"a":[1,{"c":"é","d":2}],"b":1}'.encode()
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_sha256_hex():
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_region_chosen_and_missing_selection():
    cand = Candidate(id="c0", reader="ocria.base", raw="x")
    assert Region(id="r", kind="paragraph", order=0, candidates=[cand], selected="c0").chosen() is cand
    with pytest.raises(ValueError):
        Region(id="r", kind="paragraph", order=0, candidates=[cand]).chosen()


def test_content_union_survives_json():
    cands = [
        Candidate(id="a", reader="x", raw="<table></table>", content=TableContent(n_rows=1, n_cols=2)),
        Candidate(id="b", reader="x", raw="<table>", content=OpaqueContent(reason="unclosed table")),
        Candidate(id="c", reader="x", raw="plain"),
    ]
    region = Region(id="r", kind="table", order=0, candidates=cands, selected="a", bbox=Box(x0=0, y0=0, x1=5, y1=2))
    page = Page(index=0, width=612, height=792, regions=[region],
                layout=[LayoutItem(type="Table", label="table", md="t", bbox=Box(x0=0, y0=0, x1=5, y1=2))])  # fmt: skip
    doc = Document(sha256="0" * 64, n_pages=1, pages=[page])
    again = Document.model_validate_json(doc.model_dump_json())
    assert again == doc
    assert isinstance(again.pages[0].regions[0].candidates[1].content, OpaqueContent)
    assert again.pages[0].regions[0].bbox.width == 5

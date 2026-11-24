import json

import pytest

from docparse.base.ocria_parse import parse_page, split_blocks
from docparse.core.ir import Page
from docparse.write.markdown import write_page

CASES = [
    "",
    " \n\n\t",
    "a",
    "# H\n\ntext\n",
    "x\n\n\n\ny",
    "<table><tr><td>a\n\nb</td></tr></table>\n\ntail",
    "<table><tr><td>never closed",
    "<table><tr><td>never closed\n\n# not a heading any more\n\n",
    "```\ncode\n\nmore",
    "```python\nx = 1\n\n# not a heading\n```\n\nafter",
    "\n\n lead",
    "* a\n* b\n\n1. c",
    "intro\n<table><tr><td>1</td></tr></table>\noutro",
    "# H1\ntext right under\n## H2\n\nmore",
    "<TABLE><tr><td><table><tr><td>in</td></tr></table></td></tr></TABLE>tail",
    "stray </table> close\n\nand <b>inline</b>",
    "a\r\n\r\nb",
    "trailing spaces   \n\n\n",
    "~~struck~~ and <u>under</u>\n\n- [x] yes\n- [ ] no\n\nName: [___]",
]


@pytest.mark.parametrize("md", CASES)
def test_round_trip_is_byte_exact(md):
    assert write_page(parse_page(md)) == md


@pytest.mark.parametrize("md", CASES)
def test_blocks_never_empty_and_never_padded(md):
    _leading, blocks = split_blocks(md)
    for b in blocks:
        assert b.raw and b.raw == b.raw.strip() or b.kind == "opaque"
        assert b.sep_after.strip() == ""


def test_kinds():
    md = "# H\n\np\n\n<table><tr><th>a</th></tr></table>\n\n* x\n\n```\nc\n```\n\n![logo: Acme]"
    assert [r.kind for r in parse_page(md).regions] == ["heading", "paragraph", "table", "list", "code", "image_ref"]


def test_heading_line_is_its_own_block():
    regions = parse_page("# H1\ntext right under\n## H2\n\nmore").regions
    assert [(r.kind, r.chosen().raw) for r in regions] == [
        ("heading", "# H1"),
        ("paragraph", "text right under"),
        ("heading", "## H2"),
        ("paragraph", "more"),
    ]
    assert regions[0].chosen().content.heading_level == 1
    assert regions[2].chosen().content.heading_level == 2


def test_table_between_text_is_split_out():
    regions = parse_page("intro\n<table><tr><td>1</td></tr></table>\noutro").regions
    assert [r.kind for r in regions] == ["paragraph", "table", "paragraph"]
    assert regions[1].chosen().raw == "<table><tr><td>1</td></tr></table>"


def test_unclosed_table_is_opaque_to_the_end():
    regions = parse_page("before\n\n<table><tr><td>never closed\n\n# H\n\n").regions
    assert [r.kind for r in regions] == ["paragraph", "opaque"]
    assert regions[1].chosen().content.reason == "unclosed table"
    assert regions[1].chosen().raw.endswith("# H")


def test_unclosed_fence_is_opaque():
    (region,) = parse_page("```\ncode\n\nmore").regions
    assert region.kind == "opaque" and region.chosen().content.reason == "unclosed code fence"


def test_nested_table_is_one_block():
    regions = parse_page("<table><tr><td><table><tr><td>in</td></tr></table></td></tr></table>tail").regions
    assert [r.kind for r in regions] == ["table", "paragraph"]


def test_blank_lines_inside_a_table_do_not_split_it():
    (table, tail) = parse_page("<table><tr><td>a\n\nb</td></tr></table>\n\ntail").regions
    assert table.kind == "table" and "\n\n" in table.chosen().raw


def test_heading_inside_code_fence_is_not_a_heading():
    regions = parse_page("```python\nx = 1\n\n# not a heading\n```\n\nafter").regions
    assert [r.kind for r in regions] == ["code", "paragraph"]


def test_ids_order_and_provenance():
    page = parse_page("a\n\nb", index=3, call_key="k")
    assert [r.id for r in page.regions] == ["p3r0", "p3r1"]
    assert [r.order for r in page.regions] == [0, 1]
    assert page.regions[0].chosen().reader == "ocria.base"
    assert page.regions[0].chosen().prov[0].ref == "k"


def test_page_json_round_trip():
    page = parse_page("# H\n\np\n\n<table><tr><td>1</td></tr></table>\n\n<table><tr><td>open")
    assert Page.model_validate_json(page.model_dump_json()) == page
    assert write_page(Page.model_validate(json.loads(page.model_dump_json()))) == write_page(page)


@pytest.mark.data
def test_every_published_page_round_trips(published_markdown):
    assert len(published_markdown) == 4317
    kinds: dict[str, int] = {}
    for example_id, md in published_markdown.items():
        page = parse_page(md)
        assert write_page(page) == md, example_id
        for region in page.regions:
            kinds[region.kind] = kinds.get(region.kind, 0) + 1
    assert kinds["table"] > 3000 and kinds["heading"] > 5000 and kinds["paragraph"] > 20000
    print(kinds)

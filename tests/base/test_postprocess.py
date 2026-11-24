import pytest

from docparse.base.postprocess import ocria_postprocess


def test_strips_wrapper_tags_and_comments():
    raw = "<!-- analysis -->\n```markdown\n<page_header>ACME <page_number>3</page_number></page_header>\n\n# T\n\nbody"
    out = ocria_postprocess(raw)
    assert "<!--" not in out and "```" not in out and "page_header" not in out
    assert "ACME" in out and "3" in out and "# T" in out and "body" in out


def test_html_emphasis_becomes_markdown_outside_tables():
    out = ocria_postprocess("<b>bold</b> <i>it</i> <s>gone</s>\n\n<table><tr><td><b>cell</b></td></tr></table>")
    assert "**bold**" in out and "*it*" in out and "~~gone~~" in out
    assert "<td><b>cell</b></td>" in out


def test_markdown_table_becomes_html():
    out = ocria_postprocess("| a | b |\n|---|---|\n| 1 | 2 |")
    assert "<table>" in out and "<th>a</th>" in out and "<td>2</td>" in out


def test_image_refs_and_mermaid_are_removed():
    out = ocria_postprocess("before\n\n![logo: Acme]\n\n```mermaid\ngraph TD; A-->B\n```\n\nafter")
    assert "![" not in out and "mermaid" not in out and "before" in out and "after" in out


@pytest.mark.parametrize("raw", ["", None, "<!-- only a comment", "```markdown\n", "<table><tr><td>cut"])
def test_degenerate_answers_do_not_raise(raw):
    assert isinstance(ocria_postprocess(raw), str)


def _synthetic_raw(md: str, i: int) -> str:
    """A raw answer in OCRia-1's shape built around published markdown, exercising every post-processing step."""
    parts = ["<!-- thinking about the page -->"] if i % 3 == 0 else []
    parts.append("```markdown")
    if i % 2 == 0:
        parts.append(f"<page_header>Header {i} <page_number>{i}</page_number></page_header>")
    parts.append(md.replace("**", "<b>", 1).replace("**", "</b>", 1) if i % 5 == 0 else md)
    if i % 4 == 0:
        parts.append("![photo: a caption]\n\n| k | v |\n|---|---|\n| a | 1 |")
    if i % 7 == 0:
        parts.append("<watermark>DRAFT</watermark>\n\n<hr/>\n\nx&nbsp; &nbsp; y . . . . . . . . z")
    if i % 2 == 1:
        parts.append(f"<page_footer>Footer {i}</page_footer>")
    return "\n\n".join(parts) + "\n```"


@pytest.mark.data
def test_port_matches_the_pinned_harness(published_markdown):
    from parse_bench.inference.providers.parse.ocria_1 import ocria_postprocess as harness

    ids = sorted(published_markdown)[::14][:320]
    assert len(ids) >= 300
    for i, example_id in enumerate(ids):
        raw = _synthetic_raw(published_markdown[example_id], i)
        assert ocria_postprocess(raw) == harness(raw), example_id

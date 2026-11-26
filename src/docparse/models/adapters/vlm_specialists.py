"""Specialist VLM reads served by serve/modal_vlm.py: build the keyed call for an image, and read its answer.

One request shape for all three: input blob 0 is the image (keyed by its pixels), blob 1 the UTF-8 user prompt.
Revisions are pinned to exact Hugging Face commit SHAs so upstream weight changes never answer under an old key.
"""

from __future__ import annotations

import html
import re
from collections import ChainMap
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from docparse.core.hashing import sha256_hex
from docparse.models.adapters.ocria import LazyBlobs
from docparse.models.calls import ModelCall, ModelRef, ModelResult
from docparse.pdf.render import pixel_hash

RUNTIME = "vllm-0.34.0"  # serve/modal_vlm.py's default DOCPARSE_VLLM_VERSION
QWEN3_VL = ModelRef(
    repo="Qwen/Qwen3-VL-8B-Instruct",
    revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    dtype="bf16",
)
OLMOCR2 = ModelRef(
    repo="allenai/olmOCR-2-7B-1025",
    revision="7c6fa1611328bd17dbf5f3a07509d2b6fa2eba4e",
    dtype="bf16",
)
CHANDRA2 = ModelRef(
    repo="datalab-to/chandra-ocr-2",
    revision="af93b47dba1b47b6640c86ccf487ed2260ab9a09",
    dtype="bf16",
)

CHART_PROMPT_ID = "qwen3-vl/chart-vector"
CHART_PARAMS: dict[str, Any] = {"temperature": 0.0, "max_tokens": 4096}
CHART_PROMPT = """\
This image is one chart cropped from a document page. Measurements taken from the PDF's vector drawing of the \
same chart follow; wherever they give a value, trust them over the pixels.
<measurements>
{hint}
</measurements>

Answer in exactly this form and nothing else:
Title: <chart title exactly as printed, or none>
Axes: x = <label and unit>; y = <label and unit>, or none
Legend: <series names in legend order, or none>
<table>...</table>

The table: first column the categories or x values in axis order (slices for a pie chart), then one column per \
series headed by its legend name.
- Use every value the measurements give. Read the image only for values they lack, at the precision of the axis \
ticks or data labels.
- Copy labels, series names and units exactly as printed. Head a single unnamed series with the y-axis label, \
never "Series 1", "Value" or another placeholder.
- Add no Total, Average, Sum or other row or column that the chart does not draw.
- Plain HTML (<table>, <tr>, <th>, <td>): no markdown, code fences or commentary."""

OLMOCR_PROMPT_ID = "olmocr-2/no-anchoring-v4-yaml"
OLMOCR_PARAMS: dict[str, Any] = {"temperature": 0.0, "max_tokens": 8000, "text_first": True}
OLMOCR_LONG_SIDE = 1288  # olmOCR-2's official pipeline target longest side
# Verified verbatim against olmocr==0.4.27 `olmocr.prompts.build_no_anchoring_v4_yaml_prompt()` ("LateX" sic)
OLMOCR_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. "
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the "
    "primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)

CHANDRA_PROMPT_ID = "chandra-ocr-2/ocr-layout"
CHANDRA_PARAMS: dict[str, Any] = {"temperature": 0.0, "max_tokens": 8192}
CHANDRA_MAX_SIDE = 2048
_CHANDRA_TAGS = (
    "['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', 'td', 'p', 'th', 'div', 'pre', "
    "'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', 'input', 'a', 'span', 'img', 'hr', 'tbody', "
    "'small', 'caption', 'strong', 'thead', 'big', 'code', 'chem']"
)
_CHANDRA_ATTRS = (
    "['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', 'value', 'style', "
    "'href', 'alt', 'align', 'data-bbox', 'data-label']"
)
# Verified verbatim against datalab-to/chandra `chandra/prompts.py` (`OCR_LAYOUT_PROMPT`)
CHANDRA_PROMPT = (
    "OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the "
    "data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are "
    "normalized 0-1000. The data-label attribute is the label for the block.\n\n"
    "Use the following labels:\n"
    "- Caption\n- Footnote\n- Equation-Block\n- List-Group\n- Page-Header\n- Page-Footer\n- Image\n"
    "- Section-Header\n- Table\n- Text\n- Complex-Block\n- Code-Block\n- Form\n- Table-Of-Contents\n"
    "- Figure\n- Chemical-Block\n- Diagram\n- Bibliography\n- Blank-Page\n\n"
    f"Only use these tags {_CHANDRA_TAGS}, and these attributes {_CHANDRA_ATTRS}.\n\n"
    "Guidelines:\n"
    "* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in "
    "KaTeX-compatible LaTeX. Use display for block math.\n"
    "* Tables: Use colspan and rowspan attributes to match table structure.\n"
    "* Formatting: Maintain consistent formatting with the image, including spacing, indentation, "
    "subscripts/superscripts, and special characters.\n"
    "* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out "
    "the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, "
    "and convert diagrams to mermaid.\n"
    "* Forms: Mark checkboxes and radio buttons properly.\n"
    "* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line "
    "breaks within paragraphs, but only when absolutely necessary to maintain meaning.\n"
    "* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.\n"
    "* Lists: Preserve indents and proper list markers.\n"
    "* Use the simplest possible HTML structure that accurately represents the content of the block.\n"
    "* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should "
    "be correct and natural."
)

CHART_VERIFY_PROMPT_ID = "qwen3-vl/chart-verify-v1"
CHART_VERIFY_PARAMS: dict[str, Any] = {"temperature": 0.0, "max_tokens": 6144}
CHART_VERIFY_PROMPT = """\
You are verifying and correcting data table(s) extracted from chart(s) on this document page.
You are given:
1. The page image showing the complete chart(s) (including all panels, subgroups, legends, titles, and axes).
2. Draft HTML table(s) extracted from this page:
<draft_tables>
{draft}
</draft_tables>
3. Exact vector measurements extracted directly from the PDF's drawing paths (trust these over eyeballing pixels \
wherever they match a plotted series):
<measurements>
{hint}
</measurements>

Instructions:
- Output the corrected HTML table(s) for ALL charts and ALL panels on this page.
- CRITICAL: Preserve EVERY panel, subgroup (e.g. Women/Men, Panel A/B, scenarios), category, and series shown \
in the chart and draft table. NEVER drop a panel or shorten the category list.
- Wherever <measurements> gives exact vector values for a plotted series, replace eyeballed estimates in the draft \
with the exact vector values (preserving units/signs).
- Fix any swapped legend series names, missing units, or wrong numbers using the image and legend hex colors.
- If <draft_tables> is "(none)", extract the full chart data table(s) from scratch using the image and measurements.
- Do NOT invent Total, Average, or Sum rows that are not plotted as bars/points in the chart.
- Return ONLY the chart title line (exact as printed) followed by clean HTML <table>...</table> for each chart. \
No markdown code fences or commentary."""


def _call(image, prompt_id: str, template: str, user: str, model: ModelRef, runtime: str, params: dict, extra):
    text, name = user.encode("utf-8"), pixel_hash(image)
    call = ModelCall(
        model=model,
        runtime=runtime,
        prompt_id=prompt_id,
        prompt_sha=sha256_hex(template.encode("utf-8")),
        params={**params, **(extra or {})},
        inputs=[name, sha256_hex(text)],
    )
    return call, ChainMap(LazyBlobs(image, name), {call.inputs[1]: text})


def build_chart_call(image, hint: str, model: ModelRef = QWEN3_VL, runtime: str = RUNTIME, extra: dict | None = None):
    """Vector-hinted read of a chart crop; `hint`, the formatted measurements, enters the key through blob 1."""
    user = CHART_PROMPT.format(hint=hint.strip() or "(none)")
    return _call(image, CHART_PROMPT_ID, CHART_PROMPT, user, model, runtime, CHART_PARAMS, extra)


def build_olmocr_call(image, model: ModelRef = OLMOCR2, runtime: str = RUNTIME, extra: dict | None = None):
    """olmOCR-2 page read, fitted to OLMOCR_LONG_SIDE (`call.inputs[0]` keys the fitted image)."""
    scale = OLMOCR_LONG_SIDE / max(image.width, image.height)
    if abs(scale - 1.0) > 1e-3:
        from PIL import Image

        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    return _call(image, OLMOCR_PROMPT_ID, OLMOCR_PROMPT, OLMOCR_PROMPT, model, runtime, OLMOCR_PARAMS, extra)


def build_chandra_call(image, model: ModelRef = CHANDRA2, runtime: str = RUNTIME, extra: dict | None = None):
    """Chandra-2 layout-HTML read, fitting huge scans cleanly to CHANDRA_MAX_SIDE."""
    if max(image.width, image.height) > CHANDRA_MAX_SIDE:
        from PIL import Image

        s = CHANDRA_MAX_SIDE / max(image.width, image.height)
        image = image.resize((max(1, round(image.width * s)), max(1, round(image.height * s))), Image.LANCZOS)
    return _call(image, CHANDRA_PROMPT_ID, CHANDRA_PROMPT, CHANDRA_PROMPT, model, runtime, CHANDRA_PARAMS, extra)


def build_chart_verify_call(
    image,
    draft_tables: str,
    hint: str,
    model: ModelRef = QWEN3_VL,
    runtime: str = RUNTIME,
    extra: dict | None = None,
):
    """Full-page multi-panel vector-guided chart verifier call (preserves all panels of draft tables)."""
    if max(image.width, image.height) > 1800:
        from PIL import Image

        s = 1800 / max(image.width, image.height)
        image = image.resize((max(1, round(image.width * s)), max(1, round(image.height * s))), Image.LANCZOS)
    user = CHART_VERIFY_PROMPT.format(draft=draft_tables.strip() or "(none)", hint=hint.strip() or "(none)")
    return _call(image, CHART_VERIFY_PROMPT_ID, CHART_VERIFY_PROMPT, user, model, runtime, CHART_VERIFY_PARAMS, extra)


@dataclass
class SpecialistRead:
    markdown: str
    raw_answer: str
    finish_reason: str | None = None
    front_matter: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartRead:
    markdown: str
    raw_answer: str
    finish_reason: str | None
    title: str | None
    axes: str | None
    legend: str | None
    table_html: str | None


_FLAGS = re.DOTALL | re.IGNORECASE
_YAML = {"true": True, "false": False, "": None, "null": None, "none": None, "~": None}
_FRONT_MATTER = re.compile(r"\A\s*---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_TABLE = re.compile(r"<table\b.*?</table>", _FLAGS)
_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", _FLAGS)
_CELL = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", _FLAGS)
_FIELD = re.compile(r"^[ \t*#]*(title|axes|legend)[ \t*]*:[ \t*]*(.*?)[ \t]*$", re.IGNORECASE | re.MULTILINE)
_INVENTED = re.compile(r"(?:grand )?total|sum|average|avg\.?|mean", re.IGNORECASE)
_BR_TAG = r"<br(?=[\s/>])[^>]*>"
_HEADING_BREAKS = re.compile(rf"[ \t]*(?:(?:{_BR_TAG}|[\r\n]|&(?:NewLine|#0*(?:10|13)|#x0*[ad]);)[ \t]*)+", _FLAGS)
_LIST_TAG = re.compile(r"<(/?)(ol|ul|li)(?=[\s>])([^>]*)>\s*", _FLAGS)
_LIST_BULLET = re.compile(r"[•▪‣◦●]\s*")
_LIST_ITEM_BREAK = re.compile(rf"{_BR_TAG}|</?p\b[^>]*>", _FLAGS)
_LIST_CODE = re.compile(r"<code\b[^>]*>(.*?)</code>", _FLAGS)
_LIST_LINK = re.compile(r"""<a\b[^>]*?\shref\s*=\s*(["'])(.*?)\1[^>]*>(.*?)</a>""", _FLAGS)
_IMG_ALT_TAG = r"""<img\b[^>]*?\salt\s*=\s*(["'])(.*?)\1[^>]*>"""


def _single_line(text: str) -> str:
    return _HEADING_BREAKS.sub(" ", text).strip()


def _list_inline_code(match: re.Match[str]) -> str:
    code = match[1]
    delimiter = "`" * (1 + max(map(len, re.findall(r"`+", code)), default=0))
    if code.startswith((" ", "`")) or code.endswith((" ", "`")):
        code = f" {code} "
    return f"{delimiter}{code}{delimiter}"


def _lists_to_markdown(text: str) -> str:
    """Convert HTML `<ul>`/`<ol>` lists to Markdown bullets.

    Chandra splits long or unnumbered lists into separate `<ol>` fragments starting at 1, so emitting `- `
    preserves nested structure without inventing phantom `1. 2. 3.` numbers that contradict the page.
    """
    parts: list[str] = []
    indents = [0]
    pos = 0
    for match in _LIST_TAG.finditer(text):
        body = text[pos : match.start()].rstrip()
        item_indent = indents[-1]
        if item_indent:
            body = _LIST_ITEM_BREAK.sub("\n", body).strip()
            body = _LIST_CODE.sub(_list_inline_code, body)
            body = _LIST_LINK.sub(r"[\3](\2)", body)
        parts.append(body)
        pos = match.end()
        closing, tag, _attrs = match.groups()
        tag = tag.lower()
        if tag == "li":
            if closing:
                indents[-1] = 0
                continue
            indent = indents[-2] if len(indents) > 1 else 0
            indents[-1] = indent + 2
            parts.append("\n" + " " * indent + "- ")
            bullet = _LIST_BULLET.match(text, pos)
            if bullet:
                pos = bullet.end()
        elif not closing:
            indents.append(0)
            if len(indents) == 2:
                parts.append("\n\n")
        elif len(indents) > 1:
            indents.pop()
            if len(indents) == 1:
                parts.append("\n\n")
            elif pos < len(text) and not _LIST_TAG.match(text, pos):
                parts.append("\n")
    parts.append(text[pos:])
    return "".join(parts)


_HTML_TO_MD = [
    (re.compile(pattern, _FLAGS), repl)
    for pattern, repl in [
        (r'<math\b[^>]*display="block"[^>]*>(.*?)</math>', r"\n\n$$\1$$\n\n"),
        (r"<math\b[^>]*>(.*?)</math>", r"$\1$"),
        (r"<(b|strong)\b[^>]*>(.*?)</\1>", r"**\2**"),
        (r"<(i|em)\b[^>]*>(.*?)</\1>", r"*\2*"),
        (r"<(del|s)\b[^>]*>(.*?)</\1>", r"~~\2~~"),
        (_IMG_ALT_TAG, r"![\2]()"),
        (r"<input\b[^>]*\bchecked\b[^>]*>", "[x]"),
        (r"<input\b[^>]*>", "[ ]"),
        (_BR_TAG, "\n"),
        (r"</?(?:p|div|pre|hr)\b[^>]*>", "\n\n"),
        (r"<h([1-6])\b[^>]*>(.*?)</h\1>", lambda m: f"\n\n{'#' * int(m[1])} {_single_line(m[2])}\n\n"),
        # Preserve `<u>` for underline fidelity; strip wrapper and sub/sup tags so prose tokens stay contiguous
        (r"<(?!/?u\b)[^>]+>", ""),
    ]
]


def _scalar(value: str) -> Any:
    v = value.strip().strip("'\"")
    return _YAML.get(v.lower(), int(v) if re.fullmatch(r"-?\d+", v) else v)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """olmOCR-2's answer as (front matter fields, page text); an answer without front matter comes back whole."""
    if not (m := _FRONT_MATTER.match(text)):
        return {}, text.strip()
    lines = (line.partition(":") for line in m[1].splitlines())
    return {k.strip(): _scalar(v) for k, sep, v in lines if sep}, text[m.end() :].strip()


def read_olmocr_result(result: ModelResult) -> SpecialistRead:
    """When `front_matter['is_rotation_valid']` is False, rotate by `rotation_correction` and read again."""
    raw = result.output.get("answer") or ""
    fields, body = split_front_matter(raw)
    return SpecialistRead(body, raw, result.output.get("finish_reason"), fields)


_FURNITURE_DIV = re.compile(
    r'<div\b[^>]*data-label="(?:Page-Header|Page-Footer)"[^>]*>.*?</div>',
    _FLAGS,
)
_IMAGE_DIV = re.compile(r'<div\b[^>]*data-label="Image"[^>]*>(.*?)</div>', _FLAGS)
_SEC_HDR_DIV = re.compile(r'<div\b[^>]*data-label="Section-Header"[^>]*>(.*?)</div>', _FLAGS)
_IMG_TAG = re.compile(r"<img\b[^>]*>", _FLAGS)
_HEADING_OR_BLOCK = re.compile(r"<(?:h[1-6]|p|div|table|ul|ol|li|img|pre|hr|blockquote)\b", _FLAGS)
_DIV_TAG = re.compile(r"</?div\b[^>]*>", _FLAGS)
_FIGURE_LABEL = re.compile(r"""\sdata-label\s*=\s*(["'])Figure\1""", _FLAGS)
_IMG_ALT_TEXT = re.compile(_IMG_ALT_TAG, _FLAGS)
_TABLE_HEADER = re.compile(r"<th\b", re.I)
_CHART_ALT = re.compile(r"\b(?:chart|graph|plot|bar|line|pie|histogram|scatter|series|axis|axes)\b", re.I)


def _promote_section_headers(raw_html: str) -> str:
    """Honor Chandra's `<div data-label="Section-Header">` when it omitted `<h1..h6>` tags."""

    def _repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        body = re.sub(r"^<p\b[^>]*>(.*?)</p>$", r"\1", inner, flags=_FLAGS).strip()
        if _HEADING_OR_BLOCK.search(body) or _HEADING_BREAKS.search(body):
            return m.group(0)
        visible = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
        if not visible:
            return m.group(0)
        return f'<div data-label="Section-Header"><h2>{body}</h2></div>'

    return _SEC_HDR_DIV.sub(_repl, raw_html)


def _trim_image_div_essays(raw_html: str) -> str:
    """Inside `<div data-label="Image">` blocks that contain `<img>` and no `<table>`, keep the `<img>` tags
    (preserving `![alt]()`) and drop Chandra's verbose visual description essays of logos/photos/QR codes."""

    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        if "<table" in inner.lower():
            return m.group(0)
        imgs = _IMG_TAG.findall(inner)
        if not imgs:
            return m.group(0)
        return '<div data-label="Image">' + "\n".join(imgs) + "</div>"

    return _IMAGE_DIV.sub(_repl, raw_html)


_CHANDRA_DIV_BLOCK = re.compile(r"<div\b([^>]*)>(.*?)</div>", _FLAGS)
_BBOX_ATTR = re.compile(r'\bdata-bbox="(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"', _FLAGS)
_LABEL_ATTR = re.compile(r'\bdata-label="([^"]+)"', _FLAGS)
_CHANDRA_TO_DOCLING = {
    "Table": "Table",
    "Figure": "Picture",
    "Image": "Picture",
    "Diagram": "Picture",
    "Section-Header": "Section-header",
    "Page-Header": "Page-header",
    "Page-Footer": "Page-footer",
    "Caption": "Caption",
    "Footnote": "Footnote",
    "List-Group": "List-item",
    "Code-Block": "Code",
}


def chandra_layout_boxes(
    raw_chandra: str, width_pt: float, height_pt: float
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Extract `(docling_label, (x0, y0, x1, y1))` in PDF points from Chandra's normalized 0-1000 `data-bbox` divs."""
    return [(lbl, box) for lbl, box, _ in chandra_layout_blocks(raw_chandra, width_pt, height_pt)]


def chandra_layout_blocks(
    raw_chandra: str, width_pt: float, height_pt: float
) -> list[tuple[str, tuple[float, float, float, float], str]]:
    """Extract `(docling_label, (x0, y0, x1, y1), markdown)` in PDF points from Chandra's `data-bbox` divs."""
    out: list[tuple[str, tuple[float, float, float, float], str]] = []
    if not raw_chandra or width_pt <= 0 or height_pt <= 0:
        return out
    for m in _CHANDRA_DIV_BLOCK.finditer(raw_chandra):
        attrs = m.group(1)
        mb = _BBOX_ATTR.search(attrs)
        ml = _LABEL_ATTR.search(attrs)
        if not mb or not ml:
            continue
        raw_label = ml.group(1).strip()
        if raw_label == "Blank-Page":
            continue
        x0, y0, x1, y1 = (int(mb.group(i)) for i in range(1, 5))
        if x1 <= x0 or y1 <= y0:
            continue
        box = (
            x0 * width_pt / 1000.0,
            y0 * height_pt / 1000.0,
            x1 * width_pt / 1000.0,
            y1 * height_pt / 1000.0,
        )
        lbl = _CHANDRA_TO_DOCLING.get(raw_label, "Text")
        md = "" if lbl == "Picture" else chandra_markdown(m.group(0))
        out.append((lbl, box, md))
    return out


@dataclass(frozen=True)
class _Figure:
    body: str
    body_start: int
    body_end: int
    alt: str


def _figure_divs(text: str) -> Iterator[_Figure]:
    depth = 0
    for tag in _DIV_TAG.finditer(text):
        if not depth:
            if _FIGURE_LABEL.search(tag[0]) and not tag[0].startswith("</"):
                body_start = tag.end()
                depth = 1
        elif tag[0].startswith("</"):
            depth -= 1
            if not depth:
                body = text[body_start : tag.start()]
                alt = " ".join(match[2] for match in _IMG_ALT_TEXT.finditer(body))
                yield _Figure(body, body_start, tag.start(), alt)
        else:
            depth += 1


def has_chart_figure(raw_chandra: str) -> bool:
    """True when Chandra tagged at least one `Figure` block whose alt text describes a data chart/plot."""
    return any(_CHART_ALT.search(figure.alt) for figure in _figure_divs(raw_chandra))


def _html_tokens(raw_html: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", re.sub(r"<[^>]+>", " ", raw_html).lower()))


def _match_chart_tables(
    figures: list[_Figure], chart_tables: list[tuple[str, set[str]]], protected: set[int], empty: set[int]
) -> dict[int, list[str]]:
    if len(figures) == 1:
        return {0: [table for table, _ in chart_tables]} if chart_tables and not protected else {}

    figure_tokens = [_html_tokens(f"{figure.body} {figure.alt}") for figure in figures]
    assignments: dict[int, list[str]] = {}
    unmatched = []
    for table, tokens in chart_tables:
        scores = [len(tokens & known) for known in figure_tokens]
        best = max(scores)
        if not best or scores.count(best) != 1:
            unmatched.append(table)
            continue
        index = scores.index(best)
        if index not in protected:
            assignments.setdefault(index, []).append(table)

    if len(chart_tables) == len(figures) and not protected:
        vacant = [index for index in range(len(figures)) if index not in assignments]
        if len(vacant) == len(unmatched) and all(index in empty for index in vacant):
            assignments.update((index, [table]) for index, table in zip(vacant, unmatched, strict=True))
    return assignments


def fuse_figure_charts(
    raw_chandra: str,
    ocria_tables: list[str],
    *,
    ocria_vector_measured: bool = False,
    drop_furniture: bool = False,
) -> tuple[str, bool]:
    """Fuse vector-measured chart tables from a secondary reader into Chandra's chart `Figure` blocks.

    Keeps panels together in their chart figure and protects printed insets from unmeasured replacements.
    """
    figures = list(_figure_divs(raw_chandra))
    fig_matches = [figure for figure in figures if _CHART_ALT.search(figure.alt)]
    original_tables = {figure: _TABLE.findall(figure.body) for figure in figures}
    insets = {figure: [t for t in tables if not _TABLE_HEADER.search(t)] for figure, tables in original_tables.items()}
    if not fig_matches or not ocria_tables:
        preserved_tables = frozenset(t for tables in original_tables.values() for t in tables)
        return chandra_markdown(raw_chandra, drop_furniture, _preserve_tables=preserved_tables), False

    nonfig_toks = [
        _html_tokens(table[0])
        for table in _TABLE.finditer(raw_chandra)
        if not any(figure.body_start <= table.start() < figure.body_end for figure in figures)
    ]
    chart_tbls: list[tuple[str, set[str]]] = []
    for tbl in ocria_tables:
        otoks = _html_tokens(tbl)
        if not otoks and not html.unescape(re.sub(r"<[^>]+>", "", tbl)).strip():
            continue
        if any(len(otoks & rt) / max(1, min(len(otoks), len(rt))) >= 0.35 for rt in nonfig_toks if rt):
            continue
        chart_tbls.append((tbl, otoks))

    protected = {
        i
        for i, figure in enumerate(fig_matches)
        if not ocria_vector_measured and insets[figure] and insets[figure] == original_tables[figure]
    }
    empty = {i for i, figure in enumerate(fig_matches) if not original_tables[figure]}
    replacements = _match_chart_tables(fig_matches, chart_tbls, protected, empty)
    replaced_figures = {fig_matches[i] for i in replacements}
    preserved_tables = {t for figure in figures if figure not in replaced_figures for t in original_tables[figure]}
    replaced = raw_chandra
    for figure_idx in sorted(replacements, reverse=True):
        figure = fig_matches[figure_idx]
        keep_insets = set(insets[figure]) if insets[figure] != original_tables[figure] else set()
        preserved_tables.update(keep_insets)
        body = _TABLE.sub(lambda table, keep=keep_insets: table[0] if table[0] in keep else "", figure.body)
        panels = "\n".join(replacements[figure_idx])
        replaced = replaced[: figure.body_start] + body + "\n" + panels + replaced[figure.body_end :]
    return chandra_markdown(replaced, drop_furniture, _preserve_tables=frozenset(preserved_tables)), bool(replacements)


_MATH_TAG = re.compile(r"<math\b[^>]*>(.*?)</math>", _FLAGS)
_LATEX_CMD_WRAP = re.compile(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}")
_LATEX_SYMBOLS = [
    (r"\pm", "±"),
    (r"\mp", "∓"),
    (r"\times", "×"),
    (r"\cdot", "·"),
    (r"\div", "÷"),
    (r"\leq", "≤"),
    (r"\le", "≤"),
    (r"\geq", "≥"),
    (r"\ge", "≥"),
    (r"\neq", "≠"),
    (r"\ne", "≠"),
    (r"\approx", "≈"),
    (r"\sim", "~"),
    (r"\infty", "∞"),
    (r"\degree", "°"),
    (r"^\circ", "°"),
    (r"^{\circ}", "°"),
    (r"\circ", "°"),
    (r"\Delta", "Δ"),
    (r"\delta", "δ"),
    (r"\alpha", "α"),
    (r"\beta", "β"),
    (r"\gamma", "γ"),
    (r"\Gamma", "Γ"),
    (r"\mu", "μ"),
    (r"\sigma", "σ"),
    (r"\Sigma", "Σ"),
    (r"\pi", "π"),
    (r"\theta", "θ"),
    (r"\lambda", "λ"),
    (r"\rho", "ρ"),
    (r"\chi", "χ"),
    (r"\omega", "ω"),
    (r"\Omega", "Ω"),
    (r"\phi", "φ"),
    (r"\psi", "ψ"),
    (r"\eta", "η"),
    (r"\tau", "τ"),
    (r"\%", "%"),
    (r"\$", "$"),
    (r"\&", "&"),
    (r"\_", "_"),
]
_LATEX_SYM_RE = re.compile(
    "|".join(
        re.escape(k) + (r"(?![A-Za-z])" if k[-1].isalpha() else "")
        for k, _ in sorted(_LATEX_SYMBOLS, key=lambda kv: -len(kv[0]))
    )
)
_LATEX_SYM_MAP = dict(_LATEX_SYMBOLS)


def _clean_table_math(table_html: str) -> str:
    """Normalize Chandra's `<math>`, `<input>`, and `<del>` tags inside HTML tables into clean HTML/markdown."""
    out = re.sub(r"<input\b[^>]*\bchecked\b[^>]*>", "[x]", table_html, flags=_FLAGS)
    out = re.sub(r"<input\b[^>]*>", "[ ]", out, flags=_FLAGS)
    out = re.sub(r"<(del|s)\b[^>]*>(.*?)</\1>", r"~~\2~~", out, flags=_FLAGS)
    if "<math" not in out.lower():
        return out

    def _repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        for _ in range(3):
            nxt = _LATEX_CMD_WRAP.sub(r"\1", inner)
            if nxt == inner:
                break
            inner = nxt
        inner = _LATEX_SYM_RE.sub(lambda sm: _LATEX_SYM_MAP[sm.group(0)], inner)
        # Clean trivial spacing commands (`\,`, `\;`, `\ `) and simple `^{x}` / `_{x}` / `_x` / `^x` inside tables
        inner = re.sub(r"\\[,;:! ]", " ", inner)
        inner = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", inner)
        inner = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", inner)
        inner = re.sub(r"(?<=[A-Za-z0-9)])\^([A-Za-z0-9+\-])(?![A-Za-z0-9])", r"<sup>\1</sup>", inner)
        inner = re.sub(r"(?<=[A-Za-z0-9)])_([A-Za-z0-9])(?![A-Za-z0-9])", r"<sub>\1</sub>", inner)
        # Close TeX command-terminator space in SI micro-units (e.g. `\mu M` -> `μ M` -> `μM`)
        inner = re.sub(r"μ\s+(mol|[mMgLlLsAVWFHSΩ])\b", r"μ\1", inner)
        # If no complex LaTeX commands remain, return clean inline HTML; otherwise wrap as $...$
        if "\\" not in inner and "{" not in inner and "}" not in inner:
            return re.sub(r"\s+", " ", inner).strip()
        return f"${m.group(1).strip()}$"

    return _MATH_TAG.sub(_repl, out)


def _tidy_markdown_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text))


def chandra_markdown(text: str, drop_furniture: bool = False, *, _preserve_tables: frozenset[str] = frozenset()) -> str:
    """Chandra's layout-HTML answer as markdown; tables stay valid HTML with `<math>` normalized."""
    fragments: list[str] = []

    def stash(fragment: str) -> str:
        fragments.append(fragment)
        return f"\n\n\x00{len(fragments) - 1}\x00\n\n"

    def stash_table(match: re.Match[str]) -> str:
        table = match[0]
        return stash(table if table in _preserve_tables else _tidy_markdown_whitespace(_clean_table_math(table)))

    s = re.sub(r"^\s*```\w*\n|\n```\s*$", "", text)
    if drop_furniture:
        s = _FURNITURE_DIV.sub("", s)
    s = _trim_image_div_essays(s)
    s = _promote_section_headers(s)
    s = _TABLE.sub(stash_table, s)
    s = _lists_to_markdown(s)
    for pattern, repl in _HTML_TO_MD:
        s = pattern.sub(repl, s)
    s = _tidy_markdown_whitespace(html.unescape(s)).strip()
    return re.sub(r"\x00(\d+)\x00", lambda m: fragments[int(m[1])], s)


def read_chandra_result(result: ModelResult, drop_furniture: bool = False) -> SpecialistRead:
    raw = result.output.get("answer") or ""
    return SpecialistRead(chandra_markdown(raw, drop_furniture=drop_furniture), raw, result.output.get("finish_reason"))


def read_chart_verify_tables(result: ModelResult, hint: str | None = None) -> list[str]:
    """Extract verified HTML tables from a `build_chart_verify_call` answer (dropping invented Total rows)."""
    raw = result.output.get("answer") or ""
    tables = _TABLE.findall(raw)
    if hint:
        tables = [_grounded(t, hint) for t in tables]
    return tables


def _grounded(table: str, hint: str) -> str:
    """Drop Total/Average-style rows whose label the vector measurements never mention."""
    for row in list(_ROW.finditer(table)):
        cells = _CELL.findall(row[1])
        label = html.unescape(re.sub(r"<[^>]+>", "", cells[0])).strip() if cells else ""
        if _INVENTED.fullmatch(label) and label.lower() not in hint.lower():
            table = table.replace(row[0], "", 1)
    return table


def read_chart_result(result: ModelResult, hint: str | None = None, include_axes: bool = False) -> ChartRead:
    """Parse a chart VLM answer; pass `hint` to drop ungrounded Total/Average rows."""
    raw, finish = result.output.get("answer") or "", result.output.get("finish_reason")
    found = {k.lower(): v for k, v in _FIELD.findall(_TABLE.sub("", raw)) if v.lower() != "none"}
    title, axes, legend = (found.get(k) or None for k in ("title", "axes", "legend"))
    m = _TABLE.search(raw)
    table = (_grounded(m[0], hint) if hint else m[0]) if m else None
    parts = [p for p in (title, f"Axes: {axes}" if (include_axes and axes) else None, table) if p]
    return ChartRead("\n\n".join(parts), raw, finish, title, axes, legend, table)

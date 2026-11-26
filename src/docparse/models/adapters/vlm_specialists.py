"""Specialist VLM reads served by serve/modal_vlm.py: build the keyed call for an image, and read its answer.

One request shape for all three: input blob 0 is the image (keyed by its pixels), blob 1 the UTF-8 user prompt.
Revisions are pinned to exact Hugging Face commit SHAs so upstream weight changes never answer under an old key.
"""

from __future__ import annotations

import html
import re
from collections import ChainMap
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
_HTML_TO_MD = [
    (re.compile(pattern, _FLAGS), repl)
    for pattern, repl in [
        (r'<math\b[^>]*display="block"[^>]*>(.*?)</math>', r"\n\n$$\1$$\n\n"),
        (r"<math\b[^>]*>(.*?)</math>", r"$\1$"),
        (r"<h([1-6])\b[^>]*>(.*?)</h\1>", lambda m: "\n\n" + "#" * int(m[1]) + " " + m[2].strip() + "\n\n"),
        (r"<li\b[^>]*>(.*?)</li>", r"\n- \1"),
        (r"<(b|strong)\b[^>]*>(.*?)</\1>", r"**\2**"),
        (r"<(i|em)\b[^>]*>(.*?)</\1>", r"*\2*"),
        (r'<img\b[^>]*\balt="([^"]*)"[^>]*>', r"![\1]()"),
        (r"<input\b[^>]*\bchecked\b[^>]*>", "[x]"),
        (r"<input\b[^>]*>", "[ ]"),
        (r"<br\s*/?>", "\n"),
        (r"</?(?:p|div|ul|ol|pre|hr)\b[^>]*>", "\n\n"),
        (r"<[^>]+>", ""),
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
_FIGURE_DIV = re.compile(r'<div\b[^>]*data-label="Figure"[^>]*>(.*?)</div>', _FLAGS)
_CHART_ALT = re.compile(
    r'<img\b[^>]*alt="[^"]*\b(?:chart|graph|plot|bar|line|pie|histogram|scatter|series|axis|axes)\b',
    _FLAGS,
)


def has_chart_figure(raw_chandra: str) -> bool:
    """True when Chandra tagged at least one `Figure` block whose alt text describes a data chart/plot."""
    return any(_CHART_ALT.search(m.group(1)) for m in _FIGURE_DIV.finditer(raw_chandra))


def _html_tokens(raw_html: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", re.sub(r"<[^>]+>", " ", raw_html).lower()))


def fuse_figure_charts(
    raw_chandra: str,
    ocria_tables: list[str],
    *,
    ocria_vector_measured: bool = False,
    drop_furniture: bool = False,
) -> tuple[str, bool]:
    """Fuse vector-measured chart tables from a secondary reader into Chandra's chart `Figure` blocks.

    Preserves all Chandra prose, math, multi-column text, non-chart tables, and printed inset tables inside figures.
    Only injects/replaces a Figure block's table when Chandra omitted the chart table OR the secondary table was
    measured from PDF vector chart geometry.
    """
    fig_matches = [m for m in _FIGURE_DIV.finditer(raw_chandra) if _CHART_ALT.search(m.group(1))]
    if not fig_matches or not ocria_tables:
        return chandra_markdown(raw_chandra, drop_furniture=drop_furniture), False

    nonfig_html = _FIGURE_DIV.sub("", raw_chandra)
    nonfig_toks = [_html_tokens(t) for t in _TABLE.findall(nonfig_html)]
    chart_tbls: list[str] = []
    for tbl in ocria_tables:
        otoks = _html_tokens(tbl)
        if not otoks:
            continue
        if any(len(otoks & rt) / max(1, min(len(otoks), len(rt))) >= 0.35 for rt in nonfig_toks if rt):
            continue
        chart_tbls.append(tbl)
    if not chart_tbls:
        return chandra_markdown(raw_chandra, drop_furniture=drop_furniture), False

    ch_fig_tbls = [t for m in fig_matches for t in _TABLE.findall(m.group(1))]
    if ch_fig_tbls and not ocria_vector_measured:
        # Preserve Chandra's figure table(s) (including printed inset tables) unless PDF vector measurement improved
        # the secondary chart table.
        return chandra_markdown(raw_chandra, drop_furniture=drop_furniture), False

    replaced = raw_chandra
    for idx, m in enumerate(reversed(fig_matches)):
        inner_no_tbl = _TABLE.sub("", m.group(1))
        if idx == 0:
            new_div = f'<div data-label="Figure">{inner_no_tbl}\n' + "\n".join(chart_tbls) + "</div>"
        else:
            new_div = f'<div data-label="Figure">{inner_no_tbl}</div>'
        replaced = replaced[: m.start()] + new_div + replaced[m.end() :]
    return chandra_markdown(replaced, drop_furniture=drop_furniture), True


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
    """Convert Chandra's custom `<math>...</math>` tags inside HTML tables into readable HTML/Unicode."""
    if "<math" not in table_html.lower():
        return table_html

    def _repl(m: re.Match) -> str:
        inner = m.group(1).strip()
        for _ in range(3):
            nxt = _LATEX_CMD_WRAP.sub(r"\1", inner)
            if nxt == inner:
                break
            inner = nxt
        inner = _LATEX_SYM_RE.sub(lambda sm: _LATEX_SYM_MAP[sm.group(0)], inner)
        # Clean trivial spacing commands (`\,`, `\;`, `\ `) and simple `^{x}` / `_{x}` inside tables
        inner = re.sub(r"\\[,;:! ]", " ", inner)
        inner = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", inner)
        inner = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", inner)
        # If no complex LaTeX commands remain, return clean inline HTML; otherwise wrap as $...$
        if "\\" not in inner and "{" not in inner and "}" not in inner:
            return re.sub(r"\s+", " ", inner).strip()
        return f"${m.group(1).strip()}$"

    return _MATH_TAG.sub(_repl, table_html)


def chandra_markdown(text: str, drop_furniture: bool = False) -> str:
    """Chandra's layout-HTML answer as markdown; tables stay valid HTML with `<math>` normalized."""
    tables: list[str] = []

    def stash(m: re.Match) -> str:
        tables.append(_clean_table_math(m[0]))
        return f"\n\n\x00{len(tables) - 1}\x00\n\n"

    s = re.sub(r"^\s*```\w*\n|\n```\s*$", "", text)
    if drop_furniture:
        s = _FURNITURE_DIV.sub("", s)
    s = _TABLE.sub(stash, s)
    for pattern, repl in _HTML_TO_MD:
        s = pattern.sub(repl, s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: tables[int(m[1])], html.unescape(s))
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", s)).strip()


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

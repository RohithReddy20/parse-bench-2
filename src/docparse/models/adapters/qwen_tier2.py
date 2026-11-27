"""Tier-2 frontier reads with Qwen3.5 (27B-FP8 or 122B-A10B-FP8): keyed chart, form and hard-table calls, and
reading their answers.

Request shape: input blobs 0..n-1 are the images (the page, then an optional zoomed crop), each keyed by its pixels;
the last blob is the UTF-8 user prompt. Sampling and the system prompt live in `params`, so they enter the key too.
Dense pages that run out of tokens are re-read as overlapping horizontal strips and stitched by `merge_strip_tables`
(two strips) or `merge_n_strip_tables` (any number).
"""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

from PIL import Image

from docparse.core.hashing import sha256_hex
from docparse.models.adapters.vlm_specialists import RUNTIME
from docparse.models.calls import ModelCall, ModelRef, ModelResult
from docparse.pdf.render import pixel_hash, png_bytes

QWEN35_27B_FP8 = ModelRef(
    repo="Qwen/Qwen3.5-27B-FP8",
    revision="97f5941bf617e31c5e237364a8602ce3f03a551a",
    dtype="fp8",
)
QWEN35_122B_A10B_FP8 = ModelRef(
    repo="Qwen/Qwen3.5-122B-A10B-FP8",
    revision="a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9",
    dtype="fp8",
)
QWEN38_27B_FP8 = ModelRef(
    repo="Qwen/Qwen3.8-27B-FP8",
    revision="c3a462cc3d38cf351f3b26cbd763469eeb74ef81",
    dtype="fp8",
)
TIER2_MODELS = {"27b": QWEN35_27B_FP8, "122b": QWEN35_122B_A10B_FP8, "q38": QWEN38_27B_FP8}


def resolve_tier2_model(spec: str | None) -> ModelRef:
    """The Tier-2 model a config names: a `TIER2_MODELS` alias or a repo id; Qwen3.5-27B-FP8 when unset."""
    if not spec:
        return QWEN35_27B_FP8
    for alias, model in TIER2_MODELS.items():
        if spec.lower() == alias or spec == model.repo:
            return model
    raise ValueError(f"unknown tier2 model {spec!r}: use one of {sorted(TIER2_MODELS)} or their repo ids")


CHART_PROMPT_ID = "qwen3.5/chart-v2"
SINGLE_CHART_PROMPT_ID = "qwen3.5/chart-single-v1"
FORM_PROMPT_ID = "qwen3.5/form-v2"
FORM_REFINE_PROMPT_ID = "qwen3.5/form-refine-v1"
HARD_HTML_PROMPT_ID = "qwen3.5/hard-html-v2"

CHART_SYSTEM = (
    "You are an expert document data analyst extracting exact, complete structured HTML tables from charts and "
    "figures on document pages."
)
CHART_PROMPT = """\
Extract the exact data table(s) for EVERY chart and EVERY subplot/panel shown on this document page.
You are provided with:
- The full page image (and, when present, a zoomed high-resolution crop of the chart region).
- Exact measurements extracted directly from the PDF's vector drawing paths (trust these over eyeballing pixels \
wherever they correspond to a plotted bar, line, or marker):
<measurements>
{hint}
</measurements>
- Current draft HTML table(s) from an initial parser (may have missing panels, missing/swapped labels, or eyeballed \
numbers — verify everything visually against the image and vector measurements):
<draft_tables>
{draft}
</draft_tables>

Rules:
1. Preserve EVERY chart, EVERY panel/subplot, EVERY category, and EVERY series shown on the page. Never drop a panel \
or truncate categories.
2. Wherever <measurements> gives exact vector values for a plotted series, use those exact values (with the chart's \
printed units and signs). Read the image for any values, labels, titles, units, or categories not in <measurements>.
3. Match legend colors to series names carefully. Use exact printed labels, axis units, and chart title(s). Never use \
placeholders like "Series 1" or "Value" when an axis label or legend name is printed.
4. Do NOT invent Total, Average, or Sum rows/columns that are not actually plotted in the chart.
5. Output ONLY the exact chart title line(s) and clean HTML <table>...</table> (using <table>, <tr>, <th>, <td>, \
colspan/rowspan as needed). When a chart has multiple panels or subplots, emit one separate <table> per panel/subplot. \
No markdown code fences or commentary."""
CHART_THINK_RULE = (
    "\n6. Keep your internal reasoning concise (under 400 words), then immediately output the final HTML table(s)."
)
SINGLE_CHART_PROMPT = """\
Focus ONLY on the single chart/plot shown in the second (zoomed crop) image (use the first image only for \
surrounding title, caption, units, and legend context).

PDF vector measurements for this chart region (exact axis tick labels, bar/line coordinates, and printed \
data labels; empty if raster):
<vector_hint>
{hint}
</vector_hint>

Initial draft tables from the page (may mix panels or misread values — verify every label and value against \
the zoomed chart crop):
<draft>
{draft}
</draft>

Strict rules:
1. Transcribe ONLY the chart shown in the second (zoomed) image into a clean HTML `<table>` preceded by its short \
chart/panel title line. If the zoomed box itself contains stacked/side-by-side subplots with distinct axes, output \
one separate `<table>` per subplot.
2. NEVER invent summary/Total/Average rows or columns not drawn on the chart.
3. Prefer exact printed numbers on bars/points/slices first; for unlabelled bars/points/lines, use `<vector_hint>` \
axis measurements verified visually against the zoomed crop.
4. Keep exact units, signs (negative `-` values), percentages, and decimal precision. Preserve \
full category/legend names.
5. Output ONLY the short title line and HTML `<table>...</table>` (no code fences, prose paragraphs, or commentary)."""
CHART_V2B_PROMPT_ID = "qwen/chart-v2b"
SINGLE_CHART_V2B_PROMPT_ID = "qwen/chart-single-v2b"
CHART_V2B_RULES = """
Also:
- Every plotted mark needs a number: copy a printed data label; else use the vector measurements; else read the \
value off the axis from its position between the nearest gridlines or tick labels. An estimate read from the axis \
is expected, so never leave a cell empty or write "N/A" because a value is not printed. For a dense line or area \
with more points than x-axis labels, give each series' value at every labelled x-axis tick.
- One row per category or x value and one column per series: never put two series' values in one cell."""
CHART_V3_PROMPT_ID = "qwen/chart-v3"
SINGLE_CHART_V3_PROMPT_ID = "qwen/chart-single-v3"
_CHART_V3_RULES = """\
How to read a chart:
1. Each chart, and each panel or subplot of a chart, gets its own HTML `<table>`, preceded by one line with its \
printed title (the title above or beside that chart; a panel's own heading for a panel).
2. Table shape: one row per category or x-axis value, in axis order, with the label exactly as printed in the first \
column; one column per data series, named exactly as in the legend (or by the value axis title when there is a \
single series). Never put two series in one cell, never merge two panels into one table, and never use placeholder \
names such as "Series 1" when a legend or axis names the series.
3. Values: copy a printed data label exactly. Where the measurements give a value for a mark, use it. Otherwise read \
the value off the axis: find the gridlines or tick labels on either side of the mark and estimate where it sits \
between them (a bar ending 40% of the way from 20 to 30 is 24). Every plotted mark needs a value: an estimate read \
from the axis is what is asked for, so never leave a cell empty or write "N/A" because a value is not printed. Leave \
a cell empty only where the series has no mark at that category.
4. A line or area chart with more points than x-axis labels: one row per labelled x-axis tick, with each series' \
value at that tick.
5. Stacked bars: one column per segment with that segment's own size, not the running total. Pie or donut: one row \
per slice.
6. Keep signs, decimal precision, units and percent signs as the chart prints them. Do not add totals, averages, \
rows or series that are not plotted.
7. Output only the title lines and tables: no commentary, no code fences."""
CHART_V3_PROMPT = (
    """\
Extract the data behind EVERY chart on this document page as HTML tables.

You are given the full page image (and, when present, a zoomed crop of the chart region), measurements read from \
the PDF's vector drawing (exact wherever they correspond to a plotted bar, line or marker):
<measurements>
{hint}
</measurements>
and draft table(s) from an OCR model (often incomplete or misread: a starting point, not a source):
<draft_tables>
{draft}
</draft_tables>

"""
    + _CHART_V3_RULES
)
SINGLE_CHART_V3_PROMPT = (
    """\
Extract the data behind the ONE chart shown in the second image (a zoomed crop) as HTML tables. The first image is \
the whole page: use it only for the chart's title, caption, units and legend.

Measurements read from the PDF's vector drawing for this chart region (exact wherever they correspond to a plotted \
bar, line or marker; empty for a raster chart):
<measurements>
{hint}
</measurements>
Draft table(s) from an OCR model for the whole page (often incomplete, misread, or mixing charts: a starting point, \
not a source):
<draft_tables>
{draft}
</draft_tables>

"""
    + _CHART_V3_RULES
)
CHART_V3_THINK_RULE = (
    "\n8. Before answering, work out the axis scale and then each series in turn; keep the reasoning to the point."
)
MAX_IMAGE_SIDE = 2048  # keeps visual token count in Qwen3.5's optimal regime (~3-4k tokens) and 4x smaller manifests

FORM_PROMPT = """\
Transcribe this document form page completely and accurately into clean Markdown.

Strict rules:
1. Preserve every section heading, printed instruction, field number/code, and field label paired directly with its \
filled value (typed or handwritten).
2. Mark every checkbox or radio button from visual inspection: `[x]` ONLY if visibly checked/ticked/crossed/filled, \
`[ ]` if unchecked/empty.
3. Mark blank/unfilled fields with `[___]` (never invent values, never shift values into the wrong field, and never \
repeat `<br>` tags).
4. Use clean Markdown paragraphs/lines for normal label-value sections; use HTML `<table>` ONLY for genuine \
multi-column tabular grids.
5. For character-box / comb fields (dates, IDs, postal codes), output the complete assembled string cleanly (e.g. \
`2025-04-12`).
6. Keep your internal reasoning concise (under 350 words), then output only the faithful document transcription \
without code fences."""

HARD_HTML_PROMPT = """\
Transcribe this document page completely and faithfully into clean Markdown in natural reading order.

Strict rules:
1. Output EVERY table as complete, valid HTML `<table>` using `<thead>`, `<tbody>`, `<th>`, `<td>`, `colspan`, and \
`rowspan`.
   NEVER output LaTeX `\\begin{tabular}` and NEVER output Markdown pipe tables (`| ... |`).
2. Preserve EVERY row, EVERY column, and EVERY numeric cell verbatim — never summarize, skip rows, or truncate.
3. Convert mathematical formulas to `$...$` (inline) or `$$...$$` (display).
4. Output only the faithful document transcription with no code fences or commentary."""


def _cap_image(image: Image.Image, max_side: int = MAX_IMAGE_SIDE) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def _call(
    images: list[Image.Image],
    prompt_id: str,
    template: str,
    user: str,
    model: ModelRef,
    runtime: str,
    params: dict[str, Any],
) -> tuple[ModelCall, dict[str, bytes]]:
    capped = [_cap_image(im) for im in images]
    names, text = [pixel_hash(im) for im in capped], user.encode("utf-8")
    call = ModelCall(
        model=model,
        runtime=runtime,
        prompt_id=prompt_id,
        prompt_sha=sha256_hex(template.encode("utf-8")),
        params={"n_images": len(capped), **params},
        inputs=[*names, sha256_hex(text)],
    )
    blobs = {name: png_bytes(im) for name, im in zip(names, capped, strict=True)}
    blobs[call.inputs[-1]] = text
    return call, blobs


def build_chart_call(
    page_image: Image.Image,
    crop_image: Image.Image | None = None,
    vector_hint: str = "",
    draft_tables: str = "",
    *,
    model: ModelRef = QWEN35_27B_FP8,
    runtime: str = RUNTIME,
    enable_thinking: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int | None = 20,
    presence_penalty: float = 1.5,
    max_tokens: int = 10240,
    prompt: str = "v2",
    seed: int | None = None,
) -> tuple[ModelCall, dict[str, bytes]]:
    """Full page (plus an optional zoomed chart crop) with vector measurements and the draft tables to verify.
    `seed` asks for another independent read of the same chart."""
    images = [page_image] if crop_image is None else [page_image, crop_image]
    if prompt == "v3":
        template, prompt_id = CHART_V3_PROMPT + (CHART_V3_THINK_RULE if enable_thinking else ""), CHART_V3_PROMPT_ID
    elif prompt == "v2b":
        template = CHART_PROMPT + (CHART_THINK_RULE if enable_thinking else "") + CHART_V2B_RULES
        prompt_id = CHART_V2B_PROMPT_ID
    else:
        template, prompt_id = CHART_PROMPT + CHART_THINK_RULE if enable_thinking else CHART_PROMPT, CHART_PROMPT_ID
    user = template.format(hint=vector_hint.strip() or "(none)", draft=draft_tables.strip() or "(none)")
    params = _chart_params(enable_thinking, temperature, top_p, top_k, presence_penalty, max_tokens)
    if seed is not None:
        params["seed"] = seed
    return _call(images, prompt_id, template, user, model, runtime, params)


def build_single_chart_call(
    page_image: Image.Image,
    crop_image: Image.Image,
    *,
    vector_hint: str = "",
    draft_tables: str = "",
    model: ModelRef = QWEN35_122B_A10B_FP8,
    runtime: str = RUNTIME,
    enable_thinking: bool = False,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int | None = None,
    presence_penalty: float = 1.0,
    max_tokens: int = 10240,
    prompt: str = "v2",
    seed: int | None = None,
) -> tuple[ModelCall, dict[str, bytes]]:
    """One chart of a page with several: the page for context, the chart's zoomed crop, and its vector hint."""
    if prompt == "v3":
        template = SINGLE_CHART_V3_PROMPT + (CHART_V3_THINK_RULE if enable_thinking else "")
        prompt_id = SINGLE_CHART_V3_PROMPT_ID
    elif prompt == "v2b":
        template = SINGLE_CHART_PROMPT + (CHART_THINK_RULE if enable_thinking else "") + CHART_V2B_RULES
        prompt_id = SINGLE_CHART_V2B_PROMPT_ID
    else:
        template = SINGLE_CHART_PROMPT + CHART_THINK_RULE if enable_thinking else SINGLE_CHART_PROMPT
        prompt_id = SINGLE_CHART_PROMPT_ID
    user = template.format(hint=vector_hint.strip() or "(none)", draft=draft_tables.strip() or "(none)")
    params = _chart_params(enable_thinking, temperature, top_p, top_k, presence_penalty, max_tokens)
    if seed is not None:
        params["seed"] = seed
    return _call([page_image, crop_image], prompt_id, template, user, model, runtime, params)


def _chart_params(
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int | None,
    presence_penalty: float,
    max_tokens: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "system_prompt": CHART_SYSTEM,
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
    }
    if top_k is not None:
        params["top_k"] = top_k
    return params


def build_form_call(
    page_image: Image.Image,
    *,
    model: ModelRef = QWEN35_27B_FP8,
    runtime: str = RUNTIME,
    enable_thinking: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    max_tokens: int = 10240,
) -> tuple[ModelCall, dict[str, bytes]]:
    params = {
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
    }
    return _call([page_image], FORM_PROMPT_ID, FORM_PROMPT, FORM_PROMPT, model, runtime, params)


FORM_REFINE_PROMPT = """\
Verify and correct this document form page transcription against the page image.

Current draft transcription from an initial parser (may have flipped or missed checkboxes, split character-box \
digits, or omitted blank/handwritten fields — keep all accurate printed text and headings intact):
<draft>
{draft}
</draft>

Strict rules:
1. Keep ALL accurate printed headings, field numbers, labels, instructions, and values from `<draft>` — NEVER drop \
printed text or omit sections.
2. Visually inspect EVERY checkbox and radio button on the image: mark `[x]` ONLY if visibly checked/ticked/crossed, \
and `[ ]` if unchecked/empty. Fix every misread checkbox from `<draft>`.
3. For comb / character-box fields (dates, postal codes, ID numbers written one character per box), assemble the \
complete value cleanly (e.g. `2025-04-12`, `M2 9YT`) without mixing printed box numbers into the value.
4. Keep genuinely blank/unfilled fields marked `[___]` (never invent values and never shift a value into a neighbor).
5. Output ONLY the clean, corrected Markdown transcription (HTML `<table>` only for true tabular grids). No code \
fences, commentary, or repeated `<br>` tags."""


def build_form_refine_call(
    page_image: Image.Image,
    draft_markdown: str,
    *,
    model: ModelRef = QWEN35_27B_FP8,
    runtime: str = RUNTIME,
    enable_thinking: bool = False,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int | None = None,
    presence_penalty: float = 1.0,
    max_tokens: int = 8192,
) -> tuple[ModelCall, dict[str, bytes]]:
    """Verify and correct an initial form transcription against the page image."""
    user = FORM_REFINE_PROMPT.format(draft=draft_markdown.strip() or "(none)")
    params: dict[str, Any] = {
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
    }
    if top_k is not None:
        params["top_k"] = top_k
    return _call([page_image], FORM_REFINE_PROMPT_ID, FORM_REFINE_PROMPT, user, model, runtime, params)


def build_hard_table_call(
    page_image: Image.Image,
    *,
    model: ModelRef = QWEN35_27B_FP8,
    runtime: str = RUNTIME,
    max_tokens: int = 16384,
    temperature: float = 0.1,
    top_p: float = 0.95,
    presence_penalty: float = 1.0,
) -> tuple[ModelCall, dict[str, bytes]]:
    """Non-thinking HTML-only transcription of a page (or one strip of it) dense with tables."""
    params = {
        "enable_thinking": False,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
    }
    return _call([page_image], HARD_HTML_PROMPT_ID, HARD_HTML_PROMPT, HARD_HTML_PROMPT, model, runtime, params)


def make_horizontal_strips(page_image: Image.Image, overlap: float = 0.12) -> tuple[Image.Image, Image.Image]:
    """Top and bottom halves of a page, each extended by `overlap / 2` of its height across the middle."""
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap must be within [0, 1], got {overlap}")
    image = page_image.convert("RGB")
    w, h = image.size
    cut_top = min(h, max(1, round(h * (0.5 + overlap / 2))))
    cut_bot = max(0, min(h - 1, round(h * (0.5 - overlap / 2))))
    return image.crop((0, 0, w, cut_top)), image.crop((0, cut_bot, w, h))


def make_n_horizontal_strips(page_image: Image.Image, n: int = 3, overlap: float = 0.14) -> list[Image.Image]:
    """`n` equal bands of a page, top to bottom, each extended by `overlap / 2` of the page height across every
    interior boundary (n = 2 gives `make_horizontal_strips`)."""
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap must be within [0, 1], got {overlap}")
    image = page_image.convert("RGB")
    w, h = image.size
    strips = []
    for i in range(n):
        top = 0 if i == 0 else max(0, min(h - 1, round(h * (i / n - overlap / 2))))
        bot = h if i == n - 1 else min(h, max(top + 1, round(h * ((i + 1) / n + overlap / 2))))
        strips.append(image.crop((0, top, w, bot)))
    return strips


_FLAGS = re.DOTALL | re.IGNORECASE
_TABLE = re.compile(r"<table\b.*?</table>", _FLAGS)
_THEAD = re.compile(r"<thead\b.*?</thead>", _FLAGS)
_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", _FLAGS)
_CELL = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", _FLAGS)
_TD, _TH = re.compile(r"<td\b", re.I), re.compile(r"<th\b", re.I)
_TAG = re.compile(r"<[^>]+>")
_SIMILAR = 0.85  # share of tokens two transcriptions of one row (or line) must have in common

Key = tuple[str, ...]


def _norm(text: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", text)).casefold().split())


def _row_key(row: re.Match[str]) -> Key:
    return tuple(_norm(cell) for cell in _CELL.findall(row.group(1)))


def _line_key(line: str) -> Key:
    return (" ".join(line.casefold().split()),)


def _similar(a: Key, b: Key) -> bool:
    if a == b:
        return True
    ta, tb = Counter(" ".join(a).split()), Counter(" ".join(b).split())
    return bool(ta and tb) and (ta & tb).total() / max(ta.total(), tb.total()) >= _SIMILAR


def _overlap(a: list[Key], b: list[Key]) -> int:
    """Largest k such that the last k keys of `a` match the first k of `b` pairwise, with some text among them."""
    for k in range(min(len(a), len(b)), 0, -1):
        tail = a[len(a) - k :]
        if any(any(key) for key in tail) and all(_similar(x, y) for x, y in zip(tail, b[:k], strict=True)):
            return k
    return 0


def _align(top: list[Key], bot: list[Key]) -> tuple[int, int, int]:
    """(k, j, s): `bot[j : j + k]` repeats the k rows ending `s` rows before the end of `top`.

    j = 1 skips a row the bottom strip shows cut off at its upper edge (the top strip has it whole, just above the
    run); s = 1 replaces a row cut off at the top strip's lower edge with the bottom strip's whole copy. Either skip
    needs two matching rows as an anchor."""
    for j, s in ((0, 0), (1, 0), (0, 1), (1, 1)):
        k = _overlap(top[: len(top) - s], bot[j:])
        if k >= 1 + (j or s):
            return k, j, s
    return 0, 0, 0


def _head_count(table: str, rows: list[re.Match[str]]) -> int:
    """Leading header rows: those inside <thead>, else the leading rows made only of <th> cells."""
    if thead := _THEAD.search(table):
        return sum(1 for r in rows if r.start() < thead.end())
    n = 0
    while n < len(rows) and _TH.search(rows[n].group(1)) and not _TD.search(rows[n].group(1)):
        n += 1
    return n


def _merge_tables(top: str, bot: str) -> str | None:
    """`top` extended by the rows of `bot` it lacks, or None when nothing shows they are one table."""
    t_rows, b_rows = list(_ROW.finditer(top)), list(_ROW.finditer(bot))
    t_keys, b_keys = [_row_key(r) for r in t_rows], [_row_key(r) for r in b_rows]
    t_head, b_head = _head_count(top, t_rows), _head_count(bot, b_rows)
    header = t_keys[:t_head]
    drop = b_head if b_head and all(any(_similar(k, h) for h in header) for k in b_keys[:b_head]) else 0
    k, j, s = _align(t_keys[t_head:], b_keys[drop:])
    if not (drop or k):
        return None
    extra = "".join(("\n" if "\n" in top else "") + r.group(0) for r in b_rows[drop + j + k :])
    cut, end = t_rows[len(t_rows) - s - 1].end(), t_rows[-1].end()
    thead = _THEAD.search(top)
    if thead and cut < thead.end():  # the top strip holds only the header: open a body after it
        cut = end = thead.end()
        extra = f"<tbody>{extra}</tbody>" if extra else ""
    return top[:cut] + extra + top[end:]


def _join_text(a: str, b: str) -> str:
    """`a` then `b`, minus the leading lines of `b` that repeat the end of `a` (blank lines ignored)."""
    la, lb = a.strip().splitlines(), b.strip().splitlines()
    nb = [i for i, line in enumerate(lb) if line.strip()]
    k = _overlap([_line_key(x) for x in la if x.strip()], [_line_key(lb[i]) for i in nb])
    if k:
        return "\n".join(la + lb[nb[k - 1] + 1 :]).strip()
    return "\n\n".join(p for p in (a.strip(), b.strip()) if p)


def merge_strip_tables(top_md: str, bot_md: str) -> str:
    """Stitch transcriptions of a page's top and bottom strips (see `make_horizontal_strips`) into one.

    The last table of the top strip and the first of the bottom strip are merged when a repeated header or
    repeated rows show they are one table; the bottom copy's repeated header and rows are dropped. Text around them,
    and strips without a shared table, are joined with the lines they repeat across the overlap dropped once."""
    top_tables, bot_tables = list(_TABLE.finditer(top_md)), list(_TABLE.finditer(bot_md))
    if top_tables and bot_tables:
        t, b = top_tables[-1], bot_tables[0]
        merged = _merge_tables(t.group(0), b.group(0))
        if merged is not None:
            head = _join_text(top_md[: t.start()], bot_md[: b.start()])
            tail = _join_text(top_md[t.end() :], bot_md[b.end() :])
            return "\n\n".join(p for p in (head, merged, tail) if p)
    return _join_text(top_md, bot_md)


def merge_n_strip_tables(strips: Sequence[str]) -> str:
    """Stitch transcriptions of a page's strips, top to bottom (see `make_n_horizontal_strips`), pairwise with
    `merge_strip_tables`; empty strips are skipped."""
    return reduce(merge_strip_tables, (s for s in strips if s.strip()), "")


@dataclass
class Tier2Read:
    markdown: str
    reasoning: str
    finish_reason: str | None
    usable: bool  # finished, non-empty and free of degenerate repetition


_FENCE = re.compile(r"\A\s*```[\w-]*[ \t]*\n(.*?)(?:\n?```)?\s*\Z", re.DOTALL)
_BR_LOOP = re.compile(r"(?:<br\s*/?>\s*){12,}", re.IGNORECASE)
_LOOP_LINES = 12  # this many consecutive identical non-blank lines with text is a decoding loop
_TEXT = re.compile(r"[^\W_]")


def _looped(text: str) -> bool:
    if _BR_LOOP.search(text):
        return True
    run, prev = 0, None
    for line in (" ".join(x.split()) for x in text.splitlines()):
        if not line:
            continue
        run = run + 1 if line == prev else 1
        prev = line
        if run >= _LOOP_LINES and _TEXT.search(_norm(line)):
            return True
    return False


def read_tier2_result(result: ModelResult) -> Tier2Read:
    out: dict[str, Any] = result.output
    answer, reasoning, finish = out.get("answer") or "", out.get("reasoning") or "", out.get("finish_reason")
    if "</think>" in answer:  # served without a reasoning parser: the thought leads the answer
        thought, answer = answer.rsplit("</think>", 1)
        reasoning = reasoning or thought.replace("<think>", "", 1).strip()
    if fenced := _FENCE.match(answer):
        answer = fenced.group(1)
    markdown = answer.strip()
    usable = finish == "stop" and bool(markdown) and not _looped(markdown)
    return Tier2Read(markdown, reasoning.strip(), finish, usable)


FORM_KV_PROMPT_ID = "qwen/form-kv-v2"
NOT_A_FORM = "NOT_A_FORM"
FORM_KV_PROMPT = """\
Rewrite this form page as clean Markdown that keeps every printed word and pairs each field with what was filled in.

You are given the page image and a draft transcription from an OCR model. Its words are mostly right, but it may \
detach a value from its field, lay fields out as loose table cells, flip or miss checkboxes, or split \
character-box values:
<draft>
{draft}
</draft>
{layer}
Rules:
1. One field per line, written `caption: value`. The caption is the field's printed label with its printed box or \
line number/code in front when it has one (for example `3.99 Plant, machinery and motor vehicles: £27,525`). A \
printed number or code that only identifies a box belongs to the caption and is never a value.
2. A field left empty on the page keeps its caption and gets `[___]` as its value. Never put a printed hint, example, \
instruction, unit label or box number into a value, and never guess a value.
3. Checkboxes and radio buttons: `[x]` when ticked, crossed or filled and `[ ]` when empty, written directly before \
the option's label, one option per line under its question, with every printed option listed.
4. Copy values exactly as the page shows them (digits, signs, separators, spelling). A date, number or code written \
one character per box is joined into one value.
5. A real grid (several rows that share the same column captions) stays an HTML `<table>` with the column captions \
in a header row and each row's caption in its first cell. Do not use a table for anything else.
6. Keep all other printed text (titles, section headings as `#`/`##`, instructions, notes, page furniture) in \
reading order, and keep the draft's wording wherever it matches the page.
7. Output only the Markdown: no code fences, no commentary.
8. If the page is not a form (it has no fields, boxes or blanks for a person to fill in: for example a report, a \
letter, a statement or a plain data table), output exactly `NOT_A_FORM` and nothing else."""
FORM_KV_LAYER = """
The PDF's own text layer, line by line from top to bottom (exact characters of the typed text; trust it for \
spelling and digits, and use the image to decide which value belongs to which field):
<text_layer>
{lines}
</text_layer>
"""


def build_form_kv_call(
    page_image: Image.Image,
    draft_markdown: str,
    layer_lines: str = "",
    *,
    model: ModelRef = QWEN38_27B_FP8,
    runtime: str = RUNTIME,
    enable_thinking: bool = False,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int | None = None,
    presence_penalty: float = 1.0,
    max_tokens: int = 8192,
) -> tuple[ModelCall, dict[str, bytes]]:
    """Rewrite a form page as `caption: value` lines from the page image, the draft and (when the page has one)
    the PDF text layer."""
    layer = FORM_KV_LAYER.format(lines=layer_lines.strip()) if layer_lines.strip() else ""
    user = FORM_KV_PROMPT.format(draft=draft_markdown.strip() or "(none)", layer=layer)
    params: dict[str, Any] = {
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
    }
    if top_k is not None:
        params["top_k"] = top_k
    template = FORM_KV_PROMPT + FORM_KV_LAYER
    return _call([page_image], FORM_KV_PROMPT_ID, template, user, model, runtime, params)


PAGE_PROMPT_ID = "qwen/page-v1"
PAGE_SYSTEM = "You are a meticulous document transcriber. You copy pages exactly as printed and never add commentary."
PAGE_PROMPT = """\
Transcribe this document page into Markdown, completely and exactly, in natural reading order.

Rules:
1. Copy every word, number, sign, unit and symbol exactly as printed, including running headers, footers, page \
numbers, footnotes, captions, stamps and handwriting. Never summarize, correct, translate or reorder, and never \
write text that is not on the page.
2. Headings use `#`/`##`/`###` by level. Keep bold (`**`), italic (`*`), underline (`<u>`), strikethrough (`~~`), \
superscript (`<sup>`) and subscript (`<sub>`) where the page shows them. Keep list markers and numbering.
3. Every table is an HTML `<table>` with `<thead>`/`<th>` header rows and `colspan`/`rowspan` for merged cells. \
Keep EVERY row and column in order, with empty cells left empty; never use Markdown pipe tables.
4. Formulas are LaTeX: `$...$` inline, `$$...$$` on their own line.
5. Forms: write each field label followed by its filled value on the same line; a checkbox or radio button is \
`[x]` when it is ticked, crossed or filled and `[ ]` when empty, placed by its label. Leave an unfilled field empty: \
never invent a value.
6. A chart or plot becomes its title line followed by an HTML `<table>` of the plotted data (one row per category \
or x value, one column per series, named as in the legend or axis; read unlabelled values off the axis). One table \
per chart or panel. Any other picture, logo or diagram becomes `![short description]()`.
7. Output only the transcription: no code fences, no explanations."""


def build_page_call(
    page_image: Image.Image,
    *,
    model: ModelRef = QWEN38_27B_FP8,
    runtime: str = RUNTIME,
    enable_thinking: bool = False,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int | None = None,
    presence_penalty: float = 1.0,
    max_tokens: int = 16384,
) -> tuple[ModelCall, dict[str, bytes]]:
    """A full transcription of one page by a general VLM: an independent third reading of the page."""
    params: dict[str, Any] = {
        "system_prompt": PAGE_SYSTEM,
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
    }
    if top_k is not None:
        params["top_k"] = top_k
    return _call([page_image], PAGE_PROMPT_ID, PAGE_PROMPT, PAGE_PROMPT, model, runtime, params)

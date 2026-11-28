"""The parser's only public entry point: `parse(pdf_bytes, config, client)`.

It receives bytes and nothing else: no file name, path, example id or subset (design §3, anti-goals).
A pure function of (bytes, config, cache): model answers come from the client, which reads the cache.

    S0 render -> S2 OCRia-1 page read (one cached call per page) -> S3 answer parsed into the IR
    -> S4 correctors, each behind its options (layout_options, text_options, format_options) -> writers

With `tier2_charts`, Chandra pages that show a chart are read once more by Qwen3.5 (Tier-2) before S4 (once per
chart when `per_chart_crops` finds several large ones); with `tier2_forms`, Chandra form pages without a Tier-2 chart
read have their draft refined by it the same way; with `tier2_tables`, cut-off or low-recall table pages are re-read
by it in overlapping horizontal strips.
"""

from __future__ import annotations

import html
import re
from collections import ChainMap
from dataclasses import dataclass, field

from docparse import __version__
from docparse.base.ocria_parse import parse_page
from docparse.core.config import Config
from docparse.core.hashing import sha256_hex
from docparse.core.ir import CostRecord, Document, Finding
from docparse.fixes.layout_kdl import KDL_OPTIONS
from docparse.models.adapters import heron, kdl, ppdoclayout
from docparse.models.adapters.attention import attention_call, read_attention
from docparse.models.adapters.ocria import build_page_call, read_page_result
from docparse.models.calls import ModelCall, ModelClient, ModelRef, ModelResult, Pending
from docparse.pdf.render import RenderedPage, render_pages
from docparse.write.layout_items import items_from_layout_page

_FLAGS = re.DOTALL | re.IGNORECASE
_TABLE = re.compile(r"<table\b.*?</table>", _FLAGS)
_CHANDRA_DIV = re.compile(r"<div\b([^>]*)>(.*?)</div>", _FLAGS)
_FIGURE_LABEL = re.compile(r"""\bdata-label\s*=\s*(["'])Figure\1""", re.IGNORECASE)
_BBOX = re.compile(r'\bdata-bbox\s*=\s*"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"')
_IMG_ALT = re.compile(r"""<img\b[^>]*?\salt\s*=\s*(["'])(.*?)\1""", _FLAGS)
_MATH = re.compile(r"\$\$|\\frac|\\begin\{equation\}|<math", re.IGNORECASE)
# display or inline formulas in either reader's dialect (Chandra `<math>`, LaTeX `$...$`, `\\(...\\)`)
_ANY_MATH = re.compile(r"\$\$|\\frac|\\begin\{|<math|\\\(|\$[^$\n]{0,80}[\\^_][^$\n]{0,80}\$", re.IGNORECASE)
_FORM_LABEL = re.compile(r"""<div\b[^>]*\bdata-label\s*=\s*(["'])Form\1""", re.IGNORECASE)
_CHECKBOX = re.compile(r"\[[ xX]\]|<input\b")
_BLANK_MARK = re.compile(r"\[[ \t]*_{2,}[ \t]*\]")
CHART_PAD = 0.06  # share of the page width / height added around the chart boxes before cropping
CHART_CROP_SHARE = (0.03, 0.92)  # a crop is sent only when the padded box covers this share of the page
DUPLICATE_SHARE = 0.35  # token overlap that makes a Tier-2 table a repeat of a table Chandra already read
CHART_DUP_IOU, CHART_DUP_INNER = 0.50, 0.80  # chart boxes this alike (IoU, or share of the smaller inside) repeat
CHART_ROW_BAND = 120  # page pixels: chart boxes whose tops share a band are read left to right


@dataclass
class ParseResult:
    document: Document | None  # None while any model call is still pending
    pending: list[str] = field(default_factory=list)  # keys of the calls a GPU session has to run


def _ocria_markdown(
    pdf_bytes: bytes,
    index: int,
    rp,
    read,
    call_key: str,
    config: Config,
) -> tuple[str, list[str], bool]:
    """Run the PDF-grounded OCRia-1 corrector stack and return `(markdown, html_tables, vector_measured)`."""
    from docparse.write.markdown import write_page

    markdown = read.markdown
    if config.fallback_below > 0:
        from docparse.fixes.fallback import MIN_TOKENS, has_math, looped, reading_recall, text_dump, trim_loop_tail
        from docparse.fixes.layout_refine import words_in_points

        cut_short = read.finish_reason == "length" or looped(markdown)
        if cut_short:
            markdown = trim_loop_tail(markdown)
            words = words_in_points(rp)
            recall = reading_recall(markdown, words) if len(words) >= MIN_TOKENS else 1.0
            keep_math = has_math(markdown) and recall >= 0.20
            if len(words) >= MIN_TOKENS and recall < config.fallback_below and not keep_math:
                markdown = text_dump(words)
    if config.form_options:
        from docparse.fixes.forms import repair
        from docparse.pdf.overlay import overlay_fonts

        markdown = repair(markdown, rp, overlay_fonts(pdf_bytes, index), config.form_options, pdf_bytes, index)
    if config.text_options.get("empty_read"):
        from docparse.fixes.text import empty_read

        markdown = empty_read(markdown, rp) or markdown
    pg = parse_page(markdown, index=index, call_key=call_key)
    if config.text_options:
        from docparse.fixes.text import refine_text

        refine_text(pg, rp, config.text_options)
    if config.format_options:
        from docparse.fixes.formatting import refine_format

        refine_format(pg, rp, config.format_options)
    pg.width, pg.height, pg.rotation = rp.width_pt, rp.height_pt, rp.rotation
    if config.table_options:
        from docparse.fixes.tables import fix_tables

        fix_tables(pg, rp, config.table_options)
    vector_measured = False
    if config.chart_options.get("measure") and any(r.kind == "table" for r in pg.regions):
        from docparse.fixes.charts import correct_tables
        from docparse.fixes.layout_refine import words_in_points
        from docparse.pdf.graphics import load_shapes

        try:
            correct_tables(pg, words_in_points(rp), load_shapes(pdf_bytes, index), config.chart_options)
            vector_measured = any(r.chosen().reader == "fix.charts.vector" for r in pg.regions)
        except Exception:
            for region in pg.regions:
                region.selected = region.candidates[0].id
    tables = [r.chosen().raw for r in pg.regions if r.kind == "table"]
    return write_page(pg), tables, vector_measured


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", re.sub(r"<[^>]+>", " ", text).lower()))


def _chandra_figures(raw: str) -> list[tuple[int, int, str, tuple[int, ...] | None]]:
    """`(start, end, alt text, 0-1000 data-bbox or None)` of each `Figure` block of Chandra's layout HTML."""
    figures = []
    for m in _CHANDRA_DIV.finditer(raw):
        if _FIGURE_LABEL.search(m[1]):
            bbox = _BBOX.search(m[1])
            alt = " ".join(a[2] for a in _IMG_ALT.finditer(m[2]))
            figures.append((m.start(), m.end(), alt, tuple(int(v) for v in bbox.groups()) if bbox else None))
    return figures


_PICTURE_LABEL = re.compile(r"""\bdata-label\s*=\s*(["'])(?:Figure|Image|Diagram)\1""", re.IGNORECASE)


def _picture_words(raw: str) -> int:
    """Words Chandra wrote for the page's pictures (Figure / Image / Diagram blocks: alt text, description, the
    text of a diagram). Text inside a picture is in no text layer and the other readers do not transcribe it."""
    n = 0
    for m in _CHANDRA_DIV.finditer(raw):
        if _PICTURE_LABEL.search(m[1]):
            alt = " ".join(a[2] for a in _IMG_ALT.finditer(m[2]))
            n += len(re.findall(r"\w+", alt + " " + re.sub(r"<[^>]+>", " ", m[2])))
    return n


def _picture_notes(raw: str, min_words: int) -> str:
    """Chandra's transcription of the page's pictures (Figure / Diagram blocks that are not data charts and hold no
    table) as `![...]()` lines, for a page whose text comes from another reader: that reader does not describe
    pictures, and what a picture shows is in no text layer. Blocks under `min_words` words are left out."""
    from docparse.models.adapters.vlm_specialists import _CHART_ALT

    notes = []
    for m in _CHANDRA_DIV.finditer(raw):
        if not re.search(r"""\bdata-label\s*=\s*(["'])(?:Figure|Diagram)\1""", m[1], re.IGNORECASE):
            continue
        if "<table" in m[2].lower():
            continue
        alt = " ".join(a[2] for a in _IMG_ALT.finditer(m[2]))
        if _CHART_ALT.search(alt):
            continue
        rest = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<img\b[^>]*>", " ", m[2]))).split())
        text = f"{html.unescape(alt)} {rest}".strip()
        if len(re.findall(r"\w+", text)) < min_words:
            continue
        text = text[:1500].replace("[", "(").replace("]", ")").replace("\n", " ")
        notes.append(f"![{text}]()")
    return "\n\n".join(notes)


def _chart_crop(rp: RenderedPage, boxes: list[tuple[float, ...]]):
    """`(crop, box_pt)`: the union of the chart `boxes` (page pixels) padded by CHART_PAD and clipped to the page, as
    a crop of the page image (None when it covers too little or nearly all of the page) and in PDF points."""
    if not boxes:
        return None, None
    w, h = rp.image.size
    x0, y0 = max(0.0, min(b[0] for b in boxes) - CHART_PAD * w), max(0.0, min(b[1] for b in boxes) - CHART_PAD * h)
    x1, y1 = min(w, max(b[2] for b in boxes) + CHART_PAD * w), min(h, max(b[3] for b in boxes) + CHART_PAD * h)
    if x1 <= x0 or y1 <= y0:
        return None, None
    lo, hi = CHART_CROP_SHARE
    crop = None
    if lo <= (x1 - x0) * (y1 - y0) / (w * h) <= hi:
        crop = rp.image.crop((round(x0), round(y0), round(x1), round(y1)))
    return crop, (x0 * rp.scale, y0 * rp.scale, x1 * rp.scale, y1 * rp.scale)


def _zoom_crop(crop, opts: dict):
    """A chart crop enlarged so its longer side reaches `crop_zoom_to` pixels (small charts keep their tick labels
    and thin marks legible to the reader); unchanged when larger already or the option is unset."""
    target = int(opts.get("crop_zoom_to", 0) or 0)
    if crop is None or not target or max(crop.size) >= target:
        return crop
    from PIL import Image

    scale = target / max(crop.size)
    return crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS
    )


def _chart_hint(
    pdf_bytes: bytes,
    index: int,
    rp: RenderedPage,
    box_pt: tuple[float, ...] | None,
    *,
    max_chars: int = 2500,
    max_series: int = 12,
) -> str:
    """The PDF's vector measurements of the chart in `box_pt` (the whole page when None); "" without any."""
    from docparse.fixes.chart_hints import extract_chart_hint, format_chart_hint
    from docparse.pdf.graphics import load_shapes

    try:
        return format_chart_hint(
            extract_chart_hint(rp, load_shapes(pdf_bytes, index), box_pt, max_series=max_series),
            max_chars=max_chars,
        )
    except Exception:  # the hint is optional evidence: image uploads and odd geometry read the chart from pixels
        return ""


def _box_area(b: tuple[float, ...]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _repeats(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Two boxes of positive area that show one chart: IoU >= CHART_DUP_IOU, or the smaller mostly inside."""
    inter = _box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = _box_area(a) + _box_area(b) - inter
    return inter >= CHART_DUP_IOU * union or inter >= CHART_DUP_INNER * min(_box_area(a), _box_area(b))


def _dedup_chart_boxes(boxes: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    """Chart `boxes` without repeats (the larger of two is kept), in reading order: top to bottom in bands of
    CHART_ROW_BAND pixels, then left to right."""
    kept: list[tuple[float, ...]] = []
    for box in sorted((b for b in boxes if _box_area(b) > 0), key=_box_area, reverse=True):
        if not any(_repeats(box, k) for k in kept):
            kept.append(box)
    return sorted(kept, key=lambda b: (round(b[1] / CHART_ROW_BAND), b[0]))


def _union_share(rp: RenderedPage, boxes: list[tuple[float, ...]]) -> float:
    """Share of the page the enclosing bounding box of `boxes` (clipped to the page) covers — matching `_chart_crop`."""
    if not boxes:
        return 0.0
    w, h = rp.image.size
    x0, y0 = max(0.0, min(b[0] for b in boxes)), max(0.0, min(b[1] for b in boxes))
    x1, y1 = min(w, max(b[2] for b in boxes)), min(h, max(b[3] for b in boxes))
    return _box_area((x0, y0, x1, y1)) / (w * h)


def _chart_sampling(opts: dict) -> dict:
    """Sampling of a Tier-2 chart read (TIER2_CHART_OPTIONS); a thinking read defaults to top_k 20."""
    thinking = bool(opts.get("enable_thinking", False))
    return {
        "enable_thinking": thinking,
        "temperature": float(opts.get("temperature", 0.1)),
        "top_p": float(opts.get("top_p", 0.95)),
        "top_k": int(opts["top_k"]) if "top_k" in opts else (20 if thinking else None),
        "presence_penalty": float(opts.get("presence_penalty", 1.0)),
        "max_tokens": int(opts.get("max_tokens", 10240)),
        "prompt": str(opts.get("prompt", "v2")),
        "seed": int(opts["seed"]) if opts.get("seed") is not None else None,
    }


def _tier2_chart_call(
    pdf_bytes: bytes,
    index: int,
    rp: RenderedPage,
    turn: int,
    raw_chandra: str,
    pp_boxes: list[tuple[float, ...]],
    drafts: list[str],
    opts: dict,
    model: ModelRef | None = None,
) -> tuple[ModelCall, dict[str, bytes]]:
    """Tier-2 chart read: the page, a zoomed crop of its charts, their vector hint, and the draft tables to verify."""
    from docparse.models.adapters import qwen_tier2
    from docparse.models.adapters.vlm_specialists import _CHART_ALT

    w, h = rp.image.size
    figures = [b for _, _, alt, b in _chandra_figures(raw_chandra) if b and _CHART_ALT.search(alt)]
    boxes = pp_boxes or [(b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000) for b in figures]
    crop, box_pt = _chart_crop(rp, boxes)
    h_chars = int(opts.get("hint_chars", opts.get("hint_max_chars", 2500)))
    h_series = int(opts.get("hint_max_series", 12))
    hint = "" if turn else _chart_hint(pdf_bytes, index, rp, box_pt, max_chars=h_chars, max_series=h_series)
    return qwen_tier2.build_chart_call(
        rp.image,
        crop_image=_zoom_crop(crop, opts),
        vector_hint=hint,
        draft_tables="\n\n".join(drafts or _TABLE.findall(raw_chandra)),
        model=model or qwen_tier2.resolve_tier2_model(opts.get("model")),
        **_chart_sampling(opts),
    )


def _tier2_single_chart_call(
    pdf_bytes: bytes,
    index: int,
    rp: RenderedPage,
    turn: int,
    box: tuple[float, ...],
    draft: str,
    opts: dict,
    model: ModelRef,
) -> tuple[ModelCall, dict[str, bytes]]:
    """Tier-2 read of one chart `box` (page pixels): the page for context, the chart's zoomed crop (the exact box when
    the padded one covers too little or too much of the page), its vector hint, and the page's draft tables."""
    from docparse.models.adapters import qwen_tier2

    crop, box_pt = _chart_crop(rp, [box])
    if crop is None:
        w, h = rp.image.size
        x0, y0, x1, y1 = max(0.0, box[0]), max(0.0, box[1]), min(w, box[2]), min(h, box[3])
        crop = rp.image.crop((round(x0), round(y0), round(x1), round(y1)))
        box_pt = (x0 * rp.scale, y0 * rp.scale, x1 * rp.scale, y1 * rp.scale)
    h_chars = int(opts.get("hint_chars", opts.get("hint_max_chars", 2500)))
    h_series = int(opts.get("hint_max_series", 12))
    hint = "" if turn else _chart_hint(pdf_bytes, index, rp, box_pt, max_chars=h_chars, max_series=h_series)
    return qwen_tier2.build_single_chart_call(
        rp.image, _zoom_crop(crop, opts), vector_hint=hint, draft_tables=draft, model=model, **_chart_sampling(opts)
    )


def _tier2_markdown(
    raw_chandra: str,
    markdown: str,
    answer: str,
    chart_figure: bool,
    drop_furniture: bool,
    table_figures: bool = True,
) -> str | None:
    """The page with the Tier-2 chart tables: in Chandra's chart `Figure` blocks, or (for charts only PP-DocLayoutV3
    saw) appended with their titles, minus the tables Chandra already read. None when the answer changes nothing.
    With `table_figures`, a `Figure` block holding a table also takes them (the chart was confirmed by a detector)."""
    from docparse.models.adapters import vlm_specialists

    tables = _TABLE.findall(answer)
    if not tables:
        return None
    has_tbl_fig = table_figures and any("<table" in f.body.lower() for f in vlm_specialists._figure_divs(raw_chandra))
    if chart_figure or has_tbl_fig:
        fused_md, fused = vlm_specialists.fuse_figure_charts(
            raw_chandra,
            tables,
            ocria_vector_measured=True,
            drop_furniture=drop_furniture,
            allow_table_figures=table_figures,
        )
        return fused_md if fused else None
    spans = [(start, end) for start, end, _, _ in _chandra_figures(raw_chandra)]
    known = [_tokens(t[0]) for t in _TABLE.finditer(raw_chandra) if not any(s <= t.start() < e for s, e in spans)]
    parts: list[str] = []
    pos = 0
    for t in _TABLE.finditer(answer):
        title, pos, toks = answer[pos : t.start()].strip(), t.end(), _tokens(t[0])
        if not any(k and len(toks & k) / max(1, min(len(toks), len(k))) >= DUPLICATE_SHARE for k in known):
            parts.append(f"{title}\n{t[0]}" if title else t[0])
    return "\n\n".join([markdown, *parts]) if parts else None


def _is_form_page(raw_chandra: str, markdown: str, pdf_bytes: bytes, index: int, opts: dict) -> bool:
    """A page without math that Chandra tagged as a `Form`, or with dense checkboxes, or with AcroForm widgets."""
    if _MATH.search(raw_chandra):
        return False
    if _FORM_LABEL.search(raw_chandra) or len(_CHECKBOX.findall(markdown)) >= int(opts.get("min_checkboxes", 4)):
        return True
    if opts.get("wide") and _looks_like_form(markdown):  # the rewrite itself may still answer "not a form"
        return True
    from docparse.fixes.forms_acro import acroform_widgets

    return acroform_widgets(pdf_bytes, index)[0] >= int(opts.get("min_acro_widgets", 3))


_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", _FLAGS)
_ROW = re.compile(r"<tr\b", re.IGNORECASE)
_BLANK_RUN = re.compile(r"_{4,}|\.{6,}")


def _looks_like_form(markdown: str) -> bool:
    """Signs of a form in a reading that carries no `Form` block: a checkbox, runs of blanks to write on, or several
    one- or two-row tables (boxed fields)."""
    if _CHECKBOX.search(markdown) or len(_BLANK_RUN.findall(markdown)) >= 3:
        return True
    return sum(1 for t in _TABLE.findall(markdown) if len(_ROW.findall(t)) <= 2) >= 3


def _grid_table_count(markdown: str) -> int:
    """Data tables a form rewrite must keep: a header, three or more rows, no checkbox, and mostly filled cells
    (boxed fields and half-empty form grids drawn as tables may become `caption: value` lines)."""
    n = 0
    for t in _TABLE.findall(markdown):
        cells = _CELL.findall(t)
        empty = sum(1 for c in cells if not re.sub(r"<[^>]+>|&nbsp;|\s", "", c))
        grid = "<th" in t.lower() and len(_ROW.findall(t)) >= 3 and not _CHECKBOX.search(t)
        n += grid and empty < 0.25 * max(1, len(cells))
    return n


def _data_table_count(markdown: str) -> int:
    """HTML tables with `<th>` headers and no form checkboxes (true data tables, not `<table border=0>` form grids)."""
    return sum(1 for t in _TABLE.findall(markdown) if "<th" in t.lower() and not _CHECKBOX.search(t))


def _word_coverage(draft: str, candidate: str) -> float:
    """Share of the draft's words (3+ letters or digits, tags ignored) that `candidate` keeps; 1.0 without any."""
    d_set, c_set = (set(re.findall(r"[a-z0-9]{3,}", re.sub(r"<[^>]+>", " ", s).lower())) for s in (draft, candidate))
    return len(d_set & c_set) / len(d_set) if d_set else 1.0


def _line_key(line: str) -> str:
    return " ".join(re.sub(r"[#*_]", " ", html.unescape(re.sub(r"<[^>]+>", " ", line))).split()).casefold()


EDGE_LINES = 4  # lines at the top and at the bottom of a page where a running header or footer can stand


def _drop_chandra_furniture(raw_chandra: str, candidate_md: str, edges: bool = False) -> str:
    """`candidate_md` without the lines that repeat Chandra's Page-Header / Page-Footer text (a line the page body
    shows too is kept). With `edges`, one of the first or last `EDGE_LINES` lines also goes when its words are the
    furniture's words (85% of them) and cover most of one furniture block (60%)."""
    from docparse.models.adapters.vlm_specialists import _FURNITURE_DIV, chandra_markdown

    def keys(md: str) -> set[str]:
        return {_line_key(line) for line in md.splitlines()}

    blocks = [chandra_markdown(m[0]) for m in _FURNITURE_DIV.finditer(raw_chandra)]
    furniture = keys("\n".join(blocks)) | {_line_key(b) for b in blocks}
    furniture -= keys(chandra_markdown(raw_chandra, drop_furniture=True)) | {""}
    if not furniture:
        return candidate_md
    lines = candidate_md.splitlines()
    drop = {i for i, line in enumerate(lines) if _line_key(line) in furniture}
    if edges:  # another reader words a running header a little differently: match the page's edge lines loosely
        body = keys(chandra_markdown(raw_chandra, drop_furniture=True))
        blocks_tok = [t for t in (set(re.findall(r"[a-z0-9]+", k)) for k in furniture) if t]
        union = set().union(*blocks_tok) if blocks_tok else set()
        filled = [i for i, line in enumerate(lines) if line.strip()]
        for i in [*filled[:EDGE_LINES], *filled[-EDGE_LINES:]]:
            k = _line_key(lines[i])
            toks = set(re.findall(r"[a-z0-9]+", k))
            if i in drop or not toks or len(toks) > 30 or k in body:
                continue
            inside = len(toks & union) >= 0.85 * len(toks)
            covers = any(len(toks & b) >= 0.6 * len(b) for b in blocks_tok)
            if inside and covers:
                drop.add(i)
    kept = [line for i, line in enumerate(lines) if i not in drop]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _run_tier2_tables(
    markdown: str, rp: RenderedPage, truncated: bool, config: Config, client: ModelClient, meta: dict
) -> tuple[str, list[ModelResult], list[str]]:
    """`(markdown, results used, pending keys)` after the Tier-2 strip re-read of a dense table page.

    A tabular page without math is re-read as `n_strips` overlapping horizontal strips when its read was cut off
    (`truncated`) or, on a healthy text layer, holds under `min_recall_trigger` of the PDF's words. The stitched copy
    replaces it only when every strip is usable and it recovers PDF words: as many (a cut-off read) or
    `min_recall_gain` more; without a healthy text layer to check against, only in place of a cut-off read."""
    if not config.tier2_tables or _MATH.search(markdown):
        return markdown, [], []
    from docparse.fixes.layout_refine import words_in_points
    from docparse.fixes.verify import has_healthy_text_layer, pdf_looks_tabular, pdf_word_recall
    from docparse.models.adapters import qwen_tier2

    t_opts = config.tier2_table_options
    words = words_in_points(rp)  # the text layer in points: `pdf_looks_tabular` needs word geometry
    is_tabular = bool(_TABLE.search(markdown)) or (bool(words) and pdf_looks_tabular(words))
    healthy = has_healthy_text_layer(rp)
    rec_old = pdf_word_recall(markdown, words) if healthy else None
    low_recall = rec_old is not None and rec_old < float(t_opts.get("min_recall_trigger", 0.78))
    if not (is_tabular and (truncated or low_recall)):
        return markdown, [], []
    model = qwen_tier2.resolve_tier2_model(t_opts.get("model"))
    n_strips, overlap = int(t_opts.get("n_strips", 3)), float(t_opts.get("strip_overlap", 0.14))
    calls = [
        qwen_tier2.build_hard_table_call(
            strip,
            model=model,
            max_tokens=int(t_opts.get("max_tokens", 16384)),
            temperature=float(t_opts.get("temperature", 0.1)),
            top_p=float(t_opts.get("top_p", 0.95)),
            presence_penalty=float(t_opts.get("presence_penalty", 1.0)),
        )
        for strip in qwen_tier2.make_n_horizontal_strips(rp.image, n=n_strips, overlap=overlap)
    ]
    results = [client.get(call, blobs, meta) for call, blobs in calls]
    done = [r for r in results if not isinstance(r, Pending)]
    if len(done) < len(results):  # every strip is asked for in the same batch
        return markdown, [], list(dict.fromkeys(r.key for r in results if isinstance(r, Pending)))
    reads = [qwen_tier2.read_tier2_result(r) for r in done]
    if not all(x.usable for x in reads):
        return markdown, [], []
    merged = qwen_tier2.merge_n_strip_tables([x.markdown for x in reads])
    if not merged.strip():
        return markdown, [], []
    if rec_old is not None:
        rec_new = pdf_word_recall(merged, words)
        taken = (truncated and rec_new >= rec_old) or rec_new >= rec_old + float(t_opts.get("min_recall_gain", 0.05))
        if taken and not truncated:
            from docparse.fixes.verify import table_alignment_stats

            old_nt, _, _, old_found = table_alignment_stats(markdown, rp)
            if old_nt >= 1 and old_found >= 0.80:
                new_nt, _, _, new_found = table_alignment_stats(merged, rp)
                if new_nt < 1 or new_found < old_found - 0.12 or new_nt > old_nt + 1:
                    taken = False
    else:
        taken = truncated
    return (merged, done, []) if taken else (markdown, [], [])


def _layer_lines(rp: RenderedPage, opts: dict) -> str:
    """The text layer of a born-digital page as lines, top to bottom (words of a line separated by their gaps:
    a wide gap becomes ` | `); "" when the layer is unhealthy, absent, or `layer` is off."""
    if not opts.get("layer", True):
        return ""
    from docparse.fixes.layout_refine import lines_of, words_in_points
    from docparse.fixes.verify import has_healthy_text_layer

    if rp.hidden is None or not has_healthy_text_layer(rp):
        return ""
    out = []
    for line in lines_of(words_in_points(rp)):
        parts, prev = [], None
        for w in line:
            height = max(1.0, w.y1 - w.y0)
            if prev is not None:
                parts.append(" | " if w.x0 - prev.x1 > 1.5 * height else " ")
            parts.append(w.text)
            prev = w
        text = "".join(parts).strip()
        if text:
            out.append(text)
    return "\n".join(out)[: int(opts.get("layer_chars", 12000))]


def _fuse_other(markdown: str, other_md: str, rp: RenderedPage, options: dict) -> str:
    """The kept reading with the other reading's plain words wherever the healthy text layer sides with them."""
    if options.get("fuse_words"):
        from docparse.fixes.select import fuse_layer_words

        markdown, _ = fuse_layer_words(markdown, other_md, rp)
    return markdown


def _run_tier2_forms(
    markdown: str,
    raw_chandra: str,
    rp: RenderedPage,
    pdf_bytes: bytes,
    index: int,
    config: Config,
    client: ModelClient,
    meta: dict,
) -> tuple[str, list[ModelResult], list[str]]:
    """`(markdown, results used, pending keys)` after the Tier-2 form step on a form page: the draft is verified
    (`form-refine`) or rewritten as `caption: value` lines (`prompt: kv`) against the page image. The answer replaces
    the draft only when it keeps the draft's words and data tables; a thinking read that fails falls back to a
    plain one (`fallback_plain`). `raw_chandra` is "" when the draft comes from another reader."""
    f_opts = config.tier2_form_options
    if not _is_form_page(raw_chandra, markdown, pdf_bytes, index, f_opts):
        return markdown, [], []
    from functools import partial

    from docparse.models.adapters import qwen_tier2

    kv = f_opts.get("prompt") == "kv"
    if kv:  # rewrite as `caption: value` lines, with the text layer's lines
        build_form = partial(qwen_tier2.build_form_kv_call, layer_lines=_layer_lines(rp, f_opts))
    else:
        f_imgs = (
            list(qwen_tier2.make_horizontal_strips(rp.image, overlap=float(f_opts.get("strip_overlap", 0.12))))
            if f_opts.get("two_strips")
            else None
        )
        build_form = partial(
            qwen_tier2.build_form_refine_call,
            prompt_version=str(f_opts.get("prompt_version", "v1")),
            images=f_imgs,
        )
    samplings = [
        {
            "enable_thinking": bool(f_opts.get("enable_thinking", False)),
            "temperature": float(f_opts.get("temperature", 0.1)),
            "top_p": float(f_opts.get("top_p", 0.95)),
            "top_k": int(f_opts["top_k"]) if "top_k" in f_opts else None,
            "presence_penalty": float(f_opts.get("presence_penalty", 1.0)),
            "max_tokens": int(f_opts.get("max_tokens", 8192)),
        }
    ]
    if samplings[0]["enable_thinking"] and f_opts.get("fallback_plain"):
        # a thinking read that runs out of tokens or fails the guards falls back to a plain read
        samplings.append({"enable_thinking": False, "temperature": 0.1, "top_p": 0.95, "top_k": None,
                          "presence_penalty": 1.0, "max_tokens": 8192})  # fmt: skip
    for sampling in samplings:
        f_call, f_blobs = build_form(
            rp.image, markdown, model=qwen_tier2.resolve_tier2_model(f_opts.get("model")), **sampling
        )
        f_res = client.get(f_call, f_blobs, meta)
        if isinstance(f_res, Pending):
            return markdown, [], [f_res.key]
        f_read = qwen_tier2.read_tier2_result(f_res)
        if kv and f_read.markdown.strip().strip("`") == qwen_tier2.NOT_A_FORM:
            break  # the model looked and saw no form: the draft stands
        if not f_read.usable:  # a cut-off or looping answer leaves the draft
            continue
        cand_md = f_read.markdown
        if config.strip_furniture and raw_chandra:
            cand_md = _drop_chandra_furniture(raw_chandra, cand_md)
        # a refinement that drops the draft's printed text or data tables is a rewrite, not a correction
        min_cov = float(f_opts.get("min_word_coverage", 0.93))
        if kv:  # boxed fields drawn as small tables may become lines; real data grids must stay
            keeps_tables = _grid_table_count(cand_md) >= _grid_table_count(markdown)
        else:
            keeps_tables = _data_table_count(cand_md) >= _data_table_count(markdown)
        if keeps_tables and _word_coverage(markdown, cand_md) >= min_cov:
            return cand_md, [f_res], []
    return markdown, [], []


def parse(pdf_bytes: bytes, config: Config, client: ModelClient) -> ParseResult:
    """Parse a PDF (or a page image). Raises `ParseError` on unreadable bytes."""
    decorations = styles = False
    if config.format_options:
        from docparse.fixes.formatting import needs_decorations, needs_styles

        decorations, styles = needs_decorations(config.format_options), needs_styles(config.format_options)
    layer = bool(
        config.verify_pdf or config.tier2_tables or config.form_options.get("route_blanks") or config.select_options
    )
    if config.text_options:
        from docparse.fixes.text import needs_layer

        layer = layer or needs_layer(config.text_options)
    rendered = render_pages(pdf_bytes, config.dpi, decorations=decorations, styles=styles, layer=layer)
    doc_sha = sha256_hex(pdf_bytes)
    pages, pending = [], []
    use_chandra = config.reader in ("chandra", "chandra_hybrid")
    for index, stored in enumerate(rendered):
        rp, turn = stored, 0
        if config.upright_read:  # a page printed sideways or upside down is read, and corrected, upright
            from docparse.pdf.upright import page_turn, turn_page

            turn = page_turn(stored)
            rp = turn_page(stored, turn)
        call, blobs = build_page_call(rp.image, config.page_model, config.page_runtime, config.prompt_sha)
        meta = {"doc": doc_sha, "page": index, "dpi": rp.dpi}

        chandra_call = chandra_res = None
        if use_chandra:
            from docparse.models.adapters import vlm_specialists

            ch_model = config.chandra_model or vlm_specialists.CHANDRA2
            chandra_call, ch_blobs = vlm_specialists.build_chandra_call(
                rp.image, model=ch_model, runtime=config.page_runtime
            )
            chandra_res = client.get(chandra_call, ch_blobs, meta)
            if isinstance(chandra_res, Pending):
                pending.append(chandra_res.key)
                continue
            from docparse.fixes.fallback import looped

            ch_read = vlm_specialists.read_chandra_result(chandra_res, drop_furniture=config.strip_furniture)
            if config.text_options.get("picture_alt"):
                folded = vlm_specialists.fold_picture_text(ch_read.raw_answer)
                if folded != ch_read.raw_answer:  # what Chandra wrote about a picture belongs in its alt text
                    ch_read = vlm_specialists.SpecialistRead(
                        vlm_specialists.chandra_markdown(folded, drop_furniture=config.strip_furniture),
                        folded,
                        ch_read.finish_reason,
                    )
            ch_ok = (
                ch_read.finish_reason != "length" and bool(ch_read.markdown.strip()) and not looped(ch_read.markdown)
            )
            verify_needed, verify_reason = (False, "")
            full_ch_md = ""
            if (
                ch_ok
                and config.reader == "chandra_hybrid"
                and config.verify_pdf
                and not vlm_specialists.has_chart_figure(ch_read.raw_answer)
            ):
                from docparse.fixes.verify import needs_pdf_verification

                full_ch_md = (
                    vlm_specialists.chandra_markdown(ch_read.raw_answer, drop_furniture=False)
                    if config.strip_furniture
                    else ch_read.markdown
                )
                # table geometry routes every page but a form page (by the strict form trigger, unless
                # `wide_blocks_geo` lets the wider one decide)
                geo_form_opts = (
                    config.tier2_form_options
                    if config.tier2_form_options.get("wide_blocks_geo")
                    else {**config.tier2_form_options, "wide": False}
                )
                use_tgeo = bool(config.table_options.get("table_geo")) and not (
                    config.tier2_forms
                    and _is_form_page(ch_read.raw_answer, full_ch_md, pdf_bytes, index, geo_form_opts)
                )
                verify_needed, verify_reason = needs_pdf_verification(full_ch_md, rp, table_geo=use_tgeo)
            use_form_route = bool(
                ch_ok
                and config.reader == "chandra_hybrid"
                and config.form_options.get("route_blanks")
                and not vlm_specialists.has_chart_figure(ch_read.raw_answer)
                and not _MATH.search(ch_read.raw_answer)
            )
            if use_form_route and not full_ch_md:
                full_ch_md = (
                    vlm_specialists.chandra_markdown(ch_read.raw_answer, drop_furniture=False)
                    if config.strip_furniture
                    else ch_read.markdown
                )
            need_ocria = (
                not ch_ok
                or verify_needed
                or use_form_route
                or (config.reader == "chandra_hybrid" and vlm_specialists.has_chart_figure(ch_read.raw_answer))
                or config.layout == "ground"
                or bool(config.select_options)
            )
            result = client.get(call, blobs, meta) if need_ocria else None
        else:
            ch_read = None
            ch_ok = False
            verify_needed, verify_reason = (False, "")
            use_form_route = False
            result = client.get(call, blobs, meta)

        detections = regions2 = pp_res = None
        if config.layout == "ground":  # the detector reads the same page image, so it shares the blob
            detections = client.get(heron.detect_call(call.inputs[0]), blobs, meta)
        if config.layout != "none" and any(config.layout_options.get(k) for k in KDL_OPTIONS):  # second detector
            regions2 = client.get(kdl.layout_call(call.inputs[0]), blobs, meta)
        if config.layout != "none" and any(config.layout_options.get(k) for k in ("pp_arbiter", "pp_add")):
            pp_res = client.get(ppdoclayout.detect_call(call.inputs[0]), blobs, meta)
        pp_chart = None  # PP-DocLayoutV3 finds the charts Chandra misses (sharing the layout's call)
        if config.tier2_charts and ch_ok and config.tier2_chart_options.get("use_ppdoclayout", True):
            pp_chart = pp_res
            if pp_chart is None:
                pp_chart = client.get(ppdoclayout.detect_call(call.inputs[0]), blobs, meta)
        waiting = list(
            dict.fromkeys(r.key for r in (result, detections, regions2, pp_res, pp_chart) if isinstance(r, Pending))
        )
        if waiting:
            pending += waiting
            continue

        read = read_page_result(result) if result is not None else None
        if read is not None and config.reread_truncated and read.finish_reason == "length":
            call2, _ = build_page_call(
                rp.image, config.page_model, config.page_runtime, config.prompt_sha, dict(config.reread_truncated)
            )
            second = client.get(call2, blobs, meta)
            if isinstance(second, Pending):
                pending.append(second.key)
                continue
            again = read_page_result(second)
            if again.finish_reason != "length" and again.markdown.strip():
                read, result = again, second

        attention = None
        if config.layout == "ground" and config.layout_attention != "never" and read is not None:
            from docparse.fixes.layout import has_text_layer

            # asked for only once the read exists: the answer is part of the call
            if config.layout_attention == "always" or not has_text_layer(rp):
                answer = read.raw_answer or read.markdown
                att_call, answer_blob = attention_call(call.inputs[0], answer, config.page_model, config.prompt_sha)
                attention = client.get(att_call, ChainMap(answer_blob, blobs), meta)
                if isinstance(attention, Pending):
                    pending.append(attention.key)
                    continue
                attention = read_attention(attention)

        sec_md_cached: str | None = None
        routed_form = False
        selected_alt = False
        picture_md = ""
        if use_chandra and ch_ok and ch_read is not None and verify_needed and read is not None:
            from docparse.fixes.verify import secondary_improves

            sec_md_cached, _, _ = _ocria_markdown(pdf_bytes, index, rp, read, call.key(), config)
            if secondary_improves(
                full_ch_md or ch_read.markdown,
                sec_md_cached,
                rp,
                verify_reason,
                table_geo=bool(config.table_options.get("table_geo")),
            ):
                ch_ok = False
        if (
            use_chandra
            and ch_ok
            and ch_read is not None
            and read is not None
            and config.select_options
            and read.finish_reason != "length"
            and not (config.select_options.get("skip_math") and _ANY_MATH.search(ch_read.raw_answer + read.markdown))
        ):
            from docparse.fixes.select import prefer_alternative
            from docparse.models.adapters import vlm_specialists

            ch_cmp = full_ch_md or ch_read.markdown
            pictured = _picture_words(ch_read.raw_answer) >= int(config.select_options.get("picture_words", 10**9))
            is_tgeo = verify_reason.startswith("table_geo:")
            skip_tbl = (bool(config.select_options.get("skip_tables")) and _data_table_count(ch_cmp) > 0) or (
                bool(config.select_options.get("respect_table_geo")) and is_tgeo
            )
            if not vlm_specialists.has_chart_figure(ch_read.raw_answer) and not pictured and not skip_tbl:
                if sec_md_cached is None:
                    sec_md_cached, _, _ = _ocria_markdown(pdf_bytes, index, rp, read, call.key(), config)
                sel_opts = (
                    {**config.select_options, "layer_margin": float(config.select_options["table_margin"])}
                    if is_tgeo and "table_margin" in config.select_options
                    else config.select_options
                )
                if prefer_alternative(ch_read.markdown, sec_md_cached, rp, sel_opts):
                    ch_ok = False
                    selected_alt = True
        if use_chandra and ch_ok and ch_read is not None and use_form_route and read is not None:
            from docparse.fixes.fallback import has_math
            from docparse.fixes.layout_refine import words_in_points
            from docparse.fixes.verify import (
                TABLE_MAX_MAP_ERRORS,
                TABLE_MIN_WORDS,
                has_healthy_text_layer,
                pdf_word_recall,
            )

            pp_min = float(config.tier2_chart_options.get("pp_chart_score", 0.40))
            pp_has_chart = pp_chart is not None and any(
                d.label == "chart" for d in ppdoclayout.read_raw_detections(pp_chart, pp_min)
            )
            ch_cmp = full_ch_md or ch_read.markdown
            if not pp_has_chart and not has_math(ch_cmp) and _data_table_count(ch_cmp) == 0:
                if sec_md_cached is None:
                    sec_md_cached, _, _ = _ocria_markdown(pdf_bytes, index, rp, read, call.key(), config)
                healthy = has_healthy_text_layer(rp, min_words=TABLE_MIN_WORDS, max_map_errors=TABLE_MAX_MAP_ERRORS)
                words = words_in_points(rp) if healthy else []
                ch_rec = pdf_word_recall(ch_cmp, words) if healthy else 0.0
                oc_rec = pdf_word_recall(sec_md_cached, words) if healthy else 0.0
                oc_bl = len(_BLANK_MARK.findall(sec_md_cached))
                ch_bl = len(_BLANK_MARK.findall(ch_cmp))
                if oc_bl >= 3 and ch_bl == 0 and (not healthy or oc_rec >= ch_rec - 0.03):
                    ch_ok = False
                    routed_form = True
        if (
            use_chandra
            and not ch_ok
            and ch_read is not None
            and ch_read.finish_reason != "length"
            and config.select_options.get("keep_pictures")
        ):
            picture_md = _picture_notes(ch_read.raw_answer, int(config.select_options["keep_pictures"]))
        if use_chandra and ch_ok and ch_read is not None:
            from docparse.models.adapters import vlm_specialists

            assert chandra_res is not None
            markdown = ch_read.markdown
            used: list[ModelResult] = [chandra_res]  # the calls whose answers the page shows
            ocria_tables: list[str] = []
            ch_has_chart = vlm_specialists.has_chart_figure(ch_read.raw_answer)
            if config.reader == "chandra_hybrid" and read is not None and ch_has_chart:
                _, ocria_tables, vec_measured = _ocria_markdown(pdf_bytes, index, rp, read, call.key(), config)
                markdown, fused = vlm_specialists.fuse_figure_charts(
                    ch_read.raw_answer,
                    ocria_tables,
                    ocria_vector_measured=vec_measured,
                    drop_furniture=config.strip_furniture,
                )
                if fused:
                    used.append(result)
            tier2_chart_used = tier2_form_used = tier2_tried = False
            if config.tier2_charts:
                from docparse.models.adapters import qwen_tier2

                opts = config.tier2_chart_options
                pp_boxes: list[tuple[float, ...]] = []
                if pp_chart is not None:
                    pp_min = float(opts.get("pp_chart_score", 0.40))
                    pp_boxes = _dedup_chart_boxes(
                        [d.box for d in ppdoclayout.read_raw_detections(pp_chart, pp_min) if d.label == "chart"]
                    )
                if ch_has_chart or pp_boxes:  # a chart Chandra drew as a Figure, or one only PP-DocLayoutV3 saw
                    tier2_tried = True
                    chart_model = qwen_tier2.resolve_tier2_model(opts.get("model"))
                    per_chart = (
                        bool(opts.get("per_chart_crops"))
                        and 2 <= len(pp_boxes) <= int(opts.get("per_chart_max_boxes", 6))
                        and _union_share(rp, pp_boxes) >= float(opts.get("per_chart_min_union", 0.32))
                    )
                    chart_drafts = ocria_tables
                    if opts.get("draft_format") and ocria_tables:
                        # the chart model's draft tables under other format options than the page's own
                        d_cfg = config.model_copy(
                            update={"format_options": {**config.format_options, **opts["draft_format"]}}
                        )
                        _, chart_drafts, _ = _ocria_markdown(pdf_bytes, index, rp, read, call.key(), d_cfg)
                    draft = "\n\n".join(chart_drafts or _TABLE.findall(ch_read.raw_answer))

                    def chart_calls(
                        o: dict,
                        per_chart=per_chart,
                        index=index,
                        rp=rp,
                        turn=turn,
                        draft=draft,
                        chart_model=chart_model,
                        raw=ch_read.raw_answer,
                        pp_boxes=pp_boxes,
                        ocria_tables=chart_drafts,
                    ) -> list:
                        if per_chart:  # several large charts: one zoomed read of each, all asked for at once
                            return [
                                _tier2_single_chart_call(pdf_bytes, index, rp, turn, bx, draft, o, chart_model)
                                for bx in pp_boxes
                            ]
                        return [
                            _tier2_chart_call(pdf_bytes, index, rp, turn, raw, pp_boxes, ocria_tables, o, chart_model)
                        ]

                    t2_calls = chart_calls(opts)
                    t2_results = [client.get(c, b, meta) for c, b in t2_calls]
                    t2_done = [r for r in t2_results if not isinstance(r, Pending)]
                    if len(t2_done) < len(t2_results):
                        pending += list(dict.fromkeys(r.key for r in t2_results if isinstance(r, Pending)))
                        continue
                    t2_reads = [(r, qwen_tier2.read_tier2_result(r)) for r in t2_done]
                    if opts.get("fallback_plain") and opts.get("enable_thinking"):
                        # a thinking read that ran out of tokens, looped or wrote no table is read again plainly
                        bad = [i for i, (_, x) in enumerate(t2_reads) if not (x.usable and _TABLE.search(x.markdown))]
                        if bad:
                            plain = {**opts, "enable_thinking": False, "temperature": 0.1, "max_tokens": 10240}
                            plain.pop("top_k", None)
                            again = chart_calls(plain)
                            retried = [client.get(*again[i], meta) for i in bad]
                            if any(isinstance(r, Pending) for r in retried):
                                pending += list(dict.fromkeys(r.key for r in retried if isinstance(r, Pending)))
                                continue
                            for i, r in zip(bad, retried, strict=True):
                                t2_reads[i] = (r, qwen_tier2.read_tier2_result(r))
                    extra_reads: list[list] = [[] for _ in t2_reads]
                    n_samples = int(opts.get("samples", 1))
                    if n_samples > 1:  # more independent reads of each chart: the numbers are voted cell by cell
                        waiting_more = False
                        for seed in range(2, n_samples + 1):
                            more = [client.get(c, b, meta) for c, b in chart_calls({**opts, "seed": seed})]
                            if any(isinstance(r, Pending) for r in more):
                                pending += list(dict.fromkeys(r.key for r in more if isinstance(r, Pending)))
                                waiting_more = True
                                continue
                            for slot, r in zip(extra_reads, more, strict=True):
                                slot.append(qwen_tier2.read_tier2_result(r))
                        if waiting_more:
                            continue
                    # a cut-off or looping answer, or one without a table, leaves the page as read without it
                    t2_good = []
                    for (r, x), others in zip(t2_reads, extra_reads, strict=True):
                        if not (x.usable and _TABLE.search(x.markdown)):
                            continue
                        peers = [o.markdown for o in others if o.usable and _TABLE.search(o.markdown)]
                        if peers:
                            from dataclasses import replace

                            from docparse.fixes.chart_vote import fuse_reads

                            x = replace(x, markdown=fuse_reads([x.markdown, *peers]))
                        t2_good.append((r, x))
                    if per_chart and opts.get("all_crops", True) and len(t2_good) < len(t2_calls):
                        t2_good = []  # one chart of several unread: the page keeps its own reading of them all
                    t2_markdown = None
                    if t2_good:
                        answer = "\n\n".join(x.markdown for _, x in t2_good)
                        t2_markdown = _tier2_markdown(
                            ch_read.raw_answer,
                            markdown,
                            answer,
                            ch_has_chart,
                            config.strip_furniture,
                            table_figures=bool(opts.get("table_figures", True)),
                        )
                    if t2_markdown is not None:
                        markdown = t2_markdown
                        used += [r for r, _ in t2_good]
                        tier2_chart_used = True
            # Tier-2 forms, then tables (or tables first with `before_forms`); a page already escalated to Tier-2
            # (a form page, or a table re-read that was kept) is never escalated twice
            t2_wait: list[str] = []
            for step in ("tables", "forms") if config.tier2_table_options.get("before_forms") else ("forms", "tables"):
                if tier2_tried or t2_wait:
                    break
                if step == "tables":  # recover truncated / low-recall data tables
                    markdown, t_used, t2_wait = _run_tier2_tables(markdown, rp, False, config, client, meta)
                    if t_used and not t2_wait:
                        if config.strip_furniture:
                            markdown = _drop_chandra_furniture(ch_read.raw_answer, markdown)
                        used += t_used
                        tier2_tried = True
                elif config.tier2_forms:
                    fired = _is_form_page(ch_read.raw_answer, markdown, pdf_bytes, index, config.tier2_form_options)
                    markdown, f_used, t2_wait = _run_tier2_forms(
                        markdown, ch_read.raw_answer, rp, pdf_bytes, index, config, client, meta
                    )
                    used += f_used
                    tier2_form_used = bool(f_used)
                    tier2_tried = fired
            if t2_wait:
                pending += t2_wait
                continue
            if (
                not tier2_chart_used
                and not tier2_form_used
                and read is not None
                and read.finish_reason != "length"
                and config.select_options.get("fuse_words")
            ):
                other_md = (
                    sec_md_cached
                    if sec_md_cached is not None
                    else _ocria_markdown(pdf_bytes, index, rp, read, call.key(), config)[0]
                )
                markdown = _fuse_other(markdown, other_md, rp, config.select_options)
            # Run the same PDF-grounded corrector stack (AcroForm widgets, text respell, formatting, tables, charts)
            if config.form_options:
                from docparse.fixes.forms import repair
                from docparse.pdf.overlay import overlay_fonts

                markdown = repair(markdown, rp, overlay_fonts(pdf_bytes, index), config.form_options, pdf_bytes, index)
            if config.text_options.get("empty_read"):
                from docparse.fixes.text import empty_read

                markdown = empty_read(markdown, rp) or markdown
            page = parse_page(markdown, index=index, call_key=chandra_call.key())
            if config.text_options:
                from docparse.fixes.text import refine_text

                refine_text(page, rp, config.text_options)
            if config.format_options:
                from docparse.fixes.formatting import refine_format

                refine_format(page, rp, config.format_options)
            page.width, page.height, page.rotation = stored.width_pt, stored.height_pt, stored.rotation
            if config.table_options:
                from docparse.fixes.tables import fix_tables

                fix_tables(page, rp, config.table_options)
            if (
                config.chart_options.get("measure")
                and (not tier2_chart_used or config.chart_options.get("after_tier2"))
                and any(r.kind == "table" for r in page.regions)
            ):
                from docparse.fixes.charts import correct_tables
                from docparse.fixes.layout_refine import words_in_points
                from docparse.pdf.graphics import load_shapes

                try:
                    correct_tables(page, words_in_points(rp), load_shapes(pdf_bytes, index), config.chart_options)
                except Exception as exc:
                    for region in page.regions:
                        region.selected = region.candidates[0].id
                    page.findings.append(Finding(check="charts.vector", evidence=f"{type(exc).__name__}: {exc}"))
            page.cost = CostRecord(
                calls=len(used),
                gpu_seconds=sum(r.gpu_seconds or 0.0 for r in used),
                tokens_in=sum(r.tokens_in or 0 for r in used),
                tokens_out=sum(r.tokens_out or 0 for r in used),
            )
        else:
            assert read is not None and result is not None
            markdown = read.markdown
            if use_chandra and config.strip_furniture and ch_read is not None and ch_read.finish_reason != "length":
                markdown = _drop_chandra_furniture(ch_read.raw_answer, markdown, edges=True)
            if picture_md:
                markdown = f"{markdown}\n\n{picture_md}"
            # Run Tier-2 strip recovery BEFORE plain-text fallback so a cut-off table keeps HTML structure
            t_used, t_wait = [], []
            raw_ch = ch_read.raw_answer if ch_read is not None else ""
            if not routed_form and not selected_alt:
                markdown, t_used, t_wait = _run_tier2_tables(
                    markdown, rp, read.finish_reason == "length", config, client, meta
                )
            if t_wait:
                pending += t_wait
                continue
            if t_used and use_chandra and config.strip_furniture and ch_read is not None:
                markdown = _drop_chandra_furniture(ch_read.raw_answer, markdown, edges=True)
            f_used: list[ModelResult] = []
            if config.tier2_forms and not t_used and config.tier2_form_options.get("any_reader"):
                markdown, f_used, f_wait = _run_tier2_forms(
                    markdown, raw_ch, rp, pdf_bytes, index, config, client, meta
                )
                if f_wait:
                    pending += f_wait
                    continue
            if not t_used and not f_used and config.fallback_below > 0:
                from docparse.fixes.fallback import (
                    MIN_TOKENS,
                    has_math,
                    looped,
                    reading_recall,
                    text_dump,
                    trim_loop_tail,
                )
                from docparse.fixes.layout_refine import words_in_points

                cut_short = read.finish_reason == "length" or looped(markdown)
                if cut_short:
                    markdown = trim_loop_tail(markdown)
                    words = words_in_points(rp)
                    recall = reading_recall(markdown, words) if len(words) >= MIN_TOKENS else 1.0
                    # Never overwrite valid LaTeX math with a raw plain-text PDF dump
                    keep_math = has_math(markdown) and recall >= 0.20
                    if len(words) >= MIN_TOKENS and recall < config.fallback_below and not keep_math:
                        markdown = text_dump(words)
            if (
                not t_used
                and not f_used
                and ch_read is not None
                and ch_read.finish_reason != "length"
                and config.select_options.get("fuse_words")
            ):
                markdown = _fuse_other(markdown, ch_read.markdown, rp, config.select_options)
            if config.form_options:
                from docparse.fixes.forms import repair
                from docparse.pdf.overlay import overlay_fonts

                markdown = repair(markdown, rp, overlay_fonts(pdf_bytes, index), config.form_options, pdf_bytes, index)
            if config.text_options.get("empty_read"):
                from docparse.fixes.text import empty_read

                markdown = empty_read(markdown, rp) or markdown
            page = parse_page(markdown, index=index, call_key=call.key())
            if config.text_options:
                from docparse.fixes.text import refine_text

                refine_text(page, rp, config.text_options)
            if config.format_options:
                from docparse.fixes.formatting import refine_format

                refine_format(page, rp, config.format_options)
            page.width, page.height, page.rotation = stored.width_pt, stored.height_pt, stored.rotation
            if config.table_options:
                from docparse.fixes.tables import fix_tables

                fix_tables(page, rp, config.table_options)
            if config.chart_options.get("measure") and any(r.kind == "table" for r in page.regions):
                from docparse.fixes.charts import correct_tables
                from docparse.fixes.layout_refine import words_in_points
                from docparse.pdf.graphics import load_shapes

                try:
                    correct_tables(page, words_in_points(rp), load_shapes(pdf_bytes, index), config.chart_options)
                except Exception as exc:  # unusual geometry must never cost the page: keep the reading, record why
                    for region in page.regions:
                        region.selected = region.candidates[0].id
                    page.findings.append(Finding(check="charts.vector", evidence=f"{type(exc).__name__}: {exc}"))
            billed = [result, *t_used, *f_used]
            page.cost = CostRecord(
                calls=len(billed),
                gpu_seconds=sum(r.gpu_seconds or 0.0 for r in billed),
                tokens_in=sum(r.tokens_in or 0 for r in billed),
                tokens_out=sum(r.tokens_out or 0 for r in billed),
            )
        if config.layout == "from_page_read" and read is not None and read.layout_page is not None:
            page.layout = items_from_layout_page(read.layout_page)
        elif config.layout == "ground" and read is not None:
            from docparse.fixes.layout import ground

            # OCRia-1's grounding reads the raw answer; published stand-ins only have the scored markdown
            page.layout = ground(read.raw_answer or read.markdown, rp, heron.read_detections(detections), attention)
        chandra_regions = None
        chandra_texts = None
        if ch_read is not None and any(
            config.layout_options.get(k) for k in ("chandra_arbiter", "chandra_add", "chandra_fallback")
        ):
            from docparse.core.ir import Box
            from docparse.fixes.layout_kdl import PtRegion
            from docparse.models.adapters import vlm_specialists

            blocks = vlm_specialists.chandra_layout_blocks(ch_read.raw_answer, rp.width_pt, rp.height_pt)
            chandra_regions = [PtRegion(lbl, Box(x0=b[0], y0=b[1], x1=b[2], y1=b[3])) for lbl, b, _ in blocks]
            chandra_texts = [md_txt for _, _, md_txt in blocks]
        if regions2 is not None and page.layout is not None:
            from docparse.fixes.layout_kdl import apply_kdl

            page.layout = apply_kdl(
                page.layout, rp, kdl.read_detections(regions2), config.layout_options, markdown, chandra_regions
            )
        if pp_res is not None and page.layout is not None:
            from docparse.fixes.layout_kdl import apply_kdl

            pp_min = float(config.layout_options.get("pp_min_score", 0.40))
            pp_opts = {
                "kdl_arbiter": bool(config.layout_options.get("pp_arbiter")),
                "kdl_keep_pictures": bool(config.layout_options.get("pp_keep_pictures")),
                "kdl_add": bool(config.layout_options.get("pp_add")),
                "scan_arbiter": bool(config.layout_options.get("pp_scan_arbiter")),
            }
            page.layout = apply_kdl(
                page.layout, rp, ppdoclayout.read_detections(pp_res, min_score=pp_min), pp_opts, markdown
            )
        if chandra_regions and page.layout is not None:
            from docparse.fixes.layout_kdl import apply_chandra_layout

            page.layout = apply_chandra_layout(page.layout, rp, chandra_regions, config.layout_options, chandra_texts)
        if config.layout_options and page.layout and read is not None:
            from docparse.fixes.layout_refine import refine

            page.layout = refine(page.layout, rp, config.layout_options, read.markdown)
        if turn and page.layout:  # boxes were found on the upright page; report them on the page as stored
            from docparse.pdf.upright import unturn_box

            page.layout = [
                item.model_copy(update={"bbox": unturn_box(item.bbox, turn, rp.width_pt, rp.height_pt)})
                for item in page.layout
            ]
        pages.append(page)
    if pending:
        return ParseResult(document=None, pending=pending)
    document = Document(
        sha256=doc_sha, n_pages=len(pages), pages=pages, config_hash=config.config_hash(), code_version=__version__
    )
    return ParseResult(document=document)

"""The parser's only public entry point: `parse(pdf_bytes, config, client)`.

It receives bytes and nothing else: no file name, path, example id or subset (design §3, anti-goals).
A pure function of (bytes, config, cache): model answers come from the client, which reads the cache.

    S0 render -> S2 OCRia-1 page read (one cached call per page) -> S3 answer parsed into the IR
    -> S4 correctors, each behind its options (layout_options, text_options, format_options) -> writers
"""

from __future__ import annotations

from collections import ChainMap
from dataclasses import dataclass, field

from docparse import __version__
from docparse.base.ocria_parse import parse_page
from docparse.core.config import Config
from docparse.core.hashing import sha256_hex
from docparse.core.ir import CostRecord, Document, Finding
from docparse.fixes.layout_kdl import KDL_OPTIONS
from docparse.models.adapters import heron, kdl
from docparse.models.adapters.attention import attention_call, read_attention
from docparse.models.adapters.ocria import build_page_call, read_page_result
from docparse.models.calls import ModelClient, Pending
from docparse.pdf.render import render_pages
from docparse.write.layout_items import items_from_layout_page


@dataclass
class ParseResult:
    document: Document | None  # None while any model call is still pending
    pending: list[str] = field(default_factory=list)  # keys of the calls a GPU session has to run


def parse(pdf_bytes: bytes, config: Config, client: ModelClient) -> ParseResult:
    """Parse a PDF (or a page image). Raises `ParseError` on unreadable bytes."""
    decorations = styles = False
    if config.format_options:
        from docparse.fixes.formatting import needs_decorations, needs_styles

        decorations, styles = needs_decorations(config.format_options), needs_styles(config.format_options)
    layer = False
    if config.text_options:
        from docparse.fixes.text import needs_layer

        layer = needs_layer(config.text_options)
    rendered = render_pages(pdf_bytes, config.dpi, decorations=decorations, styles=styles, layer=layer)
    doc_sha = sha256_hex(pdf_bytes)
    pages, pending = [], []
    for index, stored in enumerate(rendered):
        rp, turn = stored, 0
        if config.upright_read:  # a page printed sideways or upside down is read, and corrected, upright
            from docparse.pdf.upright import page_turn, turn_page

            turn = page_turn(stored)
            rp = turn_page(stored, turn)
        call, blobs = build_page_call(rp.image, config.page_model, config.page_runtime, config.prompt_sha)
        meta = {"doc": doc_sha, "page": index, "dpi": rp.dpi}
        result = client.get(call, blobs, meta)
        detections = regions2 = None
        if config.layout == "ground":  # the detector reads the same page image, so it shares the blob
            detections = client.get(heron.detect_call(call.inputs[0]), blobs, meta)
        if config.layout != "none" and any(config.layout_options.get(k) for k in KDL_OPTIONS):  # second detector
            regions2 = client.get(kdl.layout_call(call.inputs[0]), blobs, meta)
        waiting = [r.key for r in (result, detections, regions2) if isinstance(r, Pending)]
        if waiting:
            pending += waiting
            continue
        read = read_page_result(result)
        if config.reread_truncated and read.finish_reason == "length":
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
        if config.layout == "ground" and config.layout_attention != "never":
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
        markdown = read.markdown
        if config.fallback_below > 0:
            from docparse.fixes.fallback import MIN_TOKENS, looped, reading_recall, text_dump
            from docparse.fixes.layout_refine import words_in_points

            words = words_in_points(rp)
            cut_short = read.finish_reason == "length" or looped(markdown)
            if cut_short and len(words) >= MIN_TOKENS and reading_recall(markdown, words) < config.fallback_below:
                markdown = text_dump(words)
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
        page.cost = CostRecord(
            calls=1,
            gpu_seconds=result.gpu_seconds or 0.0,
            tokens_in=result.tokens_in or 0,
            tokens_out=result.tokens_out or 0,
        )
        if config.layout == "from_page_read" and read.layout_page is not None:
            page.layout = items_from_layout_page(read.layout_page)
        elif config.layout == "ground":
            from docparse.fixes.layout import ground

            # OCRia-1's grounding reads the raw answer; published stand-ins only have the scored markdown
            page.layout = ground(read.raw_answer or read.markdown, rp, heron.read_detections(detections), attention)
        if regions2 is not None and page.layout is not None:
            from docparse.fixes.layout_kdl import apply_kdl

            page.layout = apply_kdl(page.layout, rp, kdl.read_detections(regions2), config.layout_options, markdown)
        if config.layout_options and page.layout:
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

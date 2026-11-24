"""The parser's only public entry point: `parse(pdf_bytes, config, client)`.

It receives bytes and nothing else: no file name, path, example id or subset (design §3, anti-goals).
A pure function of (bytes, config, cache): model answers come from the client, which reads the cache.

    S0 render -> S2 OCRia-1 page read (one cached call per page) -> S3 answer parsed into the IR
    -> S4 correctors (none yet; each will sit behind a `config.fixes` flag) -> writers
"""

from __future__ import annotations

from dataclasses import dataclass, field

from docparse import __version__
from docparse.base.ocria_parse import parse_page
from docparse.core.config import Config
from docparse.core.hashing import sha256_hex
from docparse.core.ir import CostRecord, Document
from docparse.models.adapters import heron
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
    rendered = render_pages(pdf_bytes, config.dpi)
    doc_sha = sha256_hex(pdf_bytes)
    pages, pending = [], []
    for index, rp in enumerate(rendered):
        call, blobs = build_page_call(rp.image, config.page_model, config.page_runtime, config.prompt_sha)
        meta = {"doc": doc_sha, "page": index, "dpi": rp.dpi}
        result = client.get(call, blobs, meta)
        detections = None
        if config.layout == "ground":  # the detector reads the same page image, so it shares the blob
            detections = client.get(heron.detect_call(call.inputs[0]), blobs, meta)
        waiting = [r.key for r in (result, detections) if isinstance(r, Pending)]
        if waiting:
            pending += waiting
            continue
        read = read_page_result(result)
        page = parse_page(read.markdown, index=index, call_key=call.key())
        page.width, page.height, page.rotation = rp.width_pt, rp.height_pt, rp.rotation
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
            page.layout = ground(read.raw_answer or read.markdown, rp, heron.read_detections(detections))
        if config.layout_options and page.layout:
            from docparse.fixes.layout_refine import refine

            page.layout = refine(page.layout, rp, config.layout_options)
        pages.append(page)
    if pending:
        return ParseResult(document=None, pending=pending)
    document = Document(
        sha256=doc_sha, n_pages=len(pages), pages=pages, config_hash=config.config_hash(), code_version=__version__
    )
    return ParseResult(document=document)

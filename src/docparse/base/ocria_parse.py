"""OCRia-1's (post-processed) markdown -> blocks -> an IR page.

The split is lossless: blocks cover every non-whitespace character, and the whitespace between them is kept
as separators, so `write_page(parse_page(md)) == md` for any string.

Block rules:
- a ``` fence line opens a code block that ends at the next fence line of the same length or longer;
- `<table` opens a table block that ends at its matching `</table>` (nested tables counted);
- a fence or table that never closes becomes one opaque block running to the end of the text
  (a truncated answer must still parse and write);
- everything else splits on blank-line runs, and a `#` heading line is always its own block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docparse.core.ir import Candidate, OpaqueContent, Page, Provenance, Region, TextContent

_OPENER = re.compile(r"(?m)^[ \t]{0,3}(`{3,})[^\n`]*$|<table\b", re.IGNORECASE)
_TABLE_TAG = re.compile(r"<table\b|</table\s*>", re.IGNORECASE)
_BLANK_RUN = re.compile(r"\n[ \t]*\n\s*")
_HEADING = re.compile(r"[ \t]{0,3}#{1,6}(?:[ \t]|$)")
_LIST_LINE = re.compile(r"[ \t]*(?:\\?[*+-]|\d{1,9}[.)])[ \t]+\S")
_IMAGE_REF = re.compile(r"!\[[^\]\n]*\](?:\([^)\n]*\))?")
BASE_READER = "ocria.base"


@dataclass(frozen=True)
class Block:
    kind: str
    raw: str
    sep_after: str
    start: int
    reason: str = ""


def _table_end(md: str, start: int) -> int | None:
    depth = 0
    for m in _TABLE_TAG.finditer(md, start):
        if m.group(0)[1] == "/":
            depth -= 1
            if depth == 0:
                return m.end()
        else:
            depth += 1
    return None


def _fence_end(md: str, opener: re.Match) -> int | None:
    ticks = len(opener.group(1))
    closer = re.compile(rf"(?m)^[ \t]{{0,3}}`{{{ticks},}}[ \t]*$")
    m = closer.search(md, opener.end())
    return m.end() if m else None


def _text_kind(raw: str) -> str:
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    if len(lines) == 1 and _HEADING.match(lines[0]):
        return "heading"
    if _IMAGE_REF.fullmatch(raw.strip()):
        return "image_ref"
    if lines and all(_LIST_LINE.match(ln) for ln in lines):
        return "list"
    return "paragraph"


def _text_spans(md: str, a: int, b: int) -> list[tuple[int, int, str, str]]:
    """Non-blank pieces of md[a:b], split on blank-line runs and around heading lines."""
    out: list[tuple[int, int, str, str]] = []
    pos = a
    bounds: list[tuple[int, int]] = []
    for m in _BLANK_RUN.finditer(md, a, b):
        bounds.append((pos, m.start()))
        pos = m.end()
    bounds.append((pos, b))
    for s, e in bounds:
        run_start: int | None = None
        line_start = s
        while line_start < e:
            nl = md.find("\n", line_start, e)
            line_end = e if nl == -1 else nl
            line = md[line_start:line_end]
            if _HEADING.match(line) and line.strip():
                if run_start is not None:
                    out.append((run_start, line_start, "", ""))
                    run_start = None
                out.append((line_start, line_end, "", ""))
            elif run_start is None and line.strip():
                run_start = line_start
            line_start = line_end + 1
        if run_start is not None:
            out.append((run_start, e, "", ""))
    trimmed = []
    for s, e, _k, _r in out:
        while s < e and md[s].isspace():
            s += 1
        while e > s and md[e - 1].isspace():
            e -= 1
        if e > s:
            trimmed.append((s, e, _text_kind(md[s:e]), ""))
    return trimmed


def split_blocks(md: str) -> tuple[str, list[Block]]:
    """(leading whitespace, blocks). Concatenating leading + raw + sep_after for every block gives `md` back."""
    spans: list[tuple[int, int, str, str]] = []
    pos = 0
    n = len(md)
    while pos < n:
        m = _OPENER.search(md, pos)
        if m is None:
            spans += _text_spans(md, pos, n)
            break
        spans += _text_spans(md, pos, m.start())
        if m.group(1):  # a fence
            start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip(" \t")))
            end = _fence_end(md, m)
            kind, reason = ("code", "") if end is not None else ("opaque", "unclosed code fence")
        else:
            start = m.start()
            end = _table_end(md, start)
            kind, reason = ("table", "") if end is not None else ("opaque", "unclosed table")
        if end is None:
            end = n
            while end > start and md[end - 1].isspace():
                end -= 1
        spans.append((start, end, kind, reason))
        pos = end
    if not spans:
        return md, []
    blocks = []
    for i, (s, e, kind, reason) in enumerate(spans):
        nxt = spans[i + 1][0] if i + 1 < len(spans) else n
        blocks.append(Block(kind=kind, raw=md[s:e], sep_after=md[e:nxt], start=s, reason=reason))
    return md[: spans[0][0]], blocks


def parse_page(md: str, index: int = 0, call_key: str = "") -> Page:
    """One region per block, each with OCRia-1's reading as its selected base candidate."""
    leading, blocks = split_blocks(md)
    regions = []
    for order, b in enumerate(blocks):
        rid = f"p{index}r{order}"
        if b.kind == "opaque":
            content = OpaqueContent(reason=b.reason)
        elif b.kind in ("heading", "paragraph", "list", "code"):
            role = {"heading": "heading", "list": "list_item", "code": "code"}.get(b.kind, "paragraph")
            level = len(b.raw.lstrip(" \t")) - len(b.raw.lstrip(" \t").lstrip("#")) if b.kind == "heading" else None
            content = TextContent(role=role, heading_level=level)
        else:
            content = None  # tables and image references get structured content from the fix that needs it
        cand = Candidate(
            id=f"{rid}c0",
            reader=BASE_READER,
            raw=b.raw,
            content=content,
            prov=[Provenance(source="model", ref=call_key)],
        )
        regions.append(
            Region(id=rid, kind=b.kind, order=order, candidates=[cand], selected=cand.id, sep_after=b.sep_after)
        )
    return Page(index=index, leading=leading, regions=regions)

"""Formatting corrections of the page reading from the PDF itself (design §6, formatting). Each is one option:

    cell_strike  inside an HTML table, `~~text~~` becomes `<s>text</s>`. Markdown is not read inside HTML blocks,
                 so a strike the model wrote with tildes in a cell shows as literal tildes, not as struck text
    strike       words the PDF draws a line through (pdf/drawings.py) are struck in the reading: `~~...~~` in
                 prose, `<s>...</s>` in table cells. The reading's tokens are aligned to the text layer's in
                 reading order; a run of struck tokens on one line becomes one span. Text the reading already
                 strikes, and any block holding an unpaired `~~`, is left as it is

A region the options do not change keeps its base candidate, so it is written back byte for byte.
"""

from __future__ import annotations

import re

from docparse.core.ir import Candidate, Page, Provenance, Region
from docparse.fixes.align import TOKEN, key, match, tokens

FORMAT_OPTIONS = frozenset({"cell_strike", "strike"})
READER = "fix.formatting"
PROSE = frozenset({"paragraph", "list", "heading"})
MIN_STRUCK = 0.5  # share of a PDF token's characters a line must cross
_TILDE_STRIKE = re.compile(r"~~(?=[^\s~])(.+?)(?<=[^\s~])~~", re.S)
_STRIKE_SPAN = re.compile(r"~~(?=[^\s~]).+?(?<=[^\s~])~~|<(s|del|strike)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_RUN_BREAK = frozenset("\n<>|~*_[]()`$")  # a struck run never crosses a line, a tag, a cell or other markup


def needs_decorations(options: dict) -> bool:
    """Does the fix need the strikethrough / underline marks of the text layer?"""
    return bool(options.get("strike"))


def _select(region: Region, raw: str, detail: str) -> None:
    cand = Candidate(
        id=f"{region.id}c{len(region.candidates)}",
        reader=READER,
        raw=raw,
        content=region.chosen().content,
        prov=[Provenance(source="rule", ref=READER, detail=detail)],
    )
    region.candidates.append(cand)
    region.selected = cand.id


def _pdf_tokens(words) -> tuple[list[str], list[bool]]:
    """Text-layer tokens in reading order, and whether a line strikes most of each one's characters."""
    keys, struck = [], []
    for w in words:
        for m in TOKEN.finditer(w.text):
            keys.append(key(m.group()))
            marks = w.marks[m.start() : m.end()]
            struck.append(bool(marks) and marks.count("s") >= MIN_STRUCK * len(marks))
    return keys, struck


def _blocked_spans(raw: str, kind: str) -> list[tuple[int, int]]:
    """Character ranges of `raw` that must not get a new strike: text already struck, and every piece (a
    paragraph, a list line, a table) that holds a `~~` without a partner."""
    pieces = [(0, len(raw))]
    if kind == "list":
        pieces, start = [], 0
        for line in raw.split("\n"):
            pieces.append((start, start + len(line)))
            start += len(line) + 1
    out = []
    for a, b in pieces:
        piece = raw[a:b]
        spans = [(a + m.start(), a + m.end()) for m in _STRIKE_SPAN.finditer(piece)]
        rest = _STRIKE_SPAN.sub("", piece)
        out += [(a, b)] if "~~" in rest else spans
    return out


def _struck_runs(raw: str, toks: list[tuple[int, int, str]], struck: list[bool | None], blocked: list[bool]):
    """(start, end) of each run of struck tokens; an unaligned token between two struck ones joins the run."""
    runs, i, n = [], 0, len(toks)
    while i < n:
        if not struck[i] or blocked[i]:
            i += 1
            continue
        j, k = i, i + 1
        while k < n and not blocked[k] and struck[k] is not False:
            if _RUN_BREAK.intersection(raw[toks[k - 1][1] : toks[k][0]]):
                break
            if struck[k]:
                j = k
            k += 1
        runs.append((toks[i][0], toks[j][1]))
        i = j + 1
    return runs


def _strikes(page: Page, raws: list[str], words) -> list[str]:
    pdf_keys, pdf_struck = _pdf_tokens(words)
    if not any(pdf_struck):
        return raws
    per_region = [tokens(raw) for raw in raws]
    flat = [t[2] for toks in per_region for t in toks]
    partner = match(flat, pdf_keys)
    out, pos = list(raws), 0
    for r, (region, raw, toks) in enumerate(zip(page.regions, raws, per_region, strict=True)):
        mine = partner[pos : pos + len(toks)]
        pos += len(toks)
        if region.kind not in PROSE and region.kind != "table":
            continue
        struck = [None if j < 0 else pdf_struck[j] for j in mine]
        if not any(struck):
            continue
        spans = _blocked_spans(raw, region.kind)
        blocked = [any(a <= s < b for a, b in spans) for s, _, _ in toks]
        opener, closer = ("<s>", "</s>") if region.kind == "table" else ("~~", "~~")
        new = raw
        for s, e in sorted(_struck_runs(raw, toks, struck, blocked), reverse=True):
            new = new[:s] + opener + new[s:e] + closer + new[e:]
        out[r] = new
    return out


def refine_format(page: Page, rendered, options: dict) -> None:
    """Apply the enabled options to the page's regions (a changed region gets a new selected candidate)."""
    raws = [region.chosen().raw for region in page.regions]
    new = list(raws)
    if options.get("cell_strike"):
        new = [_TILDE_STRIKE.sub(r"<s>\1</s>", raw) if region.kind == "table" else raw
               for region, raw in zip(page.regions, new, strict=True)]  # fmt: skip
    words = getattr(rendered, "text_words", None) if rendered is not None else None
    if options.get("strike") and words:
        new = _strikes(page, new, words)
    for region, old, raw in zip(page.regions, raws, new, strict=True):
        if raw != old:
            _select(region, raw, ",".join(sorted(k for k in FORMAT_OPTIONS if options.get(k))))

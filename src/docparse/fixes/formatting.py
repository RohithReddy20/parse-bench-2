"""Formatting corrections of the page reading from the PDF itself (design §6, formatting). Each is one option:

    cell_strike  inside an HTML table, `~~text~~` becomes `<s>text</s>`. Markdown is not read inside HTML blocks,
                 so a strike the model wrote with tildes in a cell shows as literal tildes, not as struck text
    strike       words the PDF draws a line through (pdf/drawings.py) are struck in the reading: `~~...~~` in
                 prose, `<s>...</s>` in table cells. The reading's tokens are aligned to the text layer's in
                 reading order; a run of struck tokens on one line becomes one span. Text the reading already
                 strikes, and any block holding an unpaired `~~`, is left as it is
    unbold       a `**...**` span in a paragraph or list loses its asterisks when the PDF sets at least 80% of its
                 aligned words in a regular face, on a page whose fonts do carry bold (some word is in a bold
                 face). Bold inside headings is the heading's and is left alone
    drop_underline  a `<u>...</u>` span loses its tags when the PDF draws no line under any of its aligned words,
                 on a page that draws its decorations as lines (some word is underlined or struck)
    list_marker  a list marker the reading wrapped in emphasis at the start of a line (`**1.** text`) becomes list
                 syntax (`1. text`): markdown reads no list there, and the marker is list syntax either way

A region the options do not change keeps its base candidate, so it is written back byte for byte.
"""

from __future__ import annotations

import re

from docparse.core.ir import Candidate, Page, Provenance, Region
from docparse.fixes.align_text import TOKEN, key, match, tokens

FORMAT_OPTIONS = frozenset({"cell_strike", "strike", "underline", "unbold", "list_marker", "drop_underline"})
READER = "fix.formatting"
PROSE = frozenset({"paragraph", "list", "heading"})
MIN_STRUCK = 0.5  # share of a PDF token's characters a line must cross
_TILDE_STRIKE = re.compile(r"~~(?=[^\s~])(.+?)(?<=[^\s~])~~", re.S)
_STRIKE_SPAN = re.compile(r"~~(?=[^\s~]).+?(?<=[^\s~])~~|<(s|del|strike)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_RUN_BREAK = frozenset("\n<>|~*_[]()`$")  # a struck run never crosses a line, a tag, a cell or other markup
_BOLD_SPAN = re.compile(r"(?<!\*)\*\*(?=[^\s*])(.+?)(?<=[^\s*])\*\*(?!\*)", re.S)
_MARKER = r"(?:[-*+•◦▪●○■‣∙–—·]|[0-9]{1,3}[.)]|\([0-9]{1,3}\)|\([a-z]{1,4}\)|\([A-Z]\)|[a-z][.)]|[ivx]{1,4}[.)])"
_STYLED_MARKER = re.compile(r"(?m)^([ \t]*)(\*\*|__)(" + _MARKER + r")\2[ \t]+(?=\S)")
_UNDER_SPAN = re.compile(r"<(u|ins)\b[^>]*>(.*?)</\1\s*>", re.S | re.I)
_VISIBLE_MASK = re.compile(r"<[^>\n]*>|!\[[^\]\n]*\]\([^)\n]*\)|\]\([^)\n]*\)")
REGULAR_SHARE = 0.8  # of a bold span's aligned words that must be in a regular face
ALIGNED_SHARE = 0.5  # of a bold span's words that must be aligned to the text layer


def _visible_tokens(raw: str) -> list[tuple[int, int, str]]:
    """Tokenize visible text while masking HTML tags/attributes and markdown image/link markup to preserve offsets."""
    masked = _VISIBLE_MASK.sub(lambda m: " " * len(m.group()), raw)
    return tokens(masked)


def needs_decorations(options: dict) -> bool:
    """Does the fix need the strikethrough / underline marks of the text layer?"""
    return bool(options.get("strike") or options.get("underline") or options.get("drop_underline"))


def needs_styles(options: dict) -> bool:
    """Does the fix need to know which text-layer words are set in a bold face?"""
    return bool(options.get("unbold"))


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


def _pdf_tokens(words, mark: str = "s") -> tuple[list[str], list[bool]]:
    """Text-layer tokens in stream order, and whether a decoration line marks most of each token's characters."""
    keys, marked = [], []
    for w in words:
        for m in TOKEN.finditer(w.text):
            keys.append(key(m.group()))
            marks = w.marks[m.start() : m.end()]
            marked.append(bool(marks) and marks.count(mark) >= MIN_STRUCK * len(marks))
    return keys, marked


def _align_with_reordered_recovery(
    per_region: list[list[tuple[int, int, str]]],
    pdf_keys: list[str],
    pdf_marked: list[bool],
) -> list[list[int]]:
    """Align region tokens to PDF tokens, recovering blocks whose PDF stream order differs from visual order."""
    flat = [t[2] for toks in per_region for t in toks]
    partner = match(flat, pdf_keys)
    by_region: list[list[int]] = []
    pos = 0
    for toks in per_region:
        by_region.append(partner[pos : pos + len(toks)])
        pos += len(toks)
    used_pdf = {j for j in partner if j >= 0}
    unmatched_marked = [j for j, flag in enumerate(pdf_marked) if flag and j not in used_pdf]
    if not unmatched_marked:
        return by_region
    # Build contiguous unmatched PDF spans that contain at least one unmatched decorated token
    free_idx = [j for j in range(len(pdf_keys)) if j not in used_pdf]
    free_spans: list[list[int]] = []
    for j in free_idx:
        if not free_spans or j != free_spans[-1][-1] + 1:
            free_spans.append([j])
        else:
            free_spans[-1].append(j)
    marked_spans = [sp for sp in free_spans if any(pdf_marked[j] for j in sp)]
    if not marked_spans:
        return by_region
    taken_spans: set[int] = set()
    for r, (toks, mine) in enumerate(zip(per_region, by_region, strict=True)):
        if not toks or all(j >= 0 for j in mine):
            continue
        unmatched_pos = [i for i, j in enumerate(mine) if j < 0]
        if not unmatched_pos:
            continue
        r_keys = [toks[i][2] for i in unmatched_pos]
        best_s, best_hits, best_loc = -1, 0, []
        for s_idx, sp in enumerate(marked_spans):
            if s_idx in taken_spans:
                continue
            sp_keys = [pdf_keys[j] for j in sp]
            loc = match(r_keys, sp_keys)
            hits = sum(1 for x in loc if x >= 0)
            has_mark = any(x >= 0 and pdf_marked[sp[x]] for x in loc)
            # Require either >=2 aligned tokens or full alignment on a 1-token region with unique key match
            if has_mark and (hits >= 2 or (len(r_keys) == 1 and hits == 1 and len(sp) <= 3)) and hits > best_hits:
                best_s, best_hits, best_loc = s_idx, hits, loc
        if best_s >= 0:
            sp = marked_spans[best_s]
            updated = list(mine)
            for i_local, x in enumerate(best_loc):
                if x >= 0:
                    updated[unmatched_pos[i_local]] = sp[x]
            by_region[r] = updated
            taken_spans.add(best_s)
    return by_region


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
    pdf_keys, pdf_struck = _pdf_tokens(words, "s")
    if not any(pdf_struck):
        return raws
    per_region = [_visible_tokens(raw) for raw in raws]
    by_region = _align_with_reordered_recovery(per_region, pdf_keys, pdf_struck)
    out = list(raws)
    for r, (region, raw, toks, mine) in enumerate(zip(page.regions, raws, per_region, by_region, strict=True)):
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


def _underlines(page: Page, raws: list[str], words) -> list[str]:
    pdf_keys, pdf_under = _pdf_tokens(words, "u")
    if not any(pdf_under):
        return raws
    per_region = [_visible_tokens(raw) for raw in raws]
    by_region = _align_with_reordered_recovery(per_region, pdf_keys, pdf_under)
    out = list(raws)
    for r, (region, raw, toks, mine) in enumerate(zip(page.regions, raws, per_region, by_region, strict=True)):
        if region.kind not in PROSE and region.kind != "table":
            continue
        under = [None if j < 0 else pdf_under[j] for j in mine]
        if not any(under):
            continue
        spans = [(m.start(), m.end()) for m in _UNDER_SPAN.finditer(raw)]
        blocked = [any(a <= s < b for a, b in spans) for s, _, _ in toks]
        new = raw
        for s, e in sorted(_struck_runs(raw, toks, under, blocked), reverse=True):
            new = new[:s] + "<u>" + new[s:e] + "</u>" + new[e:]
        out[r] = new
    return out


def _unbold(page: Page, raws: list[str], words) -> list[str]:
    if not any(w.bold >= 0.5 for w in words):
        return raws  # the fonts carry no bold at all: they say nothing either way
    keys, bold = [], []
    for w in words:
        for m in TOKEN.finditer(w.text):
            keys.append(key(m.group()))
            bold.append(w.bold >= 0.5)
    per_region = [_visible_tokens(raw) for raw in raws]
    partner = match([t[2] for toks in per_region for t in toks], keys)
    out, pos = list(raws), 0
    for r, (region, raw, toks) in enumerate(zip(page.regions, raws, per_region, strict=True)):
        mine = partner[pos : pos + len(toks)]
        pos += len(toks)
        if region.kind not in ("paragraph", "list"):
            continue
        new = raw
        for m in reversed(list(_BOLD_SPAN.finditer(raw))):
            inside = [i for i, t in enumerate(toks) if m.start(1) <= t[0] < m.end(1)]
            faces = [bold[mine[i]] for i in inside if mine[i] >= 0]
            if not inside or len(faces) < ALIGNED_SHARE * len(inside):
                continue
            if faces.count(False) >= REGULAR_SHARE * len(faces):
                new = new[: m.start()] + m.group(1) + new[m.end() :]
        out[r] = new
    return out


def _drop_underline(page: Page, raws: list[str], words) -> list[str]:
    if not any("u" in w.marks or "s" in w.marks for w in words):
        return raws  # the page draws no decoration lines: no evidence either way
    keys, under = [], []
    for w in words:
        for m in TOKEN.finditer(w.text):
            keys.append(key(m.group()))
            marks = w.marks[m.start() : m.end()]
            under.append(bool(marks) and marks.count("u") >= 0.5 * len(marks))
    per_region = [_visible_tokens(raw) for raw in raws]
    partner = match([t[2] for toks in per_region for t in toks], keys)
    out, pos = list(raws), 0
    for r, (raw, toks) in enumerate(zip(raws, per_region, strict=True)):
        mine = partner[pos : pos + len(toks)]
        pos += len(toks)
        new = raw
        for m in reversed(list(_UNDER_SPAN.finditer(raw))):
            inside = [i for i, t in enumerate(toks) if m.start(2) <= t[0] < m.end(2)]
            lines = [under[mine[i]] for i in inside if mine[i] >= 0]
            if inside and len(lines) >= ALIGNED_SHARE * len(inside) and not any(lines):
                new = new[: m.start()] + m.group(2) + new[m.end() :]
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
    if options.get("underline") and words:
        new = _underlines(page, new, words)
    if options.get("unbold") and words:
        new = _unbold(page, new, words)
    if options.get("drop_underline") and words:
        new = _drop_underline(page, new, words)
    if options.get("list_marker"):
        new = [_STYLED_MARKER.sub(r"\1\3 ", raw) if region.kind in ("paragraph", "list") else raw
               for region, raw in zip(page.regions, new, strict=True)]  # fmt: skip
    for region, old, raw in zip(page.regions, raws, new, strict=True):
        if raw != old:
            _select(region, raw, ",".join(sorted(k for k in FORMAT_OPTIONS if options.get(k))))

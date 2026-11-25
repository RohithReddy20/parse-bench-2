"""Form repairs from the text a fill tool typed on top of the page (the overlay, `docparse.pdf.overlay`).

On a filled form the PDF holds every typed value exactly, with its position, and the page read holds the
form's structure. Each repair is one option of `form_options`, and acts only on a page with an overlay:

    combs    a value typed one character per box ("2 2 8 6 2" in the read, one overlay glyph per box) is
             written as one string ("22862") where the read spells out exactly those characters
    boxes    check-box states from the overlay's tick glyphs (`forms_boxes`)

The overlay's words are grouped into fills (one typed value each); each fill is placed on the markdown tokens
that spell it, next to the printed words around it (see `read_form`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docparse.fixes.align import Alignment, align, fold
from docparse.fixes.layout_refine import PtWord
from docparse.pdf.render import RenderedPage

TICKS = frozenset("✓✔✗✘☒☑xX")
COMB_GAP = 2.6  # single glyphs this many glyph heights apart (or closer) are boxes of one comb field
WORD_GAP = 0.8  # words of one typed value are at most this many heights apart
PITCH_JUMP = 1.6  # a comb gap this many times its median gap separates two values
PLACE_SHARE = 0.6  # a fill is placed where at least this share of its characters matched
_JOINABLE = re.compile(r"[ \t*_]*")  # what may sit between the glyphs of a comb in the markdown


@dataclass
class Fill:
    words: list[PtWord]

    @property
    def comb(self) -> bool:
        return len(self.words) > 1 and all(len(w.text) == 1 for w in self.words)

    @property
    def text(self) -> str:
        return ("" if self.comb else " ").join(w.text for w in self.words)

    @property
    def key(self) -> str:
        return "".join(fold(ch) for ch in self.text)


def page_words(rendered: RenderedPage, fonts: frozenset[str]) -> tuple[list[PtWord], list[PtWord]]:
    """(printed words, overlay words) in points, each in content-stream order."""
    s = rendered.scale
    printed, overlay = [], []
    for w in rendered.text_words or []:
        pt = PtWord(w.text, w.box[0] * s, w.box[1] * s, w.box[2] * s, w.box[3] * s)
        (overlay if w.font in fonts else printed).append(pt)
    return printed, overlay


def _same_line(a: PtWord, b: PtWord) -> bool:
    return abs(a.cy - b.cy) < 0.5 * max(a.y1 - a.y0, b.y1 - b.y0)


def _between(a: PtWord, b: PtWord, printed: list[PtWord]) -> bool:
    """Is a printed word on the line between a and b?"""
    return any(a.x1 - 0.5 <= p.cx <= b.x0 + 0.5 and _same_line(a, p) for p in printed)


def _pitch_runs(words: list[PtWord]) -> list[list[PtWord]]:
    """A comb cut where the gap jumps: day, month and year boxes typed as one row stay three values."""
    gaps = [b.x0 - a.x1 for a, b in zip(words, words[1:], strict=False)]
    if len(gaps) < 2:
        return [words]
    limit = PITCH_JUMP * sorted(gaps)[len(gaps) // 2]
    out = [[words[0]]]
    for gap, w in zip(gaps, words[1:], strict=True):
        if gap > limit:
            out.append([])
        out[-1].append(w)
    return out


def overlay_groups(overlay: list[PtWord], printed: list[PtWord]) -> list[list[PtWord]]:
    """Overlay words on one line, close together, with no printed word between them."""
    out: list[list[PtWord]] = []
    rows: list[list[PtWord]] = []
    for w in sorted(overlay, key=lambda w: (w.cy, w.x0)):
        for row in rows:
            if _same_line(row[0], w):
                row.append(w)
                break
        else:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: w.x0)
        current = [row[0]]
        for prev, w in zip(row, row[1:], strict=False):
            height = max(prev.y1 - prev.y0, w.y1 - w.y0)
            single = len(prev.text) == 1 and len(w.text) == 1
            limit = COMB_GAP if single else WORD_GAP
            if w.x0 - prev.x1 <= limit * height and not _between(prev, w, printed):
                current.append(w)
            else:
                out.append(current)
                current = [w]
        out.append(current)
    split: list[list[PtWord]] = []
    for group in out:
        split += _pitch_runs(group) if all(len(w.text) == 1 for w in group) else [group]
    return split


def ticks(overlay: list[PtWord], printed: list[PtWord]) -> list[PtWord]:
    """Check marks: a tick glyph standing alone (an `X` inside a typed value is a letter)."""
    return [g[0] for g in overlay_groups(overlay, printed) if len(g) == 1 and g[0].text in TICKS]


def fills(overlay: list[PtWord], printed: list[PtWord]) -> list[Fill]:
    """The overlay's typed values (check marks excluded)."""
    out = [Fill(g) for g in overlay_groups(overlay, printed) if not (len(g) == 1 and g[0].text in TICKS)]
    return [f for f in out if f.key]


def _anchor(fill: Fill, printed: list[PtWord]) -> int | None:
    """Index of the printed word a reader passes just before the fill: the nearest word to its left at its
    height, else the end of the label above it (the closest word above, followed along its line in content
    order while the words stay close)."""
    first, last = fill.words[0], fill.words[-1]
    left = [i for i, p in enumerate(printed) if p.x1 <= first.x0 + 1.0 and p.y0 < first.y1 and p.y1 > first.y0]
    if left:
        return max(left, key=lambda i: (printed[i].x1, printed[i].y0))
    above = [i for i, p in enumerate(printed) if p.y1 <= first.y0 + 1.0 and p.x0 < last.x1]
    if not above:
        return None
    i = min(above, key=lambda i: (first.y0 - printed[i].y1) + 0.3 * max(0.0, first.x0 - printed[i].x1))
    while i + 1 < len(printed):
        p, q = printed[i], printed[i + 1]
        if not (_same_line(p, q) and 0 <= q.x0 - p.x1 <= p.y1 - p.y0):
            break
        i += 1
    return i


@dataclass
class FormPage:
    """The page read aligned with the PDF: printed words in content order, each fill right after its anchor."""

    markdown: str
    printed: list[PtWord]
    fills: list[Fill]
    ticks: list[PtWord]
    al: Alignment
    owner: list[int]  # per aligned word: -1 for a printed word, else the index of its fill
    placed: dict[int, tuple[int, int]]  # fill -> (first token, last token) of the markdown tokens holding it


def _free_runs(al: Alignment, free: list[bool], key: str) -> list[tuple[int, int]]:
    keys = ["".join(fold(ch) for ch in t.text) for t in al.tokens]
    runs = []
    for t0 in range(len(al.tokens)):
        if not free[t0] or not key.startswith(keys[t0]):
            continue
        text, t1 = keys[t0], t0
        while len(text) < len(key) and t1 + 1 < len(al.tokens) and free[t1 + 1]:
            t1 += 1
            text += keys[t1]
            if not key.startswith(text):
                break
        if text == key:
            runs.append((t0, t1))
    return runs


def read_form(markdown: str, printed: list[PtWord], overlay: list[PtWord]) -> FormPage:
    page_fills = fills(overlay, printed)
    after: dict[int | None, list[int]] = {}
    for f, fill in enumerate(page_fills):
        after.setdefault(_anchor(fill, printed), []).append(f)
    words: list[PtWord] = []
    owner: list[int] = []

    def add(fs: list[int]) -> None:
        for f in sorted(fs, key=lambda f: (page_fills[f].words[0].cy, page_fills[f].words[0].x0)):
            words.extend(page_fills[f].words)
            owner.extend([f] * len(page_fills[f].words))

    for i, p in enumerate(printed):
        words.append(p)
        owner.append(-1)
        add(after.get(i, []))
    add(after.get(None, []))
    al = align(markdown, words, reorder=True)
    # a fill is placed on the tokens its characters matched, when those spell most of it and hold nothing printed
    hits: dict[int, list[int]] = {}
    total: dict[int, int] = {}
    for w in al.p_word:
        if owner[w] >= 0:
            total[owner[w]] = total.get(owner[w], 0) + 1
    token_owners: list[set[int]] = [set() for _ in al.tokens]
    for k, p in enumerate(al.amatch):
        if p is not None:
            f = owner[al.p_word[p]]
            token_owners[al.a_tok[k]].add(f)
            if f >= 0:
                hits.setdefault(f, []).append(al.a_tok[k])
    placed: dict[int, tuple[int, int]] = {}
    for f, toks in hits.items():
        t0, t1 = min(toks), max(toks)
        if len(toks) >= PLACE_SHARE * total[f] and all(token_owners[t] <= {f} for t in range(t0, t1 + 1)):
            placed[f] = (t0, t1)
    # fills the alignment missed: an exact spelling among tokens nothing matched, when there is only one
    free = [not owners for owners in token_owners]
    for f, fill in enumerate(page_fills):
        if f in placed:
            continue
        runs = _free_runs(al, free, fill.key)
        if len(runs) == 1:
            placed[f] = runs[0]
            for t in range(runs[0][0], runs[0][1] + 1):
                free[t] = False
    return FormPage(markdown, printed, page_fills, ticks(overlay, printed), al, owner, placed)


def comb_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """A comb value spelled out glyph by glyph becomes one string, where the markdown spells exactly its glyphs
    with only spaces or emphasis marks between them."""
    edits = []
    markdown, al = page.markdown, page.al
    for f, (t0, t1) in page.placed.items():
        fill = page.fills[f]
        if not fill.comb or t1 == t0:
            continue
        if "".join(fold(ch) for t in range(t0, t1 + 1) for ch in al.tokens[t].text) != fill.key:
            continue
        gaps = [markdown[al.tokens[t].end : al.tokens[t + 1].start] for t in range(t0, t1)]
        if not all(_JOINABLE.fullmatch(g) and g.count("*") % 2 == 0 and g.count("_") % 2 == 0 for g in gaps):
            continue
        edits.append((al.tokens[t0].start, al.tokens[t1].end, fill.text))
    return edits


def apply_edits(markdown: str, edits: list[tuple[int, int, str]]) -> str:
    out, pos = [], 0
    for start, end, text in sorted(edits):
        if start < pos:
            continue  # overlaps an earlier edit
        out += [markdown[pos:start], text]
        pos = end
    out.append(markdown[pos:])
    return "".join(out)


def repair(
    markdown: str,
    rendered: RenderedPage,
    fonts: frozenset[str],
    options: dict,
    pdf_bytes: bytes = b"",
    page_index: int = 0,
) -> str:
    """The page markdown with the enabled repairs; unchanged without an overlay or options."""
    if not fonts or not options or not markdown:
        return markdown
    printed, overlay = page_words(rendered, fonts)
    if not overlay:
        return markdown
    page = read_form(markdown, printed, overlay)
    edits: list[tuple[int, int, str]] = []
    if options.get("combs"):
        edits += comb_edits(page)
    if options.get("boxes"):
        from docparse.fixes.forms_boxes import box_edits
        from docparse.pdf.boxes import check_boxes

        edits += box_edits(page, check_boxes(pdf_bytes, page_index))
    return apply_edits(markdown, edits) if edits else markdown

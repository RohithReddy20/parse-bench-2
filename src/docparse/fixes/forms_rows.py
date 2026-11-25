"""Form lines whose printed numbers the read put in table cells of their own (form options `ids` and `codes`), and
character-box outlines read as empty check boxes before a typed value (option `outlines`).

A form field is found by its caption and its value is read from the caption's own slot: the rest of its cell, else
the next cell of its row. A page read often writes a form line as table cells, `| 8 | Advertising | 8 | 1,597 |`:
the line number in a cell of its own (so the caption is found without its number, or not at all when it is one
word) and the same number printed again at the value box, sitting in the caption's slot.

    ids       a cell holding only a printed line number, followed by a caption cell, is joined to that caption
              (`| | 8 Advertising | | 1,597 |`), and the number printed again before the value is dropped
    codes     a printed box code between a caption and its typed value (`| Salaries | 9060 | 2,149.10 |`) moves
              to the front of the caption (`| 9060 Salaries | | 2,149.10 |`); so does one before an empty slot
              that ends the row
    outlines  character-box outlines the read wrote as empty check boxes right before a typed value
              (`£ [ ] [ ] 7060`) are dropped (`£ 7060`)

Cells holding typed (overlay) text are never moved. A table whose first row captions its value columns
(`| | | Plus-value | Moins-value |`) is a grid read by column, and one where a caption owns more than one value slot
(several columns of amounts) is too: `ids` leaves both alone, `codes` the first.
"""

from __future__ import annotations

import bisect
import re

from docparse.fixes.forms import FormPage
from docparse.fixes.tables_html import _CELL, _ROW, HCell, HTable, parse_table

_TABLE = re.compile(r"<table\b.*?</table\s*>", re.S | re.I)
LINE_ID = re.compile(r"\(?[0-9]{1,3}[a-zA-Z]?[.)]?")  # 8, 16a, (3), 22.
CODE = re.compile(r"\(?[0-9]{1,5}[a-zA-Z]?[.)]?|[a-zA-Z]")  # anything a box may be numbered with
BOX_CODE = re.compile(r"[0-9]{2,5}[a-zA-Z]?|[0-9]{1,2}\.[0-9]{1,3}|[A-Z]{1,2}[0-9]{1,3}")  # 9060, 3.111, RN33
CAPTION = re.compile(r"(?:\[[ xX✓]?\]|[☐☑☒□■○●\s*(•-])*[^\W\d_]{2,}")  # words, maybe after a box or a bullet
_WORDY = re.compile(r"\w")
_OUTLINES = re.compile(r"((?:[ \t*_]*(?:\[[ \t]?\]|[☐□]))+)[ \t*_]*$")
_SIGNS = re.compile(r"[ \t]*[£$€¥]?[ \t]*")
_UNIT_END = re.compile(r"\n|<br[ \t]*/?>|</?t[dhr][^>\n]*>")


def _key(text: str) -> str:
    return re.sub(r"\W", "", text).lower()


def _set(cell: HCell, inner: str) -> None:
    cell.inner, cell.raw = inner, ""


def _typed_tokens(page: FormPage) -> list[bool]:
    """Per markdown token: did any of its characters match a typed (overlay) word?"""
    al = page.al
    typed = [False] * len(al.tokens)
    for k, p in enumerate(al.amatch):
        if p is not None and page.owner[al.p_word[p]] >= 0:
            typed[al.a_tok[k]] = True
    return typed


def _typed_cells(page: FormPage, typed: list[bool], start: int, end: int) -> list[list[bool]]:
    """Per row and cell of the table at markdown[start:end] (as `parse_table` reads it): holds typed text?"""
    starts = [t.start for t in page.al.tokens]
    md = page.markdown
    out = []
    for rm in _ROW.finditer(md, start, end):
        row = []
        body0 = rm.start() + re.match(r"<tr\b[^>]*>", rm.group(0), re.I).end()
        for cm in _CELL.finditer(md, body0, rm.end()):
            lo, hi = bisect.bisect_left(starts, cm.start(3)), bisect.bisect_left(starts, cm.end(3))
            row.append(any(typed[t] for t in range(lo, hi)))
        out.append(row)
    return out


def _header(t: HTable, typed: list[list[bool]]) -> bool:
    """A first row of column captions over the value columns (`| | | Plus-value | Moins-value |`): printed text
    in two or more cells, no line number, more rows below."""
    first = t.rows[0].cells
    return (
        len(t.rows) >= 2
        and len(first) >= 2
        and any(c.text for c in first)
        and not any(typed[0])
        and not any(LINE_ID.fullmatch(c.text) for c in first)
    )


def _id_plan(t: HTable, typed: list[list[bool]]) -> list[tuple[int, int, int | None]] | None:
    """(row, line-number cell, repeated-number cell or None) per joinable line; None when a caption owns several
    value slots and one of them is filled (columns of amounts)."""
    plan = []
    for r, row in enumerate(t.rows):
        cells = row.cells
        i = 0
        while i < len(cells) - 1:
            a, b = cells[i], cells[i + 1]
            if not (
                LINE_ID.fullmatch(a.text) and not typed[r][i] and a.colspan == a.rowspan == 1 and CAPTION.match(b.text)
            ):
                i += 1
                continue
            repeat, slots, filled = None, 0, 0
            for j in range(i + 2, len(cells)):
                c = cells[j]
                next_line = LINE_ID.fullmatch(c.text) and j + 1 < len(cells) and CAPTION.match(cells[j + 1].text)
                if (CAPTION.match(c.text) and not CODE.fullmatch(c.text)) or next_line:
                    break
                if repeat is None and _key(c.text) == _key(a.text):
                    repeat = j
                    continue
                slots += c.colspan
                filled += bool(c.text)
            if slots > 1 and filled:
                return None
            plan.append((r, i, repeat))
            i += 2
    return plan


def _code_plan(t: HTable, typed: list[list[bool]], done: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """(row, caption cell, code cell): a printed code right of a caption (past sign cells), followed by a typed
    value (past empty or sign cells) or by nothing up to the row's end."""
    plan = []
    for r, row in enumerate(t.rows):
        cells = row.cells
        for i, c in enumerate(cells):
            if not CAPTION.match(c.text) or typed[r][i] or (r, i) in done:
                continue
            j = i + 1
            while j < len(cells) and cells[j].text and not _WORDY.search(cells[j].text):
                j += 1  # + = £ $
            if j >= len(cells) or typed[r][j] or (r, j) in done or not BOX_CODE.fullmatch(cells[j].text.strip(" .*")):
                continue
            k = j + 1
            while k < len(cells) and not _WORDY.search(cells[k].text):
                k += 1
            if k >= len(cells) or typed[r][k]:
                plan.append((r, i, j))
    return plan


def rewrite_table(raw: str, typed: list[list[bool]], options: dict) -> str:
    """The table with the enabled moves applied; `raw` itself when nothing moves."""
    t = parse_table(raw)
    if t is None or [len(r.cells) for r in t.rows] != [len(a) for a in typed]:
        return raw
    header = _header(t, typed)
    done: set[tuple[int, int]] = set()
    if options.get("ids") and not header:
        for r, i, repeat in _id_plan(t, typed) or []:
            row = t.rows[r]
            a, b = row.cells[i], row.cells[i + 1]
            _set(b, a.inner.strip() + " " + b.inner.lstrip())
            _set(a, "")
            done.add((r, i + 1))
            if repeat is not None:
                _set(row.cells[repeat], "")
                done.add((r, repeat))
            row.edited = t.changed = True
    if options.get("codes") and not header:
        for r, i, j in _code_plan(t, typed, done):
            row = t.rows[r]
            caption, code = row.cells[i], row.cells[j]
            _set(caption, code.inner.strip() + " " + caption.inner.lstrip())
            _set(code, "")
            row.edited = t.changed = True
    return t.render() if t.changed else raw


def row_edits(page: FormPage, options: dict) -> list[tuple[int, int, str]]:
    """Whole-table replacements for the tables of the page read that change."""
    if not (options.get("ids") or options.get("codes")):
        return []
    typed = _typed_tokens(page)
    edits = []
    for m in _TABLE.finditer(page.markdown):
        new = rewrite_table(m.group(0), _typed_cells(page, typed, m.start(), m.end()), options)
        if new != m.group(0):
            edits.append((m.start(), m.end(), new))
    return edits


def outline_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """Box marks right before a placed typed value, with only spaces, emphasis or a currency sign before them on
    the value's line or cell (`£ [ ] [ ] 7060`): the outlines of the boxes the value was typed into."""
    md, tokens = page.markdown, page.al.tokens
    bounds = [0] + [m.end() for m in _UNIT_END.finditer(md)]
    edits = set()
    for t0, _ in page.placed.values():
        start = tokens[t0].start
        u0 = bounds[bisect.bisect_right(bounds, start) - 1]
        head = md[u0:start]
        m = _OUTLINES.search(head)
        if m and _SIGNS.fullmatch(re.sub(r"[*_]", "", head[: m.start()])):
            edits.add((u0 + m.start(1), u0 + m.end(1), ""))
    return sorted(edits)

"""Focused comb-divider, equation-row, and date-header repairs on filled forms (option `field_clean`).

Repairs three overlay-verified patterns on filled forms:
1. Underlined divider-noise combs (`1<u>5</u>1<u>1</u>1<u>0</u>1<u>0</u>` -> `51100`) and pipe-separated
   combs outside pipe tables (`1 | 5 | 1 | 0 | 1 | 9 | 9 | 7` -> `15101997`,
   `<u>1 | 4 | 1 | 9 | 1 | 1 | 0 | 0</u>` -> `49100`), verified against the PDF's typed overlay fills.
2. Equation-style KV rows (`<td>Formula <u>12</u></td><td></td><td>= 45,369.46 25</td>` -> `25 Formula: 45,369.46`)
   and trailing printed footnote arrows (`<u>11</u> ▶`) on rows ending with `= <overlay> <code>`.
3. Printed date-unit headers (`**Tag** **Monat** **Jahr**`, `Day Month Year`) sitting inline on blank
   caption lines with no overlay (`**vom** **Tag** **Monat** **Jahr**` -> `**vom** [___]\\n**Tag** ...`).
"""

from __future__ import annotations

import html
import re

from docparse.fixes.align import fold
from docparse.fixes.forms import FormPage
from docparse.fixes.forms_tables import _cell_spans, _cell_toks, _classify_cell, _token_flags
from docparse.fixes.tables_html import HCell, parse_table

_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_U_DIV_COMB = re.compile(r"(?:1?<u>[0-9A-Za-z]</u>){4,}1?", re.I)
_PIPE_COMB = re.compile(r"(?:<u>)?(?:[0-9A-Za-z][ \t]*\|[ \t]*){4,}[0-9A-Za-z](?:</u>)?", re.I)
_END_FOOTNOTE = re.compile(r"\s*<(?:u|sup)>\s*[0-9]{1,2}\s*</(?:u|sup)>\s*[▶▸►→]?\s*$", re.I)
_EQ_OV_CODE = re.compile(r"^\s*=\s*(.+?)\s+(?:<(?:strong|b)>)?([0-9]{1,3}[A-Za-z]?)(?:</(?:strong|b)>)?\s*$", re.I)
_DATE_UNITS_ANY = re.compile(
    r"(?<=[ \t])(\*{0,2}(?:Tag|Day|Jour)\*{0,2}[ \t]+\*{0,2}(?:Monat|Month|Mois)\*{0,2}[ \t]+"
    r"\*{0,2}(?:Jahr|Year|Année)\*{0,2})",
    re.I,
)
_READ_NOTES_TAIL = re.compile(r"[ \t]*[–—\-][ \t]*read the notes\.?[ \t]*$", re.I)
_SEE_INSTR_BEFORE_BLANK = re.compile(r"([ \t]+See[ \t]+(?:the[ \t]+)?instructions\.?[ \t]*)\n(\[_{2,}\])", re.I)
_BR_OVVERO = re.compile(r"(?:<br\s*/?>\s*)+ovvero\s*$", re.I)
_ITALIC_PAREN_BLANK = re.compile(r"[ \t]+\(\*[^*\n]{4,90}\*\)(?=[ \t]+\[_{2,}\])")
_PHONE_MASK_BEFORE_COLON = re.compile(
    r"[ \t]*\(\s*(?:999|000|xxx|XXX)\s*\)[ \t\-]*(?:999|000|xxx|XXX)[ \t\-]*(?:9999|0000|xxxx|XXXX)(?=[ \t]*\*{0,2}:)"
)


def _is_pipe_table_line(md: str, pos: int) -> bool:
    lo = md.rfind("\n", 0, pos) + 1
    hi = md.find("\n", pos)
    line = md[lo : (len(md) if hi < 0 else hi)].strip()
    if line.startswith("|") and line.endswith("|"):
        return True
    lines = md.split("\n")
    idx = md[:lo].count("\n")
    for k in (idx - 1, idx + 1):
        if 0 <= k < len(lines) and re.match(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*\|", lines[k]):
            return True
    return False


def _match_fill_with_ones(chars: list[str], fill_cat: set[str]) -> str | None:
    joined = "".join(fold(c) for c in chars)
    if joined in fill_cat:
        return "".join(chars)
    best: str | None = None
    for cand in fill_cat:
        if len(cand) < 4 or len(cand) >= len(chars):
            continue
        ci = 0
        skipped: list[str] = []
        for ch in chars:
            if ci < len(cand) and fold(ch) == cand[ci]:
                ci += 1
            else:
                skipped.append(ch)
        if ci == len(cand) and skipped and all(s == "1" for s in skipped):
            if best is None or len(cand) > len(best):
                best = cand
    return best


def field_clean_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """Return non-overlapping `(start, end, replacement)` edits for `field_clean`."""
    md, al = page.markdown, page.al
    tok_ov, tok_pr = _token_flags(page)
    printed_words = {fold(w.text) for w in page.printed}
    fill_cat = {f.key for f in page.fills if len(f.key) >= 4}
    for i in range(len(page.fills)):
        acc = ""
        for j in range(i, min(i + 4, len(page.fills))):
            acc += page.fills[j].key
            if len(acc) >= 4:
                fill_cat.add(acc)

    edits: list[tuple[int, int, str]] = []
    tranges: list[tuple[int, int]] = []

    # 1. Footnote stripping and equation-code row merging inside HTML tables
    for m in _HTML_TABLE.finditer(md):
        ts, te = m.start(), m.end()
        tranges.append((ts, te))
        raw_t = m.group(0)
        table = parse_table(raw_t)
        if table is None:
            continue
        spans = _cell_spans(raw_t, ts, table)
        kinds: dict[int, str] = {}
        ctoks: dict[int, list[int]] = {}
        for row in table.rows:
            for c in row.cells:
                sp = spans.get(id(c))
                ct = _cell_toks(al, sp[0], sp[1]) if sp else []
                ctoks[id(c)] = ct
                kinds[id(c)] = _classify_cell(c, ct, tok_ov)
        changed = False
        for r_idx, row in enumerate(table.rows):
            if any(c.rowspan > 1 for c in row.cells) or table.spanned_into(r_idx):
                continue
            if len(row.cells) >= 2 and kinds[id(row.cells[0])] == "CAPTION":
                last = row.cells[-1]
                meq = _EQ_OV_CODE.match(last.inner) if kinds[id(last)] == "MIXED_OV" else None
                # Only strip trailing `<u>NN</u>` footnote on rows ending with `= <overlay> <code>`
                if meq and _END_FOOTNOTE.search(row.cells[0].inner):
                    ct0 = ctoks[id(row.cells[0])]
                    if ct0 and not any(tok_ov[t] for t in ct0):
                        row.cells[0].inner = _END_FOOTNOTE.sub("", row.cells[0].inner).strip()
                        row.cells[0].raw = ""
                        row.edited = True
                        changed = True
                mids = row.cells[1:-1]
                if meq and all(kinds[id(c)] == "EMPTY" for c in mids):
                    val_part, code_part = meq.group(1).strip(), meq.group(2).strip()
                    lct = ctoks[id(last)]
                    if (
                        len(lct) >= 2
                        and not tok_ov[lct[-1]]
                        and (tok_pr[lct[-1]] or fold(code_part) in printed_words)
                        and all(tok_ov[t] for t in lct[:-1])
                    ):
                        cap = row.cells[0].text.strip()
                        span_sum = sum(c.colspan for c in row.cells)
                        row.cells = [
                            HCell(
                                tag="td",
                                attrs=f' colspan="{span_sum}"' if span_sum > 1 else "",
                                inner=f"{html.escape(code_part)} {html.escape(cap)}: {html.escape(val_part)}",
                                raw="",
                                colspan=span_sum,
                            )
                        ]
                        row.edited = True
                        changed = True
        if changed:
            table.changed = True
            edits.append((ts, te, table.render()))

    # 2. Underlined comb-divider noise and pipe-separated combs outside tables
    for pat in (_U_DIV_COMB, _PIPE_COMB):
        for m in pat.finditer(md):
            s, e = m.start(), m.end()
            if any(a <= s < b for a, b in tranges):
                continue
            if pat is _PIPE_COMB and _is_pipe_table_line(md, s):
                continue
            raw_no_tags = re.sub(r"</?u>", "", m.group(0), flags=re.I)
            chars = re.findall(r"[0-9A-Za-z]", raw_no_tags)
            val = _match_fill_with_ones(chars, fill_cat)
            if val:
                edits.append((s, e, val))

    # 3. Printed date-unit headers sitting inline on blank caption lines (`**vom** **Tag** **Monat** **Jahr**`)
    pos = 0
    for raw_line in md.splitlines(keepends=True):
        ls = pos
        le = pos + len(raw_line.rstrip("\r\n"))
        pos += len(raw_line)
        line = md[ls:le]
        if not line.strip() or any(a <= ls < b for a, b in tranges) or "[___]" in line:
            continue
        ms = list(_DATE_UNITS_ANY.finditer(line))
        if not ms:
            continue
        ltoks = [t for t, tok in enumerate(al.tokens) if tok.start >= ls and tok.end <= le]
        if ltoks and not any(tok_ov[t] for t in ltoks):
            for mm in ms:
                edits.append((ls + mm.start(1), ls + mm.start(1), "[___]\n"))

    return edits


def blank_hint_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """Strip printed-only hints that sit in blank field slots on filled forms (option `blank_hints`):
    1. Short box captions ending in `– read the notes` followed by a blank currency line (`£ ...`).
    2. `See instructions.` right before `\\n[___]`.
    3. Printed phone format masks `(999) 999-9999` right before `:`.
    4. Italic parenthetical references `(*...*)` immediately before `[___]`.
    5. Printed `<br/>ovvero` and empty `,00` / `,—` decimal placeholders inside `<td>` cells."""
    md, al = page.markdown, page.al
    tok_ov, _ = _token_flags(page)
    edits: list[tuple[int, int, str]] = []
    tranges: list[tuple[int, int]] = []

    for m in _HTML_TABLE.finditer(md):
        ts, te = m.start(), m.end()
        tranges.append((ts, te))
        raw_t = m.group(0)
        table = parse_table(raw_t)
        if table is None:
            continue
        spans = _cell_spans(raw_t, ts, table)
        changed = False
        for row in table.rows:
            for c in row.cells:
                if c.tag != "td":
                    continue
                sp = spans.get(id(c))
                ct = _cell_toks(al, sp[0], sp[1]) if sp else []
                if any(tok_ov[t] for t in ct):
                    continue
                if _BR_OVVERO.search(c.inner):
                    c.inner = _BR_OVVERO.sub("", c.inner).strip()
                    c.raw = ""
                    row.edited = True
                    changed = True
                elif c.text.strip() in (",00", ", 00", ",—"):
                    c.inner = "[___]"
                    c.raw = ""
                    row.edited = True
                    changed = True
                elif _PHONE_MASK_BEFORE_COLON.search(c.inner):
                    c.inner = _PHONE_MASK_BEFORE_COLON.sub("", c.inner)
                    c.raw = ""
                    row.edited = True
                    changed = True
        if changed:
            table.changed = True
            edits.append((ts, te, table.render()))

    lines = md.splitlines(keepends=True)
    offs: list[tuple[int, int]] = []
    pos = 0
    for raw_line in lines:
        offs.append((pos, pos + len(raw_line.rstrip("\r\n"))))
        pos += len(raw_line)

    for i, (ls, le) in enumerate(offs):
        line = md[ls:le]
        if not line.strip() or any(a <= ls < b for a, b in tranges):
            continue
        ltoks = [t for t, tok in enumerate(al.tokens) if tok.start >= ls and tok.end <= le]
        if any(tok_ov[t] for t in ltoks):
            continue
        mr = _READ_NOTES_TAIL.search(line)
        if mr:
            cap_words = line[: mr.start()].replace("*", "").split()
            if 1 <= len(cap_words) <= 8:
                j = i + 1
                while j < len(offs) and not md[offs[j][0] : offs[j][1]].strip():
                    j += 1
                if j < len(offs):
                    ns, ne = offs[j]
                    nxt = md[ns:ne].strip()
                    ntoks = [t for t, tok in enumerate(al.tokens) if tok.start >= ns and tok.end <= ne]
                    if nxt and nxt[0] in "£$€¥" and not any(tok_ov[t] for t in ntoks):
                        edits.append((ls + mr.start(), le, ""))
                        continue
        for mp in _PHONE_MASK_BEFORE_COLON.finditer(line):
            edits.append((ls + mp.start(), ls + mp.end(), ""))

    for m in _SEE_INSTR_BEFORE_BLANK.finditer(md):
        s, e = m.start(1), m.end(1)
        if any(a <= s < b for a, b in tranges):
            continue
        mtoks = [t for t, tok in enumerate(al.tokens) if tok.start >= s and tok.end <= e]
        if mtoks and not any(tok_ov[t] for t in mtoks):
            edits.append((s, e, ""))

    for m in _ITALIC_PAREN_BLANK.finditer(md):
        s, e = m.start(), m.end()
        if any(a <= s < b for a, b in tranges):
            continue
        mtoks = [t for t, tok in enumerate(al.tokens) if tok.start >= s and tok.end <= e]
        if mtoks and not any(tok_ov[t] for t in mtoks):
            edits.append((s, e, ""))

    return edits

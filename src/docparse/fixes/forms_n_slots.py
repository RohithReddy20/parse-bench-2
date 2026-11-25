"""Additional field-slot repairs on filled forms (option `field_slots`).

Repairs four common filled-form layout mismatches between printed captions and typed overlay values:
1. Two-row caption-over-value tables (including `colspan` parent headers over `Sub: value` cells),
   excluding signature blocks.
2. Blank field lines whose colon is immediately followed by a printed numeric-code legend
   (`Religion change: 1 = Leave, 2 = Switch` -> `Religion change: [___]\\n1 = Leave, 2 = Switch`).
3. Explicit `Country code: ... Area code: ... Number: ...` placeholder headers followed by an
   overlay-only `(cc) (ac) number` phone line.
4. Printed date-format hints (`YYYY-MM-DD`, `DAY MONTH YEAR`) sitting right before a typed overlay date.
"""

from __future__ import annotations

import html
import re

from docparse.fixes.align import fold
from docparse.fixes.forms import FormPage
from docparse.fixes.forms_tables import _cell_spans, _cell_toks, _classify_cell, _token_flags
from docparse.fixes.tables_html import HCell, parse_table

_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_HAS_BOX = re.compile(r"\[[ \t]*(?:[xX✓✔])?[ \t]*\]|[☐☑☒□■]")
_SIG_WORD = re.compile(r"\b(?:signature|sign|unterschrift|firma)\b", re.I)
_LEGEND_COLON = re.compile(
    r":\*{0,2}[ \t]+([0-9][ \t]*=[ \t]*[^,\n;]{2,35}(?:[,;/][ \t]*[0-9][ \t]*=[ \t]*[^,\n;]{2,35})+)"
)
_PHONE_BLANK_HDR = re.compile(
    r"^\*{0,2}Country code\*{0,2}[ \t]*:[ \t]*(?:\[_{2,}\]|\[ \])?[ \t\n]*(?:\[ \][ \t]*)?"
    r"\*{0,2}Area code\*{0,2}[ \t]*:[ \t]*(?:\[_{2,}\]|\[ \])?[ \t\n]*(?:\[ \][ \t]*)?"
    r"\*{0,2}Number\*{0,2}[ \t]*:[ \t]*(?:\[_{2,}\]|\[ \])?[ \t]*$",
    re.I,
)
_PHONE_GROUPS = re.compile(
    r"\([ \t]*([0-9+][0-9 ]{0,5})[ \t]*\)[ \t]*\([ \t]*([0-9][0-9 ]{0,6})[ \t]*\)[ \t]*([0-9][0-9 \-]{3,16})"
)
_DATE_HINT = re.compile(r"(?:\*{0,2}(?:YYYY|MM|DD|DAY|MONTH|YEAR)\*{0,2}[ \t/\-]*){2,}(?:[ \t/*_]*)", re.I)


def field_slot_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """Return non-overlapping `(start, end, replacement)` edits for `field_slots`."""
    md, al = page.markdown, page.al
    tok_ov, _ = _token_flags(page)
    fill_keys = {f.key for f in page.fills if len(f.key) >= 3}
    printed_str = "".join(fold(w.text) for w in page.printed)
    edits: list[tuple[int, int, str]] = []
    tranges: list[tuple[int, int]] = []

    # 1. Two-row caption-over-value tables (with repeated-fill detection and colspan expansion)
    for m in _HTML_TABLE.finditer(md):
        ts, te = m.start(), m.end()
        tranges.append((ts, te))
        raw_t = m.group(0)
        table = parse_table(raw_t)
        if table is None or len(table.rows) != 2:
            continue
        r0, r1 = table.rows[0], table.rows[1]
        if any(c.rowspan > 1 for c in r0.cells + r1.cells):
            continue
        if any(_SIG_WORD.search(c.text) for c in r0.cells):
            continue
        spans = _cell_spans(raw_t, ts, table)
        kinds: dict[int, str] = {}
        for row in table.rows:
            for c in row.cells:
                sp = spans.get(id(c))
                ct = _cell_toks(al, sp[0], sp[1]) if sp else []
                k = _classify_cell(c, ct, tok_ov)
                if k == "CAPTION":
                    ck = "".join(fold(ch) for ch in c.text)
                    if len(ck) >= 3 and ck in fill_keys and ck not in printed_str:
                        k = "OVERLAY"
                kinds[id(c)] = k
        if (
            all(kinds[id(c)] in ("CAPTION", "EMPTY", "CODE") for c in r0.cells)
            and sum(kinds[id(c)] == "CAPTION" for c in r0.cells) >= 2
            and any(kinds[id(c)] in ("OVERLAY", "MIXED_OV") for c in r1.cells)
            and all(kinds[id(c)] in ("OVERLAY", "MIXED_OV", "EMPTY", "SYM") for c in r1.cells)
        ):
            r1_by_col = {c.col: c for c in r1.cells}
            new_cells: list[HCell] = []
            ok = True
            for c0 in r0.cells:
                cap = c0.text.strip()
                if c0.colspan == 1:
                    c1 = r1_by_col.get(c0.col)
                    if c1 is None or c1.colspan != 1:
                        ok = False
                        break
                    if cap:
                        val = c1.text.strip() if kinds[id(c1)] in ("OVERLAY", "MIXED_OV") else "[___]"
                        sep = " " if cap.endswith(":") else ": "
                        new_cells.append(
                            HCell(tag="td", attrs="", inner=f"{html.escape(cap)}{sep}{html.escape(val)}", raw="")
                        )
                    else:
                        new_cells.append(c1)
                else:
                    covered = [c for c in r1.cells if c0.col <= c.col < c0.col + c0.colspan]
                    if not covered or not all(":" in c.text for c in covered):
                        ok = False
                        break
                    for c1 in covered:
                        inner = f"{html.escape(cap)} {c1.inner.strip()}" if cap else c1.inner
                        new_cells.append(HCell(tag="td", attrs=c1.attrs, inner=inner, raw="", colspan=c1.colspan))
            if ok:
                r0.cells = new_cells
                r0.edited = True
                table.rows = [r0]
                table.seps = [table.seps[0], table.seps[-1]]
                table.changed = True
                edits.append((ts, te, table.render()))

    # 2. Blank lines where a colon is followed immediately by a printed numeric-code legend
    lines = md.splitlines(keepends=True)
    offsets: list[tuple[int, int]] = []
    pos = 0
    for raw_line in lines:
        ls, le = pos, pos + len(raw_line.rstrip("\r\n"))
        offsets.append((ls, le))
        pos += len(raw_line)
        line = md[ls:le]
        if not line.strip() or any(a <= ls < b for a, b in tranges) or _HAS_BOX.search(line) or "[___]" in line:
            continue
        ml = _LEGEND_COLON.search(line)
        if not ml:
            continue
        ltoks = [t for t, tok in enumerate(al.tokens) if tok.start >= ls and tok.end <= le]
        if ltoks and not any(tok_ov[t] for t in ltoks):
            edits.append((ls + ml.start(1), ls + ml.start(1), "[___]\n"))

    # 3. `Country code: ... Area code: ... Number: ...` followed by overlay `(cc) (ac) number`
    i = 0
    while i < len(lines):
        matched_hdr = None
        for span_n in (1, 2):
            if i + span_n > len(lines):
                continue
            s0, e0 = offsets[i][0], offsets[i + span_n - 1][1]
            if any(a <= s0 < b for a, b in tranges):
                continue
            if _PHONE_BLANK_HDR.fullmatch(md[s0:e0].strip()):
                matched_hdr = (s0, e0, i + span_n)
                break
        if matched_hdr is not None:
            s0, _e0, nxt_i = matched_hdr
            j = nxt_i
            while j < len(lines) and j <= nxt_i + 2 and not md[offsets[j][0] : offsets[j][1]].strip():
                j += 1
            if j < len(lines):
                s1, e1 = offsets[j]
                l1 = md[s1:e1]
                mp = _PHONE_GROUPS.search(l1)
                if mp:
                    gs, ge = s1 + mp.start(), s1 + mp.end()
                    gtoks = [t for t, tok in enumerate(al.tokens) if tok.start >= gs and tok.end <= ge]
                    if gtoks and all(tok_ov[t] for t in gtoks):
                        prefix = l1[: mp.start()].strip()
                        g1, g2, g3 = mp.group(1).strip(), mp.group(2).strip(), mp.group(3).strip()
                        rep = (f"{prefix}\n" if prefix else "") + f"Country code: {g1}\nArea code: {g2}\nNumber: {g3}"
                        tail = l1[mp.end() :].strip()
                        if tail:
                            rep += f"\n{tail}"
                        edits.append((s0, e1, rep))
                        i = j + 1
                        continue
        i += 1

    # 4. Printed date-format hints immediately before an overlay date value
    for m in _DATE_HINT.finditer(md):
        s, e = m.start(), m.end()
        words = re.findall(r"[A-Za-z]+", m.group(0))
        if len(words) < 2 or (s > 0 and md[s - 1] == "("):
            continue
        htoks = [t for t, tok in enumerate(al.tokens) if tok.start >= s and tok.end <= e]
        if not htoks or any(tok_ov[t] for t in htoks):
            continue
        lo_n = md.find("\n", e)
        if lo_n < 0:
            lo_n = len(md)
        nxt = next((t for t, tok in enumerate(al.tokens) if tok.start >= e and tok.end <= lo_n), None)
        if nxt is not None and tok_ov[nxt] and any(c.isdigit() for c in al.tokens[nxt].text):
            between = md[e : al.tokens[nxt].start]
            if re.fullmatch(r"(?:[ \t/:\-*_]|<br\s*/?>|</?(?:strong|b|u|em)>)*", between, re.I):
                e_cut = e
                while e_cut < al.tokens[nxt].start and md[e_cut] in " \t/":
                    e_cut += 1
                edits.append((s, e_cut, ""))

    return edits

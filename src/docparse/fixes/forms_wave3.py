"""Field-note and parenthetical-example repairs on filled forms (option `field_notes`).

On filled forms, printed example text, parenthetical hints, and trailing box codes often sit where a
downstream reader expects the typed value, causing critical invented-value or wrong-value errors:
1. Parenthetical examples inside printed-only `<td>` cells (`Namenszusatz (Beispiel: Freifrau, Graf)`,
   `(z. B. ...)`, `(e.g. ...)`, `(for example ...)`, `(Leave blank)`) -> stripped when a short caption remains.
2. 2-cell KV rows `<td>Caption (note)</td><td>OVERLAY</td>` -> `<td colspan="2">Caption (note): OVERLAY</td>`.
3. Multi-column KV rows `<td>Caption</td><td></td><td>OVERLAY</td><td>CODE</td>` (or `Caption: | | OVERLAY`)
   -> merged with the trailing printed code moved before the caption.
4. Blank lines ending with `<italic numeric note> <CODE>: [___]` (`Recettes ... *...15 000 €* 4BE: [___]`)
   -> `4BE Recettes ...: [___]`.
"""

from __future__ import annotations

import html
import re

from docparse.fixes.align import fold
from docparse.fixes.forms import FormPage
from docparse.fixes.forms_tables import _cell_spans, _cell_toks, _classify_cell, _clean_code, _token_flags
from docparse.fixes.tables_html import HCell, parse_table

_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_PAREN_END = re.compile(r"\([^()\n]{4,80}\)\s*$")
_PAREN_EX = re.compile(
    r"\s*\([^()\n]*(?::|\bz\.\s*B\.|\be\.?\s*g\.?|\bfor\s+example\b|\bleave\s+blank\b|\bBeispiel\b)[^()\n]*\)",
    re.I,
)
_TRAIL_BOX_CODE = re.compile(
    r"^([ \t]*(?:[-*+][ \t]+)?)(.+?)[ \t]+(\*{0,2}([0-9][A-Z]{2}|[0-9]{1,2}[A-Z]{1,2})\*{0,2})[ \t]*:[ \t]*"
    r"(\[[ \t]*_{2,}[ \t]*\]|_{3,})[ \t]*$"
)
_ITALIC_NOTE_WITH_NUM = re.compile(r"[ \t]*\*[^*\n]*[0-9€$£%][^*\n]*\*[ \t]*$")
_TAG = re.compile(r"<[^>]+>")


def field_note_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """Repair printed parenthetical examples, paren-blocked KV rows, empty-spacer trailing-code rows,
    and trailing tax-code blank lines on filled form pages."""
    md, al = page.markdown, page.al
    tok_ov, _ = _token_flags(page)
    printed_words = {fold(w.text) for w in page.printed}
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
        kinds: dict[int, str] = {}
        ctoks: dict[int, list[int]] = {}
        for row in table.rows:
            for c in row.cells:
                sp = spans.get(id(c))
                ct = _cell_toks(al, sp[0], sp[1]) if sp else []
                ctoks[id(c)] = ct
                kinds[id(c)] = _classify_cell(c, ct, tok_ov)

        changed = False

        # 1. Strip parenthetical examples inside printed-only `<td>` cells (`(Beispiel: ...)`, `(z. B. ...)`, etc.)
        for row in table.rows:
            for cell in row.cells:
                if cell.tag != "td" or "(" not in cell.text:
                    continue
                ct = ctoks[id(cell)]
                if not ct or any(tok_ov[t] for t in ct):
                    continue
                new_in, n = _PAREN_EX.subn("", cell.inner)
                if n:
                    plain_rem = _TAG.sub(" ", new_in).split()
                    if 1 <= len(plain_rem) <= 14:
                        cell.inner = new_in.strip()
                        cell.raw = ""
                        row.edited = True
                        changed = True

        has_th = (
            bool(table.rows)
            and all(c.tag == "th" for c in table.rows[0].cells)
            and sum(bool(c.text.strip()) for c in table.rows[0].cells) >= 2
        )
        multi_ov_th = has_th and (
            len({c.col for row in table.rows[1:] for c in row.cells if kinds[id(c)] in ("OVERLAY", "MIXED_OV")}) >= 2
        )

        # 2. 2-cell KV rows where caption ends with a parenthetical note/formula (`22. Chassis No. (Affix...) | VAL`)
        if not has_th:
            for r_idx, row in enumerate(table.rows):
                if len(row.cells) != 2 or any(c.rowspan > 1 for c in row.cells) or table.spanned_into(r_idx):
                    continue
                c0, c1 = row.cells
                if kinds[id(c0)] == "CAPTION" and kinds[id(c1)] == "OVERLAY" and _PAREN_END.search(c0.text.strip()):
                    cap = c0.text.strip()
                    val = c1.text.strip()
                    span_sum = c0.colspan + c1.colspan
                    row.cells = [
                        HCell(
                            tag="td",
                            attrs=f' colspan="{span_sum}"',
                            inner=f"{html.escape(cap)}: {html.escape(val)}",
                            raw="",
                            colspan=span_sum,
                        )
                    ]
                    row.edited = True
                    changed = True

        # 3. `CAPTION | EMPTY* | OVERLAY | EMPTY* | CODE` or `CAPTION: | EMPTY+ | OVERLAY`
        if not multi_ov_th:
            for r_idx, row in enumerate(table.rows):
                if all(c.tag == "th" for c in row.cells):
                    continue
                if any(c.rowspan > 1 for c in row.cells) or table.spanned_into(r_idx):
                    continue
                cells = row.cells
                if len(cells) < 2:
                    continue
                ks = [kinds.get(id(c), "MERGED") for c in cells]
                if sum(k in ("OVERLAY", "MIXED_OV") for k in ks) > sum(k == "CAPTION" for k in ks):
                    continue
                new_cells: list[HCell] = []
                row_ch = False
                i = 0
                while i < len(cells):
                    if ks[i] != "CAPTION":
                        new_cells.append(cells[i])
                        i += 1
                        continue
                    j = i + 1
                    while j < len(cells) and ks[j] in ("EMPTY", "SYM"):
                        j += 1
                    if j < len(cells) and ks[j] == "OVERLAY":
                        k_end = j + 1
                        trail_codes: list[str] = []
                        while k_end < len(cells) and ks[k_end] in ("EMPTY", "SYM", "CODE"):
                            if ks[k_end] == "CODE":
                                trail_codes.append(_clean_code(cells[k_end].text))
                            k_end += 1
                        is_trail_code = len(trail_codes) == 1 and k_end == len(cells)
                        is_colon_gap = (
                            len(trail_codes) == 0 and j > i + 1 and cells[i].text.strip().endswith(":") and not has_th
                        )
                        if is_trail_code or is_colon_gap:
                            cap = cells[i].text.strip()
                            val = cells[j].text.strip()
                            cap_first = cap.split()[0].rstrip(".)") if cap.split() else ""
                            pre_c = [c for c in trail_codes if c and c.lower() != cap_first.lower()]
                            pref = (" ".join(pre_c) + " ") if pre_c else ""
                            sep = " " if cap.endswith(":") else ": "
                            span_sum = sum(cells[u].colspan for u in range(i, k_end))
                            new_cells.append(
                                HCell(
                                    tag=cells[i].tag,
                                    attrs=f' colspan="{span_sum}"' if span_sum > 1 else "",
                                    inner=f"{html.escape(pref + cap)}{sep}{html.escape(val)}",
                                    raw="",
                                    colspan=span_sum,
                                )
                            )
                            row_ch = True
                            i = k_end
                            continue
                    new_cells.append(cells[i])
                    i += 1
                if row_ch:
                    row.cells = new_cells
                    row.edited = True
                    changed = True

        if changed:
            table.changed = True
            edits.append((ts, te, table.render()))

    # 4. Plain-text blank lines ending with `<italic numeric note> <CODE>: [___]`
    pos = 0
    for raw_line in md.splitlines(keepends=True):
        ls = pos
        le = pos + len(raw_line.rstrip("\r\n"))
        pos += len(raw_line)
        line = md[ls:le]
        if not line.strip() or any(a <= ls < b for a, b in tranges):
            continue
        mc = _TRAIL_BOX_CODE.match(line)
        if not mc:
            continue
        bullet, cap, _cr, code_plain, blank = mc.groups()
        if fold(code_plain) not in printed_words:
            continue
        ltoks = [t for t, tok in enumerate(al.tokens) if tok.start >= ls and tok.end <= le]
        if any(tok_ov[t] for t in ltoks):
            continue
        cap_clean = _ITALIC_NOTE_WITH_NUM.sub("", cap).strip() or cap.strip()
        edits.append((ls, le, f"{bullet}{code_plain} {cap_clean}: {blank}"))

    return edits

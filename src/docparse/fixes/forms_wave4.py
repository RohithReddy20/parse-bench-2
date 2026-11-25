"""Form table-grid and watermark-comb repairs on filled forms (option `table_grids`).

Repairs four structural patterns on filled forms:
1. Box-numbered value grids (`<td><strong>3.30</strong> £ 164</td>`, `<td><strong>280.</strong><br/>53652 , 45</td>`)
   where a printed row caption sits in column 0 and bold box-code cells hold either an overlay value or a
   non-empty blank/punctuation placeholder.
2. Sequential column-number sub-header rows (`(1) | (2) | (3)...` or `(a) | (b) | (c)...`) merged into the
   leaf column headers directly above them so row 1 is not misread as data.
3. Printed `<br/>for example ...` / `e.g.` / `z. B.` / `Beispiel:` / `par exemple` example lines inside
   printed-only `<td>` caption cells.
4. Interleaved printed single-letter date-box watermarks (`d 2 d 7 m 0 m 9 y 2 y 0 y 2 y 6` -> `27 09 2026`)
   verified against the PDF's typed overlay fills.
"""

from __future__ import annotations

import html
import re

from docparse.fixes.align import fold
from docparse.fixes.forms import FormPage
from docparse.fixes.forms_tables import _cell_spans, _cell_toks, _classify_cell, _token_flags
from docparse.fixes.tables_html import HCell, HRow, HTable, parse_table

_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_FIRST_STRONG = re.compile(r"<(?:strong|b)>(.*?)</(?:strong|b)>", re.S | re.I)
_UNIT_TOK = re.compile(r"^(?:EUR|USD|GBP|CAD|AUD|CHF|JPY|km|kg|cm|mm|m2|ha|%|£|\$|€|¥|zł|gr|元)$", re.I)
_BOX_NUM_STRICT = re.compile(
    r"^\s*<(?:strong|b)>([0-9]{1,4}\.|[0-9]{1,2}\.[0-9]{1,3})</(?:strong|b)>\s*(?:<br\s*/?>|\s)*(.*)$",
    re.S | re.I,
)
_PUNCT_BLANK_NONEMPTY = re.compile(r"^[\s£$€¥.,·•\-–—_/|()\[\]]+$")
_COL_NUM = re.compile(r"^\(?([0-9]{1,2}|[a-z])\)?\.?$")
_BR_EXAMPLE = re.compile(
    r"(?:<br\s*/?>\s*)+(?:<(?:em|i|small)>)?(?:for\s+example|e\.?\s*g\.?|z\.\s*B\.|Beispiel\s*:|par\s+exemple)\b.*$",
    re.S | re.I,
)
_DMY_WATERMARK = re.compile(r"(?:\b[Xx]\s+){0,2}(?:[dmyjtaDMYJTA]\s+[0-9]\s+){3,}[dmyjtaDMYJTA]\s+[0-9]\b")


def _plain(s: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", s)).split())


def _first_title(cell: HCell) -> str:
    m = _FIRST_STRONG.search(cell.inner)
    if m:
        t = _plain(m.group(1))
        if len(t) >= 4:
            return t
    for part in re.split(r"<br\s*/?>", cell.inner, flags=re.I):
        t = _plain(part)
        if t:
            return t
    return cell.text.strip()


def _merge_col_num_header(table: HTable, ctoks: dict[int, list[int]], tok_ov: list[bool]) -> bool:
    """Merge a printed-only sequential column-number row in rows 1 or 2 into the leaf headers above it."""
    if len(table.rows) < 3:
        return False
    for num_r in (1, 2):
        if num_r >= len(table.rows) - 1:
            continue
        r_num = table.rows[num_r]
        if len(r_num.cells) < 3 or any(c.colspan > 1 or c.rowspan > 1 for c in r_num.cells):
            continue
        if any(r_idx + c.rowspan > num_r for r_idx, rr in enumerate(table.rows[:num_r]) for c in rr.cells):
            continue
        vals: list[str] = []
        ok = True
        for c in r_num.cells:
            txt = c.text.strip()
            if not txt:
                vals.append("")
                continue
            if not _COL_NUM.fullmatch(txt) or any(tok_ov[t] for t in ctoks[id(c)]):
                ok = False
                break
            vals.append(txt)
        non_empty = [v for v in vals if v]
        if not ok or len(non_empty) < 3:
            continue
        cores = [m.group(1) for v in non_empty if (m := _COL_NUM.fullmatch(v))]
        is_num_seq = all(x.isdigit() for x in cores) and [int(x) for x in cores] == list(
            range(int(cores[0]), int(cores[0]) + len(cores))
        )
        is_let_seq = all(x.isalpha() for x in cores) and [ord(x) for x in cores] == list(
            range(ord(cores[0]), ord(cores[0]) + len(cores))
        )
        if not (is_num_seq or is_let_seq):
            continue
        if any(tok_ov[t] for rr in table.rows[:num_r] for c in rr.cells for t in ctoks[id(c)]):
            continue
        leaf_by_col: dict[int, tuple[HRow, HCell]] = {}
        for rr in table.rows[:num_r]:
            for c in rr.cells:
                if c.colspan == 1:
                    leaf_by_col[c.col] = (rr, c)
        if not all(c.col in leaf_by_col for c in r_num.cells if c.text.strip()):
            continue
        for c in r_num.cells:
            v = c.text.strip()
            if v:
                rr, hdr_c = leaf_by_col[c.col]
                if v not in hdr_c.text:
                    hdr_c.inner = f"{html.escape(v)} {hdr_c.inner.strip()}" if hdr_c.inner.strip() else html.escape(v)
                    hdr_c.raw = ""
                    rr.edited = True
        table.rows.pop(num_r)
        table.seps.pop(num_r)
        return True
    return False


def _strip_cell_examples(table: HTable, ctoks: dict[int, list[int]], tok_ov: list[bool]) -> bool:
    """Strip `<br/>for example ...` printed example lines inside printed-only `<td>` cells."""
    changed = False
    for row in table.rows:
        for cell in row.cells:
            if cell.tag != "td" or "<br" not in cell.inner.lower():
                continue
            ct = ctoks[id(cell)]
            if not ct or any(tok_ov[t] for t in ct):
                continue
            new_in, n = _BR_EXAMPLE.subn("", cell.inner)
            if n:
                rem = _TAG.sub(" ", new_in).split()
                if 1 <= len(rem) <= 16:
                    cell.inner = new_in.strip()
                    cell.raw = ""
                    row.edited = True
                    changed = True
    return changed


def _format_box_num_grid(
    table: HTable,
    page: FormPage,
    kinds: dict[int, str],
    ctoks: dict[int, list[int]],
    tok_ov: list[bool],
    tok_pr: list[bool],
    printed_words: set[str],
    fill_pair_keys: set[str],
) -> bool:
    """Rewrite bold box-numbered value cells (`<strong>3.30</strong> £ 164`, `<strong>280.</strong><br/>53652 , 45`)
    into explicit `code row_title [— col_hdr]: value` cells."""
    if len(table.rows) < 2:
        return False
    md, al = page.markdown, page.al
    col_hdr: dict[int, str] = {}
    row_hits: list[tuple[HRow, HCell, str, dict[int, str], list[tuple[HCell, str, str]]]] = []
    for r_idx, row in enumerate(table.rows):
        cells = row.cells
        if not cells:
            continue
        if all(c.tag == "th" for c in cells):
            for c in cells:
                t = _plain(c.inner)
                if c.colspan == 1 and t and not _UNIT_TOK.fullmatch(t.rstrip(",.")):
                    col_hdr[c.col] = t
            continue
        if (
            cells[0].col == 0
            and kinds[id(cells[0])] == "EMPTY"
            and sum(kinds[id(c)] == "CAPTION" and "<strong" in c.inner.lower() for c in cells[1:]) >= 2
            and not any(kinds[id(c)] in ("OVERLAY", "MIXED_OV") for c in cells)
        ):
            for c in cells[1:]:
                t = _plain(c.inner)
                if c.colspan == 1 and t:
                    col_hdr[c.col] = t
            continue
        if table.spanned_into(r_idx) or any(c.rowspan > 1 for c in cells) or kinds[id(cells[0])] != "CAPTION":
            continue
        row_title = _first_title(cells[0])
        box_cells: list[tuple[HCell, str, str]] = []
        for c in cells[1:]:
            mb = _BOX_NUM_STRICT.match(c.inner)
            if not mb:
                continue
            code_s, tail_html = mb.group(1), mb.group(2)
            ts_list = ctoks[id(c)]
            if not ts_list or tok_ov[ts_list[0]]:
                continue
            if not (tok_pr[ts_list[0]] or fold(code_s.rstrip(".")) in printed_words):
                continue
            ov_toks = [t for t in ts_list if tok_ov[t]]
            tail_plain = _plain(tail_html)
            tail_key = "".join(fold(ch) for ch in tail_plain)
            if ov_toks:
                val_str = md[al.tokens[ov_toks[0]].start : al.tokens[ov_toks[-1]].end]
                val_str = re.sub(r"(\d)[ \t]+,[ \t]+(\d)", r"\1,\2", val_str)
                box_cells.append((c, code_s, val_str))
            elif tail_key and tail_key in fill_pair_keys and tail_key not in printed_words:
                val_str = re.sub(r"(\d)[ \t]+,[ \t]+(\d)", r"\1,\2", tail_plain)
                box_cells.append((c, code_s, val_str))
            elif tail_plain and (
                _PUNCT_BLANK_NONEMPTY.fullmatch(tail_plain) or re.fullmatch(r"[£$€¥]?\s*\[\s*\]\s*", tail_plain)
            ):
                box_cells.append((c, code_s, "[___]"))
        if box_cells:
            row_hits.append((row, cells[0], row_title, dict(col_hdr), box_cells))
    total_box = sum(len(bc) for *_, bc in row_hits)
    if total_box < 2 or (len(row_hits) < 2 and max(len(bc) for *_, bc in row_hits) < 2):
        return False
    for row, c0, row_title, ch_map, box_cells in row_hits:
        for c, code_s, val_str in box_cells:
            ch = ch_map.get(c.col, "")
            cap = f"{code_s} {row_title} — {ch}" if ch else f"{code_s} {row_title}"
            c.inner = f"{html.escape(cap)}: {html.escape(val_str)}"
            c.raw = ""
        if "<br" in c0.inner.lower():
            c0.inner = ""
            c0.raw = ""
        row.edited = True
    return True


def table_grid_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """Return non-overlapping `(start, end, replacement)` edits for `table_grids`."""
    md, al = page.markdown, page.al
    tok_ov, tok_pr = _token_flags(page)
    printed_words = {fold(w.text) for w in page.printed}
    fill_pair_keys = {f.key for f in page.fills if len(f.key) >= 2}
    for a, b in zip(page.fills, page.fills[1:], strict=False):
        fill_pair_keys.add(a.key + b.key)
    fill_cat = {f.key for f in page.fills}
    for i in range(len(page.fills)):
        acc = ""
        for j in range(i, min(i + 4, len(page.fills))):
            acc += page.fills[j].key
            fill_cat.add(acc)

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

        changed = _merge_col_num_header(table, ctoks, tok_ov)
        changed = _strip_cell_examples(table, ctoks, tok_ov) or changed
        changed = (
            _format_box_num_grid(table, page, kinds, ctoks, tok_ov, tok_pr, printed_words, fill_pair_keys) or changed
        )
        if changed:
            table.changed = True
            edits.append((ts, te, table.render()))

    for m in _DMY_WATERMARK.finditer(md):
        s, e = m.start(), m.end()
        if any(a <= s < b for a, b in tranges):
            continue
        pairs = re.findall(r"([dmyjtaDMYJTA])\s+([0-9])", m.group(0))
        if len(pairs) < 4:
            continue
        if "".join(d for _, d in pairs) not in fill_cat:
            continue
        groups: list[str] = []
        cur_l, cur_d = "", ""
        for lt, dg in pairs:
            ll = lt.lower()
            if cur_l and ll != cur_l and cur_d:
                groups.append(cur_d)
                cur_d = ""
            cur_l = ll
            cur_d += dg
        if cur_d:
            groups.append(cur_d)
        edits.append((s, e, " ".join(groups)))

    return edits

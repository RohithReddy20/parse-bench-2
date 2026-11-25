"""Key-value form table repairs from the PDF's typed overlay (option `table_rows`).

On filled forms whose fields are laid out in HTML tables, a page read often puts printed box codes,
currency signs, or pre-printed cents placeholders between a caption cell and its typed value (or blank
slot), or lays a 2-row caption-over-value box grid, or repeats empty multi-label template cells:

- `<td>Line 119 plus line 120</td><td>40400</td><td>=</td><td>5,452.47</td><td></td><td>121</td>`
  -> `<td colspan="6">40400 121 Line 119 plus line 120: 5,452.47</td>`
- `<td>1 ,00</td>` (printed only) -> `<td></td>`; `<td>2 2581 ,00</td>` (code + overlay + cents) -> `<td>2581</td>`
- 2-row caption-over-value form box -> single row of `caption: value` (or `[___]`) cells
- Repeated (>= 3x) empty `<td>` multi-label template cells (`NIK/NPWP:<br/>NAMA:`) -> cleared.

Every repair acts only on pages with a PDF overlay and consults the overlay/printed token alignment.
"""

from __future__ import annotations

import html
import re
from collections import Counter

from docparse.fixes.forms import FormPage
from docparse.fixes.tables_html import HCell, HRow, HTable, parse_table

_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_CODE_PART = re.compile(r"^[•·*\-(]?(?:[0-9]{1,5}[A-Za-z]?|[A-Za-z]|[0-9]{1,2}\.[0-9]{1,3}|[0-9]{1,2}[A-Za-z])[.)]?$")
_SYM_ONLY = re.compile(r"^[\s£$€¥%+=×÷•·.,:;\-–—/()\[\]_]*$")
_CENTS_ONLY = re.compile(r"^[.,]\s*00\s*[£$€¥%]?$|^[£$€¥]\s*$")
_BLANK_PLACEHOLDER = re.compile(r"^(?:\[_{2,}\]|_{3,}|,00|\.00|\.\s*00|,\s*00|—|–|-|\s)*$")
_TRAILING_CODE_CENTS = re.compile(r"^(.+?)\s+([0-9]{1,3})\s+[.,]\s*00\s*$")
_PURE_CODE_CENTS = re.compile(r"^[0-9]{1,3}\s*[.,]\s*00\s*$")
_HAS_CENTS_END = re.compile(r"[.,]\s*00\s*$")
_HAS_BOX = re.compile(r"\[[ \t]*(?:[xX✓✔])?[ \t]*\]|[☐☑☒□■]")
_BR_SPLIT = re.compile(r"<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")


def _token_flags(page: FormPage) -> tuple[list[bool], list[bool]]:
    al = page.al
    n = len(al.tokens)
    ov = [False] * n
    pr = [False] * n
    for k, p in enumerate(al.amatch):
        if p is not None:
            t = al.a_tok[k]
            if page.owner[al.p_word[p]] >= 0:
                ov[t] = True
            else:
                pr[t] = True
    for _, (t0, t1) in page.placed.items():
        for t in range(t0, t1 + 1):
            ov[t] = True
    return ov, pr


def _cell_spans(raw_table: str, table_start: int, table: HTable) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    pos = 0
    for row in table.rows:
        r_idx = raw_table.find(row.head, pos)
        if r_idx < 0:
            continue
        pos = r_idx + len(row.head)
        for cell in row.cells:
            c_idx = raw_table.find(cell.raw, pos) if cell.raw else -1
            if c_idx < 0:
                continue
            out[id(cell)] = (table_start + c_idx, table_start + c_idx + len(cell.raw))
            pos = c_idx + len(cell.raw)
    return out


def _cell_toks(al, cs: int, ce: int) -> list[int]:
    toks = al.tokens
    lo, hi = 0, len(toks)
    while lo < hi:
        mid = (lo + hi) // 2
        if toks[mid].end <= cs:
            lo = mid + 1
        else:
            hi = mid
    res: list[int] = []
    i = lo
    while i < len(toks) and toks[i].start < ce:
        if toks[i].start >= cs and toks[i].end <= ce:
            res.append(i)
        i += 1
    return res


def _is_code_text(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    parts = t.split()
    if len(parts) > 2:
        return False
    clean = [p for p in parts if p not in ("•", "·", "-", "–", "+")]
    if not clean or len(clean) > 2:
        return False
    return all(_CODE_PART.fullmatch(p) for p in clean)


def _clean_code(text: str) -> str:
    parts = [p.strip("•·*()-") for p in text.strip().split()]
    return " ".join(p for p in parts if p)


def _classify_cell(cell: HCell, toks: list[int], tok_ov: list[bool]) -> str:
    txt = cell.text.strip()
    if not txt or _BLANK_PLACEHOLDER.fullmatch(txt):
        return "EMPTY"
    if _HAS_BOX.search(txt):
        return "BOX"
    if _SYM_ONLY.fullmatch(txt) or _CENTS_ONLY.fullmatch(txt):
        return "SYM"
    n_ov = sum(1 for t in toks if tok_ov[t])
    if n_ov > 0:
        return "OVERLAY" if n_ov == len(toks) else "MIXED_OV"
    if _PURE_CODE_CENTS.fullmatch(txt):
        return "CODE_CENTS"
    if _is_code_text(txt):
        return "CODE"
    return "CAPTION"


def _row_has_kv_code_pattern(row: HRow, kinds: dict[int, str]) -> bool:
    ks = [kinds[id(c)] for c in row.cells]
    for i, k in enumerate(ks):
        if k != "CAPTION":
            continue
        j = i + 1
        saw_code = False
        while j < len(ks) and ks[j] in ("CODE", "SYM", "EMPTY"):
            if ks[j] == "CODE":
                saw_code = True
            j += 1
        if saw_code and j < len(ks) and ks[j] == "OVERLAY":
            return True
    return False


def _is_multi_val_header(table: HTable, kinds: dict[int, str]) -> bool:
    ov_cols: set[int] = set()
    for row in table.rows[1:]:
        for cell in row.cells:
            if kinds[id(cell)] in ("OVERLAY", "MIXED_OV"):
                ov_cols.add(cell.col)
    hdr_groups_with_val = 0
    for hcell in table.rows[0].cells:
        if not hcell.text.strip():
            continue
        cols_covered = set(range(hcell.col, hcell.col + hcell.colspan))
        has_val = bool(cols_covered & ov_cols) or any(
            cell.col in cols_covered and kinds[id(cell)] == "SYM" and any(ch in cell.text for ch in "£$€¥")
            for row in table.rows[1:]
            for cell in row.cells
        )
        if has_val:
            hdr_groups_with_val += 1
    return hdr_groups_with_val >= 2 or len(ov_cols) >= 2


def table_row_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """Repair HTML form tables using overlay/printed alignment on filled form pages."""
    md, al = page.markdown, page.al
    tok_ov, tok_pr = _token_flags(page)
    edits: list[tuple[int, int, str]] = []

    for m in _HTML_TABLE.finditer(md):
        ts, te = m.start(), m.end()
        raw_t = m.group(0)
        table = parse_table(raw_t)
        if table is None:
            continue
        spans = _cell_spans(raw_t, ts, table)

        kinds: dict[int, str] = {}
        ctoks: dict[int, list[int]] = {}
        for row in table.rows:
            for cell in row.cells:
                sp = spans.get(id(cell))
                ts_list = _cell_toks(al, sp[0], sp[1]) if sp else []
                ctoks[id(cell)] = ts_list
                kinds[id(cell)] = _classify_cell(cell, ts_list, tok_ov)

        changed = False

        # 1. Repeated (>=3x) empty `<td>` multi-label template cells (`NIK/NPWP:<br/>NAMA:`)
        br_counts: Counter[str] = Counter()
        br_cells: list[tuple[HRow, HCell, str]] = []
        for row in table.rows:
            for cell in row.cells:
                if cell.tag != "td" or kinds[id(cell)] != "CAPTION" or "<br" not in cell.inner.lower():
                    continue
                parts = _BR_SPLIT.split(cell.inner)
                vis = [" ".join(html.unescape(_TAG.sub("", p)).split()) for p in parts]
                non_empty = [v for v in vis if v]
                if len(non_empty) >= 2 and all(v.endswith(":") and len(v) <= 40 for v in non_empty):
                    key = "|".join(non_empty)
                    br_counts[key] += 1
                    br_cells.append((row, cell, key))
        for row, cell, key in br_cells:
            if br_counts[key] >= 3:
                cell.inner = ""
                cell.raw = ""
                kinds[id(cell)] = "EMPTY"
                row.edited = True
                changed = True

        # 2. Printed code+cents cells (`1 ,00` -> ``, `Importo rata 1 ,00` -> `1 Importo rata: [___]`)
        #    and mixed leading printed code + overlay + `,00` (`2 2581 ,00` -> `2581`)
        for row in table.rows:
            for cell in row.cells:
                if cell.tag != "td":
                    continue
                k = kinds[id(cell)]
                if k == "CODE_CENTS":
                    cell.inner = ""
                    cell.raw = ""
                    kinds[id(cell)] = "EMPTY"
                    row.edited = True
                    changed = True
                elif k == "CAPTION":
                    mc = _TRAILING_CODE_CENTS.match(cell.text.strip())
                    if mc and any(ch.isalpha() for ch in mc.group(1)):
                        cap, code = mc.group(1).strip(), mc.group(2)
                        cell.inner = f"{html.escape(code)} {html.escape(cap)}: [___]"
                        cell.raw = ""
                        row.edited = True
                        changed = True
                elif k == "MIXED_OV" and _HAS_CENTS_END.search(cell.text.strip()):
                    ts_list = ctoks[id(cell)]
                    if (
                        len(ts_list) >= 2
                        and not tok_ov[ts_list[0]]
                        and tok_pr[ts_list[0]]
                        and re.fullmatch(r"[0-9]{1,3}", al.tokens[ts_list[0]].text)
                    ):
                        ov_toks = [t for t in ts_list[1:] if tok_ov[t]]
                        non_ov_rest = [t for t in ts_list[1:] if not tok_ov[t]]
                        if ov_toks and all(al.tokens[t].text == "00" for t in non_ov_rest):
                            val = md[al.tokens[ov_toks[0]].start : al.tokens[ov_toks[-1]].end]
                            cell.inner = html.escape(val)
                            cell.raw = ""
                            kinds[id(cell)] = "OVERLAY"
                            row.edited = True
                            changed = True

        # 3. 2-row caption-over-value form box
        cap_over_applied = False
        if len(table.rows) == 2:
            r0, r1 = table.rows[0], table.rows[1]
            if (
                all(c.rowspan == 1 for c in r0.cells + r1.cells)
                and len(r0.cells) == len(r1.cells) >= 2
                and all(c0.colspan == c1.colspan for c0, c1 in zip(r0.cells, r1.cells, strict=True))
                and all(kinds[id(c)] in ("CAPTION", "EMPTY", "CODE") for c in r0.cells)
                and sum(kinds[id(c)] == "CAPTION" for c in r0.cells) >= 2
                and not any(_HAS_BOX.search(c.text) for c in r0.cells)
                and any(kinds[id(c)] in ("OVERLAY", "MIXED_OV") for c in r1.cells)
                and all(kinds[id(c)] in ("OVERLAY", "MIXED_OV", "EMPTY", "SYM") for c in r1.cells)
            ):
                new_cells = []
                for c0, c1 in zip(r0.cells, r1.cells, strict=True):
                    cap = c0.text.strip()
                    val = c1.text.strip() if kinds[id(c1)] in ("OVERLAY", "MIXED_OV") else "[___]"
                    if cap:
                        sep = " " if cap.endswith(":") else ": "
                        inner = f"{html.escape(cap)}{sep}{html.escape(val)}"
                    else:
                        inner = c1.inner
                    new_cells.append(HCell(tag="td", attrs=c0.attrs, inner=inner, raw="", colspan=c0.colspan))
                r0.cells = new_cells
                r0.edited = True
                table.rows = [r0]
                table.seps = [table.seps[0], table.seps[-1]]
                changed = True
                cap_over_applied = True

        # 4. Key-value table rows with printed box codes between caption and overlay value (or blank)
        if not cap_over_applied:
            has_th_header = (
                bool(table.rows)
                and all(c.tag == "th" for c in table.rows[0].cells)
                and sum(bool(c.text.strip()) for c in table.rows[0].cells) >= 2
            )
            multi_val_header = has_th_header and _is_multi_val_header(table, kinds)
            has_kv_filled = any(_row_has_kv_code_pattern(row, kinds) for row in table.rows)

            if not multi_val_header:
                for r_idx, row in enumerate(table.rows):
                    if all(c.tag == "th" for c in row.cells):
                        continue
                    if any(c.rowspan > 1 for c in row.cells) or table.spanned_into(r_idx):
                        continue
                    cells = row.cells
                    ks = [kinds[id(c)] for c in cells]
                    if sum(k in ("OVERLAY", "MIXED_OV") for k in ks) > sum(k == "CAPTION" for k in ks):
                        continue
                    new_cells = []
                    row_changed = False
                    i = 0
                    while i < len(cells):
                        if ks[i] != "CAPTION" or _HAS_BOX.search(cells[i].text):
                            new_cells.append(cells[i])
                            i += 1
                            continue
                        j = i + 1
                        codes: list[str] = []
                        saw_sym = False
                        saw_empty = False
                        while j < len(cells) and ks[j] in ("CODE", "SYM", "EMPTY"):
                            if ks[j] == "CODE":
                                codes.append(_clean_code(cells[j].text))
                            elif ks[j] == "SYM":
                                saw_sym = True
                            else:
                                saw_empty = True
                            j += 1

                        if j < len(cells) and ks[j] == "OVERLAY" and (codes or (saw_sym and not has_th_header)):
                            val_text = cells[j].text.strip()
                            k_end = j + 1
                            trailing_codes: list[str] = []
                            if (
                                k_end < len(cells)
                                and ks[k_end] == "SYM"
                                and _CENTS_ONLY.fullmatch(cells[k_end].text.strip())
                            ):
                                k_end += 1
                            rem_ks = ks[k_end:]
                            if (
                                rem_ks
                                and all(rk in ("EMPTY", "SYM", "CODE") for rk in rem_ks)
                                and rem_ks.count("CODE") <= 1
                            ):
                                for kk in range(k_end, len(cells)):
                                    if ks[kk] == "CODE":
                                        trailing_codes.append(_clean_code(cells[kk].text))
                                k_end = len(cells)

                            cap_text = cells[i].text.strip()
                            all_codes = [c for c in codes + trailing_codes if c]
                            cap_first = cap_text.split()[0].rstrip(".)") if cap_text.split() else ""
                            prefix_codes = [c for c in all_codes if c.lower() != cap_first.lower()]
                            prefix = (" ".join(prefix_codes) + " ") if prefix_codes else ""
                            sep = " " if cap_text.endswith(":") else ": "
                            span_sum = sum(cells[u].colspan for u in range(i, k_end))
                            attrs = f' colspan="{span_sum}"' if span_sum > 1 else ""
                            new_cells.append(
                                HCell(
                                    tag=cells[i].tag,
                                    attrs=attrs,
                                    inner=f"{html.escape(prefix + cap_text)}{sep}{html.escape(val_text)}",
                                    raw="",
                                    colspan=span_sum,
                                )
                            )
                            row_changed = True
                            i = k_end
                            continue

                        if (
                            codes
                            and (has_kv_filled or saw_sym or saw_empty)
                            and (j == len(cells) or ks[j] == "CAPTION")
                        ):
                            k_end = j
                            if (
                                j < len(cells)
                                and ks[j] == "CAPTION"
                                and k_end - 1 > i
                                and ks[k_end - 1] == "CODE"
                                and ks[k_end - 2] == "EMPTY"
                            ):
                                k_end -= 1
                                codes = codes[:-1]
                            if codes:
                                cap_text = cells[i].text.strip()
                                cap_first = cap_text.split()[0].rstrip(".)") if cap_text.split() else ""
                                prefix_codes = [c for c in codes if c and c.lower() != cap_first.lower()]
                                prefix = (" ".join(prefix_codes) + " ") if prefix_codes else ""
                                sep = " " if cap_text.endswith(":") else ": "
                                span_sum = sum(cells[u].colspan for u in range(i, k_end))
                                attrs = f' colspan="{span_sum}"' if span_sum > 1 else ""
                                new_cells.append(
                                    HCell(
                                        tag=cells[i].tag,
                                        attrs=attrs,
                                        inner=f"{html.escape(prefix + cap_text)}{sep}[___]",
                                        raw="",
                                        colspan=span_sum,
                                    )
                                )
                                row_changed = True
                                i = k_end
                                continue

                        new_cells.append(cells[i])
                        i += 1

                    if row_changed:
                        row.cells = new_cells
                        row.edited = True
                        changed = True

        if changed:
            table.changed = True
            edits.append((ts, te, table.render()))

    return edits

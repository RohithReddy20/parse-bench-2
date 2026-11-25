"""AcroForm fillable-PDF form repairs (option `acroform`).

Acts ONLY on PDF pages that contain interactive AcroForm text/checkbox/radio widgets (>= 3 widgets):
1. Merges a dedicated leading row-ID table column (`<td>3</td><td>Caption</td>`) into the caption cell
   (`<td colspan="2">3 Caption</td>`), resolving sub-item letters (`<td>a</td>` -> `8a`) and tracking
   col-0/1 rowspans only so right-hand spanning columns never block the merge.
2. Expands inline sub-letter captions when the adjacent right cell holds the full compound ID
   (`<td><strong>b</strong> Taxable interest</td><td><strong>2b</strong></td>` -> `2b Taxable interest`).
3. Inserts `[ ] ` or `[x] ` beside aligned printed words in mark-free units when an AcroForm checkbox
   or radio widget sits immediately to the word's left on the same PDF line.
"""

from __future__ import annotations

import ctypes
import html
import re

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from docparse.fixes.align import align
from docparse.fixes.forms import apply_edits
from docparse.fixes.layout_refine import PtWord
from docparse.fixes.tables_html import HCell, parse_table
from docparse.pdf.render import RenderedPage

_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_ID_MAIN = re.compile(r"^[0-9]{1,3}[a-z]?\.?$", re.I)
_ID_SUB = re.compile(r"^[a-z]\.?$|^[ivx]{1,4}\.?$", re.I)
_ID_COMPOUND = re.compile(r"^[0-9]{1,3}[a-z]$", re.I)
_SUB_PREFIX = re.compile(r"^(\s*(?:<(?:strong|b)>)?)\s*([a-z])\s*(\.?)(?:</(?:strong|b)>)?(\s+)", re.I)
_HAS_BOX = re.compile(r"\[[ \t]*(?:[xX✓✔]|checked|unchecked)?[ \t]*\]|[☐☑☒□■○◯●◉]", re.I)
_UNIT_END = re.compile(r"\n|<br[ \t]*/?>|</?t[dhr][^>\n]*>|(?<!\\)\|")
_ACRO_NUM_BLANK = re.compile(r"^\s*(?:<(?:strong|b)>)?[0-9]{1,2}[.)](?:</(?:strong|b)>)?\s*\[_{2,}\]\s*$", re.I)
_ACRO_MID_BLANK_NOTE = re.compile(r"(:\s*\[_{2,}\])[ \t]+(\*\*[^*\n]{8,80}[0-9]{1,3}[a-z]?\.\*\*[ \t]*\[_{2,}\])")
_ACRO_TRAIL_PAREN = re.compile(r"(:[ \t]*\[_{2,}\])[ \t]+(\([^()\n]{4,80}\))[ \t]*$", re.M)
_ACRO_TRAIL_INSTR = re.compile(r"(\[_{2,}\])[ \t]+(See\s+(?:separate\s+)?instructions\.?)[ \t]*$", re.M | re.I)
_ACRO_SSN_MASK = re.compile(r"(?:<(?:strong|b)>|\*{2})?\bX{2,3}-X{2}-(?:</(?:strong|b)>|\*{2})?[ \t]*(\[_{2,}\])")


def acroform_widgets(
    pdf_bytes: bytes, page_index: int = 0
) -> tuple[int, list[tuple[float, float, float, float, bool]]]:
    """Return (interactive_widget_count, checkbox/radio rects `(x0, y0, x1, y1, checked)` in top-down PDF pts)."""
    if not pdf_bytes:
        return 0, []
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        if page_index < 0 or page_index >= len(doc):
            return 0, []
        page = doc[page_index]
        n_ann = pdfium_c.FPDFPage_GetAnnotCount(page.raw)
        if n_ann <= 0:
            return 0, []
        h = page.get_height()
        form_info = pdfium_c.FPDF_FORMFILLINFO()
        form_info.version = 1
        fh = pdfium_c.FPDFDOC_InitFormFillEnvironment(doc.raw, ctypes.byref(form_info))
        try:
            n_interactive = 0
            cboxes: list[tuple[float, float, float, float, bool]] = []
            for ai in range(n_ann):
                ann = pdfium_c.FPDFPage_GetAnnot(page.raw, ai)
                if pdfium_c.FPDFAnnot_GetSubtype(ann) == pdfium_c.FPDF_ANNOT_WIDGET:
                    ft = pdfium_c.FPDFAnnot_GetFormFieldType(fh, ann)
                    if ft in (2, 3, 6):  # checkbox, radio, text field
                        n_interactive += 1
                    if ft in (2, 3):
                        rect = pdfium_c.FS_RECTF()
                        if pdfium_c.FPDFAnnot_GetRect(ann, ctypes.byref(rect)):
                            checked = bool(pdfium_c.FPDFAnnot_IsChecked(fh, ann))
                            x0, x1 = min(rect.left, rect.right), max(rect.left, rect.right)
                            y0, y1 = h - max(rect.top, rect.bottom), h - min(rect.top, rect.bottom)
                            if 3.0 <= x1 - x0 <= 25.0 and 3.0 <= y1 - y0 <= 25.0:
                                cboxes.append((x0, y0, x1, y1, checked))
                pdfium_c.FPDFPage_CloseAnnot(ann)
            return n_interactive, cboxes
        finally:
            pdfium_c.FPDFDOC_ExitFormFillEnvironment(fh)
    finally:
        doc.close()


def acro_id_edits(markdown: str) -> list[tuple[int, int, str]]:
    """Merge dedicated col-0 row-ID cells into col-1 captions and expand inline sub-letter captions."""
    edits: list[tuple[int, int, str]] = []
    for m in _HTML_TABLE.finditer(markdown):
        ts, te = m.start(), m.end()
        table = parse_table(m.group(0))
        if table is None or len(table.rows) < 2:
            continue
        span_rem: dict[int, int] = {}
        cur_main = ""
        plan: list[tuple[int, str]] = []
        changed = False

        for r_idx, row in enumerate(table.rows):
            blocked01 = (span_rem.get(0, 0) > 0) or (span_rem.get(1, 0) > 0)
            new_rem = {c: v - 1 for c, v in span_rem.items() if v > 1}
            for cell in row.cells:
                if cell.rowspan > 1:
                    for cc in range(cell.col, cell.col + cell.colspan):
                        new_rem[cc] = max(new_rem.get(cc, 0), cell.rowspan - 1)
            span_rem = new_rem

            for ci in range(len(row.cells) - 1):
                c_cap, c_id = row.cells[ci], row.cells[ci + 1]
                if c_cap.tag != "td" or c_id.tag != "td":
                    continue
                cid_txt = c_id.text.strip().rstrip(".")
                if not _ID_COMPOUND.fullmatch(cid_txt):
                    continue
                ms = _SUB_PREFIX.match(c_cap.inner)
                if ms and ms.group(2).lower() == cid_txt[-1].lower():
                    lead, _ltr, dot, sp = ms.groups()
                    end_tag = "</strong>" if "<strong" in lead.lower() else ("</b>" if "<b" in lead.lower() else "")
                    c_cap.inner = f"{lead}{html.escape(cid_txt)}{dot}{end_tag}{sp}{c_cap.inner[ms.end() :]}"
                    c_cap.raw = ""
                    row.edited = True
                    changed = True

            if blocked01 or len(row.cells) < 3:
                continue
            c0, c1 = row.cells[0], row.cells[1]
            if c0.tag != "td" or c1.tag != "td" or c0.col != 0 or c0.colspan != 1 or c0.rowspan > 1 or c1.rowspan > 1:
                continue
            t0 = c0.text.strip().rstrip(".")
            t1 = c1.text.strip()
            if not t0 or not t1 or len(t1) < 3 or _HAS_BOX.search(t0):
                continue
            if _ID_MAIN.fullmatch(t0):
                mn = re.match(r"^([0-9]{1,3})", t0)
                if mn:
                    cur_main = mn.group(1)
                full_id = t0
            elif _ID_SUB.fullmatch(t0) and len(t0) == 1 and cur_main:
                t2 = row.cells[2].text.strip().rstrip(".") if len(row.cells) >= 3 else ""
                full_id = (
                    t2
                    if (_ID_COMPOUND.fullmatch(t2) and t2.lower() == f"{cur_main}{t0.lower()}")
                    else f"{cur_main}{t0}"
                )
            else:
                continue
            if t1.lower().startswith(full_id.lower() + " "):
                continue
            plan.append((r_idx, full_id))

        if len(plan) >= 2:
            for r_idx, full_id in plan:
                row = table.rows[r_idx]
                c0, c1 = row.cells[0], row.cells[1]
                span_sum = c0.colspan + c1.colspan
                row.cells = [
                    HCell(
                        tag="td",
                        attrs=f' colspan="{span_sum}"',
                        inner=f"{html.escape(full_id)} {c1.inner.strip()}",
                        raw="",
                        colspan=span_sum,
                    )
                ] + row.cells[2:]
                row.edited = True
                changed = True

        if changed:
            table.changed = True
            edits.append((ts, te, table.render()))

    return edits


def acro_checkbox_edits(
    markdown: str, rendered: RenderedPage, cboxes: list[tuple[float, float, float, float, bool]]
) -> list[tuple[int, int, str]]:
    """Insert `[ ] ` or `[x] ` beside aligned printed words in mark-free units when an AcroForm checkbox/radio
    widget sits immediately to the word's left on the same PDF line."""
    if not cboxes or not rendered.text_words:
        return []
    s = rendered.scale
    words = [PtWord(w.text, w.box[0] * s, w.box[1] * s, w.box[2] * s, w.box[3] * s) for w in rendered.text_words]
    al = align(markdown, words, reorder=True)
    edits: list[tuple[int, int, str]] = []
    bounds = [0] + [m.end() for m in _UNIT_END.finditer(markdown)] + [len(markdown)]
    t_ptr = 0
    for u0, u1 in zip(bounds, bounds[1:], strict=False):
        while t_ptr < len(al.tokens) and al.tokens[t_ptr].start < u0:
            t_ptr += 1
        mine: list[int] = []
        k = t_ptr
        while k < len(al.tokens) and al.tokens[k].end <= u1:
            if al.tokens[k].start >= u0:
                mine.append(k)
            k += 1
        if not mine or _HAS_BOX.search(markdown[u0:u1]):
            continue
        used_boxes: set[int] = set()
        for t in mine:
            tws = al.token_words(t)
            if not tws:
                continue
            w0 = al.words[tws[0]]
            h = max(6.0, w0.y1 - w0.y0)
            for bi, (_bx0, by0, bx1, by1, chk) in enumerate(cboxes):
                if bi in used_boxes:
                    continue
                bcy = (by0 + by1) / 2
                if w0.y0 - 0.6 * h <= bcy <= w0.y1 + 0.6 * h and bx1 <= w0.x0 + 2.0 and 0 <= w0.x0 - bx1 <= 2.5 * h:
                    if not any(bx1 < p.cx < w0.x0 and abs(p.cy - w0.cy) < 0.5 * h for p in words):
                        mark = "[x] " if chk else "[ ] "
                        edits.append((al.tokens[t].start, al.tokens[t].start, mark))
                        used_boxes.add(bi)
                        break
    return edits


def acro_clean_markdown(markdown: str, *, has_boxes: bool = False) -> str:
    """Clean blank-field artifacts on AcroForm fillable PDFs:
    1. Strip printed row indices (`1) [___]`, `2. [___]`) inside blank table cells -> `[___]`.
    2. Fill empty `Yes`/`No` table columns with `[ ] Yes` / `[ ] No` when check boxes exist on the page.
    3. Split inline bold notes / parenthetical hints / `See instructions` that trail a `[___]` placeholder.
    4. Strip printed `XXX-XX-` mask prefixes right before `[___]`."""

    def _fix_table(m: re.Match[str]) -> str:
        t = parse_table(m.group(0))
        if t is None:
            return m.group(0)
        ch = False
        for row in t.rows:
            for c in row.cells:
                if c.tag == "td" and _ACRO_NUM_BLANK.fullmatch(c.inner):
                    c.inner = "[___]"
                    c.raw = ""
                    row.edited = True
                    ch = True
        if has_boxes and len(t.rows) >= 3:
            hdr = t.rows[0]
            yn = {
                c.col: c.text.strip() for c in hdr.cells if c.colspan == 1 and c.text.strip().lower() in ("yes", "no")
            }
            if len(yn) == 2 and not any("[" in c.text for row in t.rows[1:] for c in row.cells if c.col in yn):
                empty_yn = [
                    (row, c)
                    for row in t.rows[1:]
                    for c in row.cells
                    if c.col in yn and c.colspan == 1 and not c.text.strip() and any(x.text.strip() for x in row.cells)
                ]
                if len(empty_yn) >= 4:
                    for row, c in empty_yn:
                        c.inner = f"[ ] {yn[c.col]}"
                        c.raw = ""
                        row.edited = True
                        ch = True
        if ch:
            t.changed = True
            return t.render()
        return m.group(0)

    out = _HTML_TABLE.sub(_fix_table, markdown)
    out = _ACRO_MID_BLANK_NOTE.sub(r"\1\n\2", out)
    out = _ACRO_TRAIL_PAREN.sub(r"\1\n\2", out)
    out = _ACRO_TRAIL_INSTR.sub(r"\1\n\2", out)
    out = _ACRO_SSN_MASK.sub(r"\1", out)
    return out


def repair_acroform(
    markdown: str,
    rendered: RenderedPage,
    pdf_bytes: bytes,
    page_index: int = 0,
    *,
    do_id_boxes: bool = True,
    do_clean: bool = False,
) -> str:
    """Apply AcroForm fillable-PDF repairs when the page has >= 3 interactive text/checkbox/radio widgets."""
    if not markdown or not pdf_bytes:
        return markdown
    n_int, cboxes = acroform_widgets(pdf_bytes, page_index)
    if n_int < 3:
        return markdown
    if do_id_boxes:
        edits = acro_id_edits(markdown)
        if edits:
            markdown = apply_edits(markdown, edits)
        if cboxes:
            cb_edits = acro_checkbox_edits(markdown, rendered, cboxes)
            if cb_edits:
                markdown = apply_edits(markdown, cb_edits)
    if do_clean:
        has_boxes = len(cboxes) >= 2
        if not has_boxes:
            from docparse.pdf.boxes import check_boxes

            has_boxes = len(check_boxes(pdf_bytes, page_index)) >= 4
        markdown = acro_clean_markdown(markdown, has_boxes=has_boxes)
    return markdown

"""AcroForm fillable-PDF form repairs (option `acroform`).

Acts ONLY on PDF pages that contain interactive AcroForm text/checkbox/radio widgets (>= 3 widgets):
1. Merges a dedicated leading row-ID table column (`<td>3</td><td>Caption</td>`) into the caption cell
   (`<td colspan="2">3 Caption</td>`), resolving sub-item letters (`<td>a</td>` -> `8a`) and tracking
   col-0/1 rowspans only so right-hand spanning columns never block the merge.
2. Expands inline sub-letter captions when the adjacent right cell holds the full compound ID
   (`<td><strong>b</strong> Taxable interest</td><td><strong>2b</strong></td>` -> `2b Taxable interest`).
3. Inserts `[ ] ` or `[x] ` beside aligned printed words in mark-free units when an AcroForm checkbox
   or radio widget sits immediately to the word's left on the same PDF line; where such a word already
   carries a single `[ ]`/`[x]` mark, corrects the mark to the widget's authoritative checked state.
4. Writes the values of visible, non-password AcroForm text / combo-box / list-box fields that the markdown
   lacks after the field's printed label (same line to the left, or just above), or into a `[___]` blank.
"""

from __future__ import annotations

import bisect
import ctypes
import html
import re
import unicodedata

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
_BOX_MARK = re.compile(r"\[[ \t]*([xX])?[ \t]*\]")  # a plain `[ ]` / `[x]` check-box mark
_MARK_GAP = re.compile(r"(?:[ \t*_]|<(?:strong|b|em|i|u)>)*")  # what may sit between a mark and its word
_MARK_WINDOW = 12  # a word's mark must start within this many characters before the word
_BLANK = re.compile(r"\[_{2,}\]|(?<![\w\\])_{3,}(?!\w)")  # a `[___]` (or bare `___`) blank to fill
_LABEL_TAIL = re.compile(r"(?:[)\]:.*]|</(?:strong|b|em|i|u)>|[ \t]+(?=:))*")  # closing punctuation after a label
_ENDS_COLON = re.compile(r":(?:\*\*|__|</(?:strong|b|em|i|u)>|\s)*$")
_TABLE_SPAN = re.compile(r"<table\b.*?(?:</table>|\Z)", re.S | re.I)
_TEXT_FIELDS = (4, 5, 6)  # FPDF_FORMFIELD_COMBOBOX, _LISTBOX, _TEXTFIELD
_ANNOT_HIDDEN = 2 | 32  # FPDF_ANNOT_FLAG_HIDDEN | FPDF_ANNOT_FLAG_NOVIEW
_FIELD_PASSWORD = 1 << 13  # FPDF_FORMFLAG_TEXT_PASSWORD
_MAX_VALUE_BYTES = 1 << 16
_MAX_VALUE_CHARS = 300

CheckWidget = tuple[float, float, float, float, bool]  # (x0, y0, x1, y1, checked), top-down PDF pts
TextWidget = tuple[float, float, float, float, str]  # (x0, y0, x1, y1, value), top-down PDF pts


def acroform_widgets(pdf_bytes: bytes, page_index: int = 0) -> tuple[int, list[CheckWidget]]:
    """Return (interactive_widget_count, checkbox/radio rects `(x0, y0, x1, y1, checked)` in top-down PDF pts)."""
    n_int, cboxes, _texts = _scan(pdf_bytes, page_index)
    return n_int, cboxes


def acroform_text_widgets(pdf_bytes: bytes, page_index: int = 0) -> list[TextWidget]:
    """Visible, non-password text / combo-box / list-box widgets with a value: `(x0, y0, x1, y1, value)` in
    top-down PDF pts of the displayed (cropped) page; none on rotated pages."""
    return _scan(pdf_bytes, page_index)[2]


def _scan(pdf_bytes: bytes, page_index: int) -> tuple[int, list[CheckWidget], list[TextWidget]]:
    if not pdf_bytes or not pdf_bytes.lstrip()[:5].startswith(b"%PDF-"):
        return 0, [], []  # an image input, or not a PDF at all
    from docparse.pdf.render import _PDFIUM_LOCK

    with _PDFIUM_LOCK:
        try:
            return _widgets(pdfium.PdfDocument(pdf_bytes), page_index)
        except pdfium.PdfiumError:
            return 0, [], []  # unreadable (corrupt or encrypted): the renderer has already said so


def _widgets(doc, page_index: int) -> tuple[int, list[CheckWidget], list[TextWidget]]:
    try:
        if page_index < 0 or page_index >= len(doc):
            return 0, [], []
        page = doc[page_index]
        n_ann = pdfium_c.FPDFPage_GetAnnotCount(page.raw)
        if n_ann <= 0:
            return 0, [], []
        h = page.get_height()
        try:  # text widgets map through the CropBox; a rotated page's display axes differ, so it gets none
            crop_left, _cb, _cr, crop_top = page.get_cropbox()
            want_texts = page.get_rotation() % 360 == 0
        except (pdfium.PdfiumError, AttributeError, TypeError, ValueError):
            crop_left, crop_top, want_texts = 0.0, h, False
        form_info = pdfium_c.FPDF_FORMFILLINFO()
        form_info.version = 1
        fh = pdfium_c.FPDFDOC_InitFormFillEnvironment(doc.raw, ctypes.byref(form_info))
        try:
            n_interactive = 0
            cboxes: list[CheckWidget] = []
            texts: list[TextWidget] = []
            for ai in range(n_ann):
                ann = pdfium_c.FPDFPage_GetAnnot(page.raw, ai)
                try:
                    if pdfium_c.FPDFAnnot_GetSubtype(ann) != pdfium_c.FPDF_ANNOT_WIDGET:
                        continue
                    ft = pdfium_c.FPDFAnnot_GetFormFieldType(fh, ann)
                    if ft in (2, 3, 6):  # checkbox, radio, text field
                        n_interactive += 1
                    if ft in (2, 3):
                        rect = pdfium_c.FS_RECTF()
                        if pdfium_c.FPDFAnnot_GetRect(ann, ctypes.byref(rect)):
                            checked = _is_checked(fh, ann)
                            x0, x1 = min(rect.left, rect.right), max(rect.left, rect.right)
                            y0, y1 = h - max(rect.top, rect.bottom), h - min(rect.top, rect.bottom)
                            if 3.0 <= x1 - x0 <= 25.0 and 3.0 <= y1 - y0 <= 25.0:
                                cboxes.append((x0, y0, x1, y1, checked))
                    elif ft in _TEXT_FIELDS and want_texts:
                        tw = _text_widget(fh, ann, ft, crop_left, crop_top)
                        if tw is not None:
                            texts.append(tw)
                finally:
                    pdfium_c.FPDFPage_CloseAnnot(ann)
            return n_interactive, cboxes, texts
        finally:
            pdfium_c.FPDFDOC_ExitFormFillEnvironment(fh)
    finally:
        doc.close()


def _text_widget(fh, ann, ft: int, left: float, top: float) -> TextWidget | None:
    """A visible, non-password value field's rect and normalized value; None when hidden, empty or unreadable."""
    try:
        if pdfium_c.FPDFAnnot_GetFlags(ann) & _ANNOT_HIDDEN:
            return None
        if ft == 6 and pdfium_c.FPDFAnnot_GetFormFieldFlags(fh, ann) & _FIELD_PASSWORD:
            return None
        value = " ".join(_field_value(fh, ann).split())
        if not value or len(value) > _MAX_VALUE_CHARS or not re.search(r"[^\W_]", value):
            return None  # empty, oversized, or only a placeholder (`___`, `__/__/____`)
        default_val = " ".join(_utf16(lambda buf, n: pdfium_c.FPDFAnnot_GetStringValue(ann, b"DV", buf, n)).split())
        if default_val and value == default_val:
            return None  # untouched template default (/V == /DV)
        rect = pdfium_c.FS_RECTF()
        if not pdfium_c.FPDFAnnot_GetRect(ann, ctypes.byref(rect)):
            return None
    except (AttributeError, TypeError, ValueError, OSError, ctypes.ArgumentError):
        return None  # a PDFium build without these calls, or a malformed annotation
    x0, x1 = min(rect.left, rect.right) - left, max(rect.left, rect.right) - left
    y0, y1 = top - max(rect.top, rect.bottom), top - min(rect.top, rect.bottom)
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return x0, y0, x1, y1, value


def _is_checked(fh, ann) -> bool:
    """A checkbox/radio widget's displayed state: its `/AS` appearance state is on unless it is `Off` (PDF
    32000-1 12.7.4.2.3). `FPDFAnnot_IsChecked` needs a PDFium page view (an `FFI_GetPage` callback) that this
    bare form environment lacks, so it is only the fallback for widgets without `/AS`."""
    state = _utf16(lambda buf, n: pdfium_c.FPDFAnnot_GetStringValue(ann, b"AS", buf, n))
    if state:
        return state != "Off"
    try:
        return bool(pdfium_c.FPDFAnnot_IsChecked(fh, ann))
    except (AttributeError, TypeError, ValueError, OSError, ctypes.ArgumentError):
        return False


def _field_value(fh, ann) -> str:
    """A form field's value ('' when absent or unreadable)."""
    return _utf16(lambda buf, n: pdfium_c.FPDFAnnot_GetFormFieldValue(fh, ann, buf, n))


def _utf16(call) -> str:
    """Read a PDFium UTF-16LE string out-parameter: `call(None, 0)` gives the byte length (with the trailing
    NUL), `call(buffer, length)` fills it. '' when absent, oversized or the call fails."""
    try:
        need = int(call(None, 0))
        if need <= 2 or need > _MAX_VALUE_BYTES:
            return ""
        buf = ctypes.create_string_buffer(need)
        got = int(call(ctypes.cast(buf, ctypes.POINTER(pdfium_c.FPDF_WCHAR)), need))
        if got <= 2 or got > need:
            return ""
        return buf.raw[: got - got % 2].decode("utf-16-le", errors="ignore").rstrip("\x00")
    except (AttributeError, TypeError, ValueError, OSError, ctypes.ArgumentError):
        return ""


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


def _pt_words(rendered: RenderedPage) -> list[PtWord]:
    s = rendered.scale
    return [PtWord(w.text, w.box[0] * s, w.box[1] * s, w.box[2] * s, w.box[3] * s) for w in rendered.text_words or []]


def _box_left_of(w0: PtWord, cboxes: list[CheckWidget], words: list[PtWord], skip: set[int]) -> int | None:
    """The first unused checkbox/radio widget immediately left of printed word w0 on its line, with no printed
    word between them."""
    h = max(6.0, w0.y1 - w0.y0)
    for bi, (_bx0, by0, bx1, by1, _chk) in enumerate(cboxes):
        if bi in skip:
            continue
        bcy = (by0 + by1) / 2
        if w0.y0 - 0.6 * h <= bcy <= w0.y1 + 0.6 * h and bx1 <= w0.x0 + 2.0 and 0 <= w0.x0 - bx1 <= 2.5 * h:
            if not any(bx1 < p.cx < w0.x0 and abs(p.cy - w0.cy) < 0.5 * h for p in words):
                return bi
    return None


def acro_checkbox_edits(markdown: str, rendered: RenderedPage, cboxes: list[CheckWidget]) -> list[tuple[int, int, str]]:
    """Insert `[ ] ` or `[x] ` beside aligned printed words in mark-free units when an AcroForm checkbox/radio
    widget sits immediately to the word's left on the same PDF line. In units that already hold marks, a single
    `[ ]`/`[x]` mark right before such a word is set to the widget's checked state."""
    if not cboxes or not rendered.text_words:
        return []
    words = _pt_words(rendered)
    al = align(markdown, words, reorder=True)
    edits: list[tuple[int, int, str]] = []
    bounds = [0] + [m.end() for m in _UNIT_END.finditer(markdown)] + [len(markdown)]
    t_ptr = 0
    corrected: set[int] = set()  # widgets whose mark was already checked (one mark per widget per page)
    for u0, u1 in zip(bounds, bounds[1:], strict=False):
        while t_ptr < len(al.tokens) and al.tokens[t_ptr].start < u0:
            t_ptr += 1
        mine: list[int] = []
        k = t_ptr
        while k < len(al.tokens) and al.tokens[k].end <= u1:
            if al.tokens[k].start >= u0:
                mine.append(k)
            k += 1
        if not mine:
            continue
        if _HAS_BOX.search(markdown[u0:u1]):
            edits += _mark_corrections(markdown, u0, mine, al, words, cboxes, corrected)
            continue
        used_boxes: set[int] = set()
        for t in mine:
            tws = al.token_words(t)
            if not tws:
                continue
            bi = _box_left_of(al.words[tws[0]], cboxes, words, used_boxes)
            if bi is not None:
                mark = "[x] " if cboxes[bi][4] else "[ ] "
                edits.append((al.tokens[t].start, al.tokens[t].start, mark))
                used_boxes.add(bi)
    return edits


def _mark_corrections(
    markdown: str, u0: int, mine: list[int], al, words: list[PtWord], cboxes: list[CheckWidget], corrected: set[int]
) -> list[tuple[int, int, str]]:
    """Set the single `[ ]`/`[x]` mark right before each aligned word of a unit to the checked state of the
    widget immediately left of that word in the PDF."""
    edits: list[tuple[int, int, str]] = []
    for t in mine:
        tws = al.token_words(t)
        if not tws:
            continue
        bi = _box_left_of(al.words[tws[0]], cboxes, words, corrected)
        if bi is None:
            continue
        start = al.tokens[t].start
        marks = list(_BOX_MARK.finditer(markdown, max(u0, start - _MARK_WINDOW), start))
        if len(marks) != 1 or not _MARK_GAP.fullmatch(markdown, marks[0].end(), start):
            continue
        corrected.add(bi)
        checked = cboxes[bi][4]
        if bool(marks[0].group(1)) != checked:
            edits.append((marks[0].start(), marks[0].end(), "[x]" if checked else "[ ]"))
    return edits


_LABEL_GAP = 216.0  # pts: the farthest a same-line label may sit left of its field
_LABEL_ABOVE = 20.0  # pts: the farthest a label line may sit above its field
_ZERO_DEFAULT = re.compile(r"^[\s$€£¥,+()-]*0+(?:\.0+)?%?$")


def acro_text_value_edits(markdown: str, rendered: RenderedPage, texts: list[TextWidget]) -> list[tuple[int, int, str]]:
    """Write AcroForm field values the markdown lacks next to their printed labels.

    A value counts as present when its case/space-folded text (letters and digits only, for values of 4+ of
    them) occurs anywhere in the markdown; present values are never repeated. A missing value goes into the
    first free `[___]` blank after its label in the label's line/cell, else right after the label (`Label: value`).
    The label is the nearest printed word left of the field on its line, or the closest line just above it
    (within `_LABEL_ABOVE` pts, over the field's width); fields whose label did not align are left alone."""
    if not texts or not markdown or not rendered.text_words:
        return []
    if all(_ZERO_DEFAULT.fullmatch(tw[4]) for tw in texts):
        return []  # blank auto-calculated worksheet where every /V is a zero subtotal default
    md_spaced, _sq, md_alnum = _fold_text(markdown)
    missing = [tw for tw in texts if not _present(tw[4], md_spaced, md_alnum)]
    if not missing:
        return []
    words = _pt_words(rendered)
    al = align(markdown, words, reorder=True)
    word_tok = al.word_tokens()
    units = _units(markdown)
    starts = [u[0] for u in units]
    tables = [(m.start(), m.end()) for m in _TABLE_SPAN.finditer(markdown)]
    edits: list[tuple[int, int, str]] = []
    claimed: set[int] = set()
    inserts: dict[int, tuple[str, list[tuple[float, str]]]] = {}  # position -> (separator, [(x0, value)])
    for tw in sorted(missing, key=lambda w: (w[1], w[0])):
        found = _label_word(tw, texts, words, word_tok)
        if found is None:
            continue
        pi, above = found
        tok = al.tokens[word_tok[pi][-1]]
        ui = bisect.bisect_right(starts, tok.start) - 1
        us, ue, _term = units[ui]
        if tok.end > ue:
            continue
        blank = _free_blank(markdown, units, ui, tok.end, above, claimed)
        if blank is not None:
            claimed.add(blank[0])
            edits.append((blank[0], blank[1], _escape_value(tw[4], blank[0], markdown, tables)))
            continue
        if above:  # after the end of the label's line/cell
            pos = us + len(markdown[us:ue].rstrip())
            sep = " " if _ENDS_COLON.search(markdown[us:pos]) else ": "
        else:  # right after the label phrase (and its closing punctuation)
            tail = _LABEL_TAIL.match(markdown, tok.end, ue)
            pos = tail.end() if tail else tok.end
            sep = " " if tail and ":" in tail.group(0) else ": "
        inserts.setdefault(pos, (sep, []))[1].append((tw[0], tw[4]))
    for pos, (sep, vals) in inserts.items():
        text = " ".join(v for _x, v in sorted(vals))
        edits.append((pos, pos, sep + _escape_value(text, pos, markdown, tables)))
    return edits


def _fold_text(s: str) -> tuple[str, str, str]:
    """`s` without tags, entities, markdown escapes/emphasis; lowercased NFKC as (space-collapsed, squeezed,
    letters-and-digits only)."""
    s = html.unescape(re.sub(r"<[^<>\n]*>", " ", s))
    s = re.sub(r"[\\*_`]", "", unicodedata.normalize("NFKC", s).lower())
    spaced = " ".join(s.split())
    return spaced, spaced.replace(" ", ""), "".join(c for c in spaced if c.isalnum())


def _present(value: str, md_spaced: str, md_alnum: str) -> bool:
    v_spaced, _v_sq, v_alnum = _fold_text(value)
    if not v_alnum:
        return True  # nothing worth writing
    if len(v_alnum) >= 4:
        return v_alnum in md_alnum
    return re.search(rf"(?<![^\W_]){re.escape(v_spaced)}(?![^\W_])", md_spaced) is not None


def _units(markdown: str) -> list[tuple[int, int, str]]:
    """(start, content end, terminator) of each line / cell / `<br>`-separated unit."""
    out: list[tuple[int, int, str]] = []
    pos = 0
    for m in _UNIT_END.finditer(markdown):
        out.append((pos, m.start(), m.group(0)))
        pos = m.end()
    out.append((pos, len(markdown), ""))
    return out


def _label_word(
    tw: TextWidget, texts: list[TextWidget], words: list[PtWord], word_tok: list[list[int]]
) -> tuple[int, bool] | None:
    """(index of the field's aligned printed label word, label is above the field) or None."""
    x0, y0, x1, y1, _v = tw
    fh = y1 - y0
    near = None
    for pi, p in enumerate(words):
        if not any(c.isalnum() for c in p.text):
            continue
        overlap = min(p.y1, y1) - max(p.y0, y0)
        if overlap >= 0.5 * min(p.y1 - p.y0, fh) and p.x1 <= x0 + 2.0 and x0 - p.x1 <= _LABEL_GAP:
            if near is None or p.x1 > words[near].x1:
                near = pi
    if near is not None:
        return (near, False) if word_tok[near] else None  # an unaligned same-line label: no safe anchor
    above = [
        pi
        for pi, p in enumerate(words)
        if any(c.isalnum() for c in p.text) and p.y1 <= y0 + 2.0 and y0 - p.y1 <= _LABEL_ABOVE and x0 - 4.0 <= p.x0 < x1
    ]
    if not above:
        return None
    top = max(above, key=lambda pi: words[pi].y1)
    ref = words[top]
    lh = max(6.0, ref.y1 - ref.y0)
    if any(o is not tw and o[0] < x1 and o[2] > x0 and o[1] >= ref.y1 - 1.0 and o[3] <= y0 + 1.0 for o in texts):
        return None  # another field sits between that label line and this field
    line = [pi for pi in above if abs(words[pi].cy - ref.cy) < 0.5 * lh and word_tok[pi]]
    if not line:
        return None
    return max(line, key=lambda pi: words[pi].x1), True


def _free_blank(
    markdown: str, units: list[tuple[int, int, str]], ui: int, after: int, above: bool, claimed: set[int]
) -> tuple[int, int] | None:
    """The first unclaimed `[___]` blank after `after` in unit ui; for a label above its field, also a unit that
    holds only a blank on the label's next line(s) of the same cell."""
    for m in _BLANK.finditer(markdown, after, units[ui][1]):
        if m.start() not in claimed:
            return m.start(), m.end()
    if not above:
        return None
    for k in range(ui, min(ui + 3, len(units) - 1)):
        if not (units[k][2] == "\n" or units[k][2].lower().startswith("<br")):
            return None
        ns, ne, _t = units[k + 1]
        content = markdown[ns:ne].strip()
        if not content:
            continue
        m = _BLANK.fullmatch(content)
        if m is None:
            return None
        bs = ns + markdown[ns:ne].index(content)
        return None if bs in claimed else (bs, bs + len(content))
    return None


def _escape_value(value: str, pos: int, markdown: str, tables: list[tuple[int, int]]) -> str:
    if any(a <= pos < b for a, b in tables):
        return html.escape(value, quote=False)
    v = value.replace("<", "&lt;")
    if markdown[markdown.rfind("\n", 0, pos) + 1 : pos].lstrip().startswith("|"):
        v = v.replace("|", "\\|")  # inside a pipe-table row
    return v


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
    n_int, cboxes, texts = _scan(pdf_bytes, page_index)
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
        if texts:
            tv_edits = acro_text_value_edits(markdown, rendered, texts)
            if tv_edits:
                markdown = apply_edits(markdown, tv_edits)
    if do_clean:
        has_boxes = len(cboxes) >= 2
        if not has_boxes:
            from docparse.pdf.boxes import check_boxes

            has_boxes = len(check_boxes(pdf_bytes, page_index)) >= 4
        markdown = acro_clean_markdown(markdown, has_boxes=has_boxes)
    return markdown

"""PDF-grounded self-verification for primary VLM page reads.

Detects two objective failure modes on born-digital PDFs with a healthy, visible text layer:
  1. Severe PDF word-recall drop (the reader silently omitted a substantial portion of visible PDF text).
  2. Flattened table (`<table>` absent in the reading even though the PDF text layer forms a multi-column grid).

Both checks depend strictly on the PDF's visible text layer and word coordinates — never on dataset wording or
document types — and only accept a secondary reader's output when it objectively improves PDF alignment.
"""

from __future__ import annotations

import re
from collections import Counter

from docparse.fixes.align_text import TOKEN, key
from docparse.fixes.layout_refine import PtWord, _segments, lines_of, words_in_points
from docparse.fixes.tables_align import align_table, page_words
from docparse.fixes.tables_align import lines_of as table_lines_of
from docparse.fixes.tables_html import parse_table
from docparse.pdf.render import RenderedPage

MAX_HIDDEN = 0.5
MAX_MAP_ERRORS = 0.01
MIN_WORDS = 50
TABLE_MIN_WORDS = 15
TABLE_MAX_MAP_ERRORS = 0.03
TABLE_GEO_MIN_CELLS = 8
TABLE_GEO_MARGIN = 0.02
_TAG_RE = re.compile(r"<[^>]+>")
_TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)
_CHECKBOX_RE = re.compile(r"\[[ x]\]|<input\b", re.I)


def has_healthy_text_layer(
    rendered: RenderedPage, *, min_words: int = MIN_WORDS, max_map_errors: float = MAX_MAP_ERRORS
) -> bool:
    """True when the page has a visible, cleanly unicode-mapped PDF text layer of at least `min_words` words."""
    if rendered.hidden is None or rendered.hidden > MAX_HIDDEN or (rendered.map_errors or 0) > max_map_errors:
        return False
    return len(words_in_points(rendered)) >= min_words


def pdf_word_recall(markdown: str, words: list[PtWord]) -> float:
    """Fraction of visible PDF word tokens present in `markdown` (multiset recall)."""
    layer = Counter(key(m.group()) for w in words for m in TOKEN.finditer(w.text))
    total = sum(layer.values())
    if total < 30:
        return 1.0
    have = Counter(key(m.group()) for m in TOKEN.finditer(_TAG_RE.sub(" ", markdown)))
    return sum(min(n, have[k]) for k, n in layer.items()) / total


def pdf_looks_tabular(words: list[PtWord], min_rows: int = 4, min_cols: int = 3) -> bool:
    """True when PDF word coordinates form at least `min_rows` lines with `min_cols` separated column segments."""
    if len(words) < min_rows * min_cols:
        return False
    multi_col_lines = 0
    for line in lines_of(words):
        segs = _segments(line, 1.5)
        if len(segs) >= min_cols:
            multi_col_lines += 1
            if multi_col_lines >= min_rows:
                return True
    return False


def table_alignment_stats(markdown: str, rendered: RenderedPage) -> tuple[int, int, int, float]:
    """Measure `(n_tables, matched_cells, total_cells, weighted_found)` of HTML tables against the PDF text layer."""
    twords = page_words(rendered)
    if not twords:
        return 0, 0, 0, 1.0
    tlines = table_lines_of(twords)
    used: set[int] = set()
    n_tables = matched = total = 0
    for raw_tbl in _TABLE_RE.findall(markdown):
        ht = parse_table(raw_tbl)
        if ht is None or len(ht.rows) < 2 or ht.width() < 2:
            continue
        n_tables += 1
        al = align_table(ht, tlines, used)
        for ws in al.words.values():
            used.update(w.index for w in ws)
        matched += len(al.words)
        total += al.cells
    return n_tables, matched, total, (matched / total if total else 1.0)


def table_geo_metrics(markdown: str, rendered: RenderedPage) -> dict[str, float | int]:
    """Measure objective PDF-coordinate geometric fidelity of HTML tables on a born-digital page."""
    twords = page_words(rendered)
    if not twords:
        return {
            "n_tbl": 0,
            "matched": 0,
            "total": 0,
            "found": 0.0,
            "col_viol": 0,
            "empty_cols": 0,
            "word_cov": 0.0,
        }
    tlines = table_lines_of(twords)
    used: set[int] = set()
    n_tbl = matched = total = col_viol = empty_cols = 0
    matched_words: set[int] = set()
    for raw_tbl in _TABLE_RE.findall(markdown):
        ht = parse_table(raw_tbl)
        if ht is None or len(ht.rows) < 2 or ht.width() < 2:
            continue
        n_tbl += 1
        al = align_table(ht, tlines, used)
        for ws in al.words.values():
            for w in ws:
                used.add(w.index)
                matched_words.add(w.index)
        matched += len(al.words)
        total += al.cells
        w_tbl = ht.width()
        col_has = [False] * w_tbl
        for row in ht.rows:
            for c in row.cells:
                if c.text:
                    for k in range(c.col, min(w_tbl, c.col + c.colspan)):
                        col_has[k] = True
        empty_cols += sum(1 for has in col_has if not has)
        bands: dict[int, list[tuple[float, float]]] = {}
        for row in ht.rows:
            for c in row.cells:
                sp = al.span(c)
                if sp and c.colspan == 1:
                    bands.setdefault(c.col, []).append(sp)
        cols = sorted(bands)
        for c1, c2 in zip(cols, cols[1:], strict=False):
            m1 = sorted((a + b) / 2 for a, b in bands[c1])[len(bands[c1]) // 2]
            m2 = sorted((a + b) / 2 for a, b in bands[c2])[len(bands[c2]) // 2]
            if m1 >= m2:
                col_viol += 1
        for spans in bands.values():
            if len(spans) < 3:
                continue
            x0_med = sorted(a for a, _ in spans)[len(spans) // 2]
            x1_med = sorted(b for _, b in spans)[len(spans) // 2]
            pad = max(12.0, 0.5 * (x1_med - x0_med))
            for a, b in spans:
                cx = (a + b) / 2
                if cx < x0_med - pad or cx > x1_med + pad:
                    col_viol += 1
    return {
        "n_tbl": n_tbl,
        "matched": matched,
        "total": total,
        "found": matched / total if total else 0.0,
        "col_viol": col_viol,
        "empty_cols": empty_cols,
        "word_cov": len(matched_words) / max(1, len(twords)),
    }


def table_geo_score(g: dict[str, float | int]) -> float:
    """Composite PDF-geometry table score: rewards word coverage and cell alignment, penalizes empty/misaligned cols."""
    return float(g["word_cov"]) + 0.5 * float(g["found"]) - 0.04 * int(g["empty_cols"]) - 0.01 * int(g["col_viol"])


def needs_pdf_verification(
    markdown: str,
    rendered: RenderedPage,
    min_recall: float = 0.68,
    min_table_found: float = 0.80,
    *,
    table_geo: bool = False,
) -> tuple[bool, str]:
    """Return `(should_verify, reason)` when `markdown` looks incomplete, flattened, or poorly table-aligned."""
    from docparse.fixes.fallback import has_math

    is_math = has_math(markdown)
    if not table_geo:
        if not has_healthy_text_layer(rendered) or is_math:
            return False, ""
        words = words_in_points(rendered)
        recall = pdf_word_recall(markdown, words)
        if recall < min_recall:
            return True, f"low_recall:{recall:.3f}"
        if "<table" not in markdown.lower():
            if pdf_looks_tabular(words):
                return True, f"missing_table:{recall:.3f}"
            return False, ""
        if len(_CHECKBOX_RE.findall(markdown)) >= 4:
            return False, ""
        n_tbl, matched, total, found = table_alignment_stats(markdown, rendered)
        if n_tbl >= 1 and total >= 10 and found < min_table_found:
            return True, f"weak_table:{found:.3f}:{matched}"
        return False, ""

    if not has_healthy_text_layer(rendered, min_words=TABLE_MIN_WORDS, max_map_errors=TABLE_MAX_MAP_ERRORS):
        return False, ""
    words = words_in_points(rendered)
    recall = pdf_word_recall(markdown, words)
    if not is_math and len(words) >= MIN_WORDS and recall < min_recall:
        return True, f"low_recall:{recall:.3f}"
    if len(_CHECKBOX_RE.findall(markdown)) >= 4:
        return False, ""
    if "<table" not in markdown.lower():
        if not is_math and (
            pdf_looks_tabular(words)
            or (len(words) < 60 and pdf_looks_tabular(words, min_rows=3, min_cols=2))
        ):
            return True, f"missing_table:{recall:.3f}"
        return False, ""
    n_tbl, matched, total, found = table_alignment_stats(markdown, rendered)
    if not is_math and n_tbl >= 1 and total >= 10 and found < min_table_found:
        return True, f"weak_table:{found:.3f}:{matched}"
    if not is_math and n_tbl >= 1 and total >= TABLE_GEO_MIN_CELLS:
        return True, f"table_geo:{found:.3f}:{matched}"
    return False, ""


def secondary_improves(
    primary_md: str,
    secondary_md: str,
    rendered: RenderedPage,
    reason: str,
    recall_margin: float = 0.12,
    table_health: float = 0.65,
    table_found_margin: float = 0.12,
    *,
    table_geo: bool = False,
    geo_margin: float = TABLE_GEO_MARGIN,
) -> bool:
    """True only when `secondary_md` objectively fixes the flagged PDF verification defect."""
    if not secondary_md.strip():
        return False
    words = words_in_points(rendered)
    p_rec = pdf_word_recall(primary_md, words)
    s_rec = pdf_word_recall(secondary_md, words)
    if reason.startswith("low_recall:"):
        return s_rec >= p_rec + recall_margin
    if reason.startswith("missing_table:"):
        if s_rec < p_rec - (0.03 if table_geo else 0.04):
            return False
        tbls = _TABLE_RE.findall(secondary_md)
        if not tbls:
            return False
        twords = page_words(rendered)
        if not twords:
            return False
        tlines = table_lines_of(twords)
        for raw_tbl in tbls:
            ht = parse_table(raw_tbl)
            if ht is None or len(ht.rows) < 3 or ht.width() < (2 if table_geo else 3):
                continue
            al = align_table(ht, tlines, set())
            if ht.width() >= 3 and al.cells and al.found >= table_health:
                return True
            if table_geo and ht.width() == 2 and len(al.words) >= 6 and al.found >= 0.85:
                return True
        return False
    if reason.startswith("weak_table:"):
        if s_rec < p_rec - 0.02:
            return False
        _, p_matched, p_total, p_found = table_alignment_stats(primary_md, rendered)
        s_nt, s_matched, s_total, s_found = table_alignment_stats(secondary_md, rendered)
        margin = 0.10 if table_geo else table_found_margin
        return (
            s_nt >= 1
            and p_total >= 10
            and s_total >= max(10, int(0.70 * p_total))
            and s_matched >= p_matched
            and s_found >= p_found + margin
        )
    if reason.startswith("table_geo:"):
        if s_rec < p_rec - 0.01:
            return False
        pg = table_geo_metrics(primary_md, rendered)
        sg = table_geo_metrics(secondary_md, rendered)
        return (
            int(sg["n_tbl"]) >= 1
            and max(int(pg["total"]), int(sg["total"])) >= TABLE_GEO_MIN_CELLS
            and float(sg["found"]) >= table_health
            and table_geo_score(sg) > table_geo_score(pg) + geo_margin
        )
    return False

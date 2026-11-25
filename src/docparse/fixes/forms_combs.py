"""Extended comb-noise and mini value-table repairs on filled forms (options `comb_noise`, `mini_tables`).

On filled forms with character-box fields, a page read may write:
- comb digits wrapped in brackets (`[1][5][6]...`) or empty comb-divider noise (`1__1__1__1__`);
- a 1- or 2-row HTML table of character boxes or a single box-code + value right under its caption line
  (`Net profit` / `<table><tr><th>3.83</th><th>£</th><th>78,595</th></tr></table>`).

Each repair acts only when the PDF's typed-in overlay confirms which tokens are typed values and which are printed.
"""

from __future__ import annotations

import re

from docparse.fixes.align import fold
from docparse.fixes.forms import FormPage
from docparse.fixes.tables_html import parse_table

_NOISE_EMPTY_COMB = re.compile(r"(?:1_{1,4}){3,}1?_?")
_BRACK_GAP = re.compile(r"(?:[ \t*_]|</?(?:u|strong|b|em|i)>)*\][ \t]*\[(?:[ \t*_]|</?(?:u|strong|b|em|i)>)*")
_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_HINT_SINGLE = re.compile(r"^[DMYJAdmyja]{1,2}$")
_BOX_CODE = re.compile(r"^(?:[0-9]{1,5}[a-z]?|[0-9]{1,2}\.[0-9]{1,3})$", re.I)
_SYM_ONLY = re.compile(r"^[£$€¥%—–\-/.,·\s]*$")


def _token_overlay_flags(page: FormPage) -> list[bool]:
    al = page.al
    ov = [False] * len(al.tokens)
    for k, p in enumerate(al.amatch):
        if p is not None and page.owner[al.p_word[p]] >= 0:
            ov[al.a_tok[k]] = True
    for _, (t0, t1) in page.placed.items():
        for t in range(t0, t1 + 1):
            ov[t] = True
    return ov


def comb_noise_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """Join bracketed comb values (`[1][2][3]` -> `123`) and replace empty comb-divider noise (`1__1__1__`)
    with `[___]`."""
    edits: list[tuple[int, int, str]] = []
    md, al = page.markdown, page.al
    tok_ov = _token_overlay_flags(page)

    for f, (t0, t1) in page.placed.items():
        fill = page.fills[f]
        if not fill.comb or t1 == t0:
            continue
        if "".join(fold(ch) for t in range(t0, t1 + 1) for ch in al.tokens[t].text) != fill.key:
            continue
        gaps = [md[al.tokens[t].end : al.tokens[t + 1].start] for t in range(t0, t1)]
        if all(_BRACK_GAP.fullmatch(g) for g in gaps):
            s0, e1 = al.tokens[t0].start, al.tokens[t1].end
            m_l = re.search(r"\[[ \t]*$", md[:s0])
            m_r = re.match(r"[ \t]*\]", md[e1:])
            if m_l and m_r:
                s0 = m_l.start()
                e1 += m_r.end()
            elif s0 > 0 and md[s0 - 1] == "[" and e1 < len(md) and md[e1] == "]":
                s0 -= 1
                e1 += 1
            edits.append((s0, e1, fill.text))

    for m in _NOISE_EMPTY_COMB.finditer(md):
        s, e = m.start(), m.end()
        if not any(tok_ov[i] for i, t in enumerate(al.tokens) if t.start < e and t.end > s):
            edits.append((s, e, "[___]"))

    return edits


def mini_table_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """Flatten a 1- or 2-row HTML table right after a caption line when its cells hold only an overlay value
    (and optional leading box code, currency/separators, empty padding, or single-letter D/M/Y format hints)."""
    md, al = page.markdown, page.al
    tok_ov = _token_overlay_flags(page)
    edits: list[tuple[int, int, str]] = []

    for m in _HTML_TABLE.finditer(md):
        s, e = m.start(), m.end()
        t = parse_table(m.group(0))
        if t is None or len(t.rows) > 2:
            continue
        before = md[:s].rstrip(" \t\r\n")
        if not before or before.endswith("</table>") or md[len(before) : s].count("\n") > 2:
            continue
        prev_nl = before.rfind("\n")
        prev_line = before[prev_nl + 1 :].strip()
        if not prev_line or prev_line.startswith(("#", "|")):
            continue
        if len(t.rows) == 2:
            r1_texts = [c.text.strip() for c in t.rows[1].cells if c.text.strip()]
            if r1_texts and not all(_HINT_SINGLE.fullmatch(x) for x in r1_texts):
                continue
        r0_end_m = re.search(r"</tr\s*>", m.group(0), re.I)
        if not r0_end_m:
            continue
        r0_abs_end = s + r0_end_m.end()
        r0_toks = [i for i, tok in enumerate(al.tokens) if s <= tok.start and tok.end <= r0_abs_end]

        codes, vals, has_ov, ok = [], [], False, True
        raw_t = m.group(0)
        pos = 0
        for c in t.rows[0].cells:
            txt = c.text.strip()
            c_idx = raw_t.find(c.raw, pos) if c.raw else -1
            if c_idx >= 0:
                pos = c_idx + len(c.raw)
            if not txt:
                continue
            if _SYM_ONLY.fullmatch(txt):
                if txt in ("£", "$", "€", "¥", "—", "-", "/"):
                    vals.append(txt)
                continue
            if c_idx < 0:
                ok = False
                break
            cs, ce = s + c_idx, s + c_idx + len(c.raw)
            c_toks = [i for i in r0_toks if cs <= al.tokens[i].start and al.tokens[i].end <= ce]
            if c_toks and all(tok_ov[i] for i in c_toks):
                has_ov = True
                vals.append(txt)
            elif not codes and not vals and _BOX_CODE.fullmatch(txt) and not any(tok_ov[i] for i in c_toks):
                codes.append(txt)
            else:
                ok = False
                break
        if not ok or (not has_ov and not codes):
            continue
        val_str = " ".join(vals) if has_ov else "[___]"
        line_start, line_end = prev_nl + 1, len(before)
        new_prev = md[line_start:line_end]
        if codes and not new_prev.lstrip("*_ ").startswith(codes[0]):
            new_prev = f"{codes[0]} {new_prev}"
        sep = " " if new_prev.rstrip("*_ ").endswith(":") else ": "
        edits.append((line_start, e, f"{new_prev}{sep}{val_str}"))

    return edits

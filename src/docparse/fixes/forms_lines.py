"""Plain-line field slot cleanup on filled forms (option `field_lines`).

On filled forms with plain-text `Caption: value` lines outside HTML tables:
- Moves a printed box code sitting between `:` and the typed value or blank marker into the caption
  (`einfache Entfernung: 111 25 km` -> `111 einfache Entfernung: 25`;
   `Arbeitslohn: 127 [___] EUR` -> `127 Arbeitslohn: [___]`).
- Strips parenthesised format examples containing digits or colons right before `: [___]`
  (`Telephone Number (999) 999-9999: [___]` -> `Telephone Number: [___]`;
   `Namenszusatz (Beispiel: Freifrau, Graf): [___]` -> `Namenszusatz: [___]`).
- Strips short printed hints/units after a blank slot (`: [___] (2TS)` -> `: [___]`, `: AUD` -> `: [___]`)
  when the PDF overlay confirms no value was typed on that line.
- Splits hard trailing printed furniture (`▶`, `►`, bold header/watermark `**...**`) onto a new line after
  an overlay value.
"""

from __future__ import annotations

import re

from docparse.fixes.align import fold
from docparse.fixes.forms import FormPage

_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S | re.I)
_MARK_RE = re.compile(r"\[[ \t]*(?:[xX✓✔✗✘☒☑*]|checked|unchecked|selected)?[ \t]*\]|[☐☑☒□■○◯●◉]", re.I)
_CODE_TOK = re.compile(r"^(?:[0-9]{1,5}[A-Za-z]?|[0-9]{1,2}\.[0-9]{1,3}|[0-9]{1,2}[A-Za-z]|[A-Za-z][0-9]{1,3})$")
_UNIT_TOK = re.compile(
    r"^(?:EUR|USD|GBP|CAD|AUD|CHF|JPY|CHF/EUR|km|kg|cm|mm|m²|m2|ha|%|£|\$|€|¥|years?|months?|days?|hours?|hrs?)$",
    re.I,
)
_BLANK_MARK = re.compile(r"\[[ \t]*_{2,}[ \t]*\]|_{3,}")
_BLANK_COMBO = re.compile(
    r"(?:\[[ \t]*_{2,}[ \t]*\]|_{3,})(?:[ \t]*[,.\uFF0C\u3002][ \t]*(?:\[[ \t]*_{2,}[ \t]*\]|_{3,}))+"
    r"(?:[ \t]*[（(][^()\n]{1,6}[）)])?"
)
_LEAD_BLANK_BEFORE_OV = re.compile(r"((?:\[[ \t]*_{2,}[ \t]*\]|_{3,})[ \t]*[,.\uFF0C][ \t]*)+")
_EXAMPLE_BEFORE_BLANK = re.compile(
    r"([ \t]*\([^()\n]*[:0-9][^()\n]*\)(?:[ \t]*[0-9\-]+)?)[ \t*_]*:[ \t]*(\[[ \t]*_{2,}[ \t]*\]|_{3,})[ \t]*$"
)
_AFTER_BLANK_PRINTED = re.compile(r"(:[ \t*_]*(?:\[[ \t]*_{2,}[ \t]*\]|_{3,}))([ \t]+.+)$")


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


def field_line_edits(page: FormPage) -> list[tuple[int, int, str]]:
    """Repair plain-line field slots outside HTML tables using overlay/printed token alignment."""
    md, al = page.markdown, page.al
    tok_ov, tok_pr = _token_flags(page)
    printed_words = {fold(w.text) for w in page.printed}
    overlay_words = {fold(w.text) for f in page.fills for w in f.words}
    tranges = [(m.start(), m.end()) for m in _HTML_TABLE.finditer(md)]
    edits: list[tuple[int, int, str]] = []

    pos = 0
    tok_idx = 0
    n_toks = len(al.tokens)
    for raw_line in md.splitlines(keepends=True):
        line_s = pos
        line_e = pos + len(raw_line.rstrip("\r\n"))
        pos += len(raw_line)
        line = md[line_s:line_e]
        if not line.strip() or any(a <= line_s < b for a, b in tranges) or line.lstrip().startswith("#") or "|" in line:
            continue
        while tok_idx < n_toks and al.tokens[tok_idx].end <= line_s:
            tok_idx += 1
        ltoks: list[int] = []
        k = tok_idx
        while k < n_toks and al.tokens[k].start < line_e:
            if al.tokens[k].start >= line_s and al.tokens[k].end <= line_e:
                ltoks.append(k)
            k += 1
        if not ltoks:
            continue

        # 0. Leading `[___], [___], ` directly before an overlay value, and comma/dot-separated
        #    blank placeholder runs (`[___], [___]. [___]（元）` -> `[___]`)
        combo_hit = False
        for m in _LEAD_BLANK_BEFORE_OV.finditer(line):
            me = line_s + m.end()
            nxt = next((t for t in ltoks if al.tokens[t].start >= me), None)
            if nxt is not None and tok_ov[nxt] and al.tokens[nxt].start - me <= 3:
                edits.append((line_s + m.start(), me, ""))
                combo_hit = True
        if combo_hit:
            continue
        for m in _BLANK_COMBO.finditer(line):
            ms, me = line_s + m.start(), line_s + m.end()
            if not any(tok_ov[t] for t in ltoks if al.tokens[t].start >= ms and al.tokens[t].end <= me):
                edits.append((ms, me, "[___]"))
                combo_hit = True
        if combo_hit:
            continue

        if _MARK_RE.search(line):
            continue

        # 1. Parenthesised example with colon/digits right before `: [___]` on a blank line
        if not any(tok_ov[t] for t in ltoks):
            me = _EXAMPLE_BEFORE_BLANK.search(line)
            if me and re.search(r"[A-Za-zÀ-ÿ]", line[: me.start()]):
                edits.append((line_s + me.start(), line_e, f": {me.group(2)}"))
                continue

        # Locate top-level label colons (outside parentheses and HH:MM times)
        colons = {
            m.start()
            for m in re.finditer(r":(?=[ \t*_]|$)", line)
            if not (
                m.start() > 0
                and line[m.start() - 1].isdigit()
                and m.start() + 1 < len(line)
                and line[m.start() + 1].isdigit()
            )
        }
        top_colons: list[int] = []
        depth = 0
        for ci, ch in enumerate(line):
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            elif ch == ":" and depth == 0 and ci in colons:
                top_colons.append(ci)

        if len(top_colons) != 1:
            continue

        cpos = top_colons[0]
        abs_c = line_s + cpos
        before_toks = [t for t in ltoks if al.tokens[t].end <= abs_c]
        after_toks = [t for t in ltoks if al.tokens[t].start > abs_c]
        if not before_toks or not after_toks or any(tok_ov[t] for t in before_toks):
            continue

        # 2. Printed box code right after `:` (`einfache Entfernung: 111 25 km`, `Arbeitslohn: 127 [___] EUR`)
        t_first = after_toks[0]
        first_txt = al.tokens[t_first].text
        is_printed_code = (
            not tok_ov[t_first]
            and _CODE_TOK.fullmatch(first_txt)
            and (tok_pr[t_first] or (fold(first_txt) in printed_words and fold(first_txt) not in overlay_words))
        )
        if is_printed_code:
            ov_after = [t for t in after_toks[1:] if tok_ov[t]]
            after_str = line[cpos + 1 :]
            if ov_after:
                ov_s, ov_e = al.tokens[ov_after[0]].start, al.tokens[ov_after[-1]].end
                mid = [t for t in after_toks[1:] if t < ov_after[0]]
                tail = [t for t in after_toks if t > ov_after[-1]]
                if all(
                    not tok_ov[t] and (_CODE_TOK.fullmatch(al.tokens[t].text) or _UNIT_TOK.fullmatch(al.tokens[t].text))
                    for t in mid
                ) and all(not tok_ov[t] and _UNIT_TOK.fullmatch(al.tokens[t].text) for t in tail):
                    lead_m = re.match(r"^([ \t]*(?:[-*+][ \t]+)?)", line)
                    bullet = lead_m.group(1) if lead_m else ""
                    cap_raw = line[len(bullet) : cpos + 1]
                    new_cap = cap_raw if al.tokens[before_toks[0]].text == first_txt else f"{first_txt} {cap_raw}"
                    edits.append((line_s, line_e, f"{bullet}{new_cap} {md[ov_s:ov_e]}"))
                    continue
            elif not any(tok_ov[t] for t in after_toks) and (
                _BLANK_MARK.search(after_str)
                or all(
                    not tok_ov[t] and (_CODE_TOK.fullmatch(al.tokens[t].text) or _UNIT_TOK.fullmatch(al.tokens[t].text))
                    for t in after_toks
                )
            ):
                lead_m = re.match(r"^([ \t]*(?:[-*+][ \t]+)?)", line)
                bullet = lead_m.group(1) if lead_m else ""
                cap_raw = line[len(bullet) : cpos + 1]
                new_cap = cap_raw if al.tokens[before_toks[0]].text == first_txt else f"{first_txt} {cap_raw}"
                edits.append((line_s, line_e, f"{bullet}{new_cap} [___]"))
                continue

        # 3. Blank slot with only printed hint/unit after `:`
        if not any(tok_ov[t] for t in after_toks):
            mb = _AFTER_BLANK_PRINTED.match(line[cpos:])
            if mb and all(tok_pr[t] for t in after_toks) and len(after_toks) <= 4 and ":" not in mb.group(2):
                edits.append((line_s + cpos, line_e, mb.group(1)))
                continue
            if len(after_toks) <= 2 and all(tok_pr[t] and _UNIT_TOK.fullmatch(al.tokens[t].text) for t in after_toks):
                edits.append((line_s + cpos + 1, line_e, " [___]"))
                continue

        # 4. Hard trailing printed furniture (`▶`, `►`, `**`) right after an overlay value
        ov_after = [t for t in after_toks if tok_ov[t]]
        if ov_after:
            last_ov = ov_after[-1]
            tail_toks = [t for t in after_toks if t > last_ov]
            if len(tail_toks) >= 2 and all(tok_pr[t] and not tok_ov[t] for t in tail_toks):
                between = md[al.tokens[last_ov].end : al.tokens[tail_toks[0]].start]
                if "▶" in between or "►" in between or "**" in between:
                    split_at = al.tokens[last_ov].end
                    m_close = re.match(r"[ \t]*(?:\*{1,2}|_{1,2}|</u>|</strong>|</b>)(?=[ \t])", md[split_at:line_e])
                    if m_close:
                        split_at += m_close.end()
                    edits.append((split_at, split_at, "\n"))

    return edits

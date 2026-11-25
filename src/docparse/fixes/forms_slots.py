"""Printed text the read wrote in a blank field's slot (form options `blanks`, `cells` and `hints`).

A form field's value is read from its caption's own slot: the rest of the caption's line or table cell, up to the next
caption. On a filled form every value is typed text (the overlay); printed text in a slot is a hint, a box code, a
column number or an instruction, and a reader that takes it for the value invents one. These options rewrite the
lines and cells holding no typed text so the slot after the caption is a blank marker and the printed text sits
where a reader skips it: a bracketed hint, or a line of its own.

    blanks  lines with blank markers (`[___]`): printed text after a marker starts a line of its own (one field per
            line), printed text between a caption's colon and its marker becomes a bracketed hint before the colon
            (`Arbeitslohn: 127 [___] EUR` -> `Arbeitslohn (127): [___]` / `EUR`), a lone box code there moves to the
            caption's front (`Elle s'est terminée le : D15 [___]` -> `D15 Elle s'est terminée le: [___]`), and a
            caption in two parts gets a colon before its marker (`支店名 Name of branch: [___]`)
    cells   table cells holding only printed text: the same for a cell with a marker; a cell ending in a box's
            decoration (column number, printed cents, `(4)`) or in a worded hint gets the decoration as a bracketed
            hint and a blank marker (`Totale credito 5 ,00` -> `Totale credito (5 ,00): [___]`,
            `Branch number (5 digits)` -> `Branch number (5 digits): [___]`); not beside check boxes, and not where
            an emptied slot would read the printed cell below
    hints   lines without a marker: the same decoration rule, and a short printed tail after a caption's colon
            (`Änderung der Religion: 1 = Austritt, 2 = Wechsel` -> `Änderung der Religion (1 = Austritt, 2 =
            Wechsel): [___]`)

Text is moved only where every word of it is aligned to a printed word of the PDF and nothing on the line or in the
cell is typed.
"""

from __future__ import annotations

import bisect
import html
import re
from collections.abc import Callable

from docparse.fixes.align import fold
from docparse.fixes.forms import FormPage

MARK = re.compile(r"\[[ \t]*_{2,}[ \t]*\]|_{3,}")
BOXES = re.compile(
    r"\[[ \t]*(?:[xX✓✔✗✘☒☑*]|checked|unchecked|selected|unselected)?[ \t]*\]|[☐☑☒□■○●◯✓✔]|\([ xX●•✓]?\)", re.I
)
EMPH = re.compile(r"\*+|(?<!\w)_+(?!\w)|</?(?:u|strong|b|em|i)>")
BOX_CODE = re.compile(r"[0-9]{2,5}[a-zA-Z]?|[0-9]{1,2}\.[0-9]{1,3}|[A-Z]{1,2}[0-9]{1,3}|[0-9][A-Z]{2,3}")
LEAD = re.compile(r"[ \t]*(?:[-*+][ \t]+|#{1,6}[ \t]+|[0-9]{1,2}[.)][ \t]+)?")
LABEL_COLON = re.compile(r"[:：](?=[ \t*_<]|$)")
DATE_UNITS = frozenset(
    "d dd jj day days jour tag dia dag m mm month mois monat mes maand y yy yyyy year annee année jahr ano año anio "
    "jaar aaaa".split()
)
CHOICE_WORDS = frozenset("yes no oui non ja nein si sí".split())
_TABLE = re.compile(r"<table\b.*?</table\s*>", re.S | re.I)
_BRACKET = re.compile(r"\((?:[^()]|\([^()]*\))*\)|\[[^\[\]]*\]")
# a trailing box decoration: bracket groups, numbers, signs (`(col. 7 minus col. 21) 9`, `5 ,00`, `60%`, `( 4 )`)
_DECO = re.compile(r"(?:[ \t]*(?:\((?:[^()]|\([^()]*\))*\)|\[[^\[\]]*\]|[0-9][0-9.,'’]*%?|[,.;:%€$£–-]))+[ \t]*$")
_WORDS = re.compile(r"[^\W\d_]{2,}")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_LATIN = re.compile(r"[A-Za-z]{2,}")
HINT_TAIL = 12  # a printed tail after a caption's colon this long or shorter is a hint (option `hints`)

Printed = Callable[[int, int], bool]


def fold_word(text: str) -> str:
    return "".join(fold(ch) for ch in text)


def _clean(text: str) -> str:
    """Text for inside a bracketed hint: no emphasis, brackets or colons."""
    t = EMPH.sub(" ", text)
    t = re.sub(r"[()\[\]{}]", " ", t)
    t = re.sub(r"[:：]", " ", t)
    return " ".join(t.split())


def _two_parts(head: str) -> bool:
    """A caption in two parts: an emphasised run followed by plain words, or words in two scripts."""
    t = head[LEAD.match(head).end() :].strip()
    if _CJK.search(t) and _LATIN.search(t):
        return True
    m = re.match(r"(\*\*|\*|<u>|<b>|<strong>).*?(\*\*|\*|</u>|</b>|</strong>)", t)
    return bool(m and _WORDS.search(EMPH.sub(" ", t[m.end() :])))


def _unit(text: str) -> bool:
    words = re.findall(r"\w+", EMPH.sub(" ", text).lower())
    return len(words) == 1 and words[0] in DATE_UNITS


def _choice(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return bool(words) and all(w in CHOICE_WORDS for w in words)


def _head(head: str, at: int, printed: Printed | None) -> str:
    """The text before a blank marker, written so the marker is its caption's slot."""
    if not _WORDS.search(head):
        return head
    # a colon inside a bracketed hint is no label colon: `Namenszusatz (Beispiel: Freifrau, Graf):`
    head = _BRACKET.sub(lambda m: re.sub(r"[ \t]*[:：][ \t]*", " ", m.group(0)), head)
    colons = list(LABEL_COLON.finditer(head))
    if colons:
        c = colons[-1].start()
        tail = _clean(head[c + 1 :])
        if not re.search(r"\w", tail):
            new = _before_colon(head[:c], at, printed)
            return head if new is None else new + ": " + head[c + 1 :].lstrip(" \t*_")
        if _choice(tail):
            return head
        if printed is not None and not printed(at + c + 1, at + len(head)):
            return head
        cap = head[:c].rstrip(" \t*_")
        if cap.count("**") % 2:
            cap += "**"
        if BOX_CODE.fullmatch(tail):  # a lone box code labels the caption
            lead = LEAD.match(cap).end()
            return cap[:lead] + tail + " " + cap[lead:].lstrip() + ": "
        return cap + " (" + tail + "): "
    if _two_parts(head) and not _unit(head):
        return head.rstrip() + ": "
    return head


_CODE_END = re.compile(r"(?:^|(?<=[ \t*_]))\**(" + BOX_CODE.pattern + r")\**[ \t*_]*$")
_ITALIC_OPEN = re.compile(r"(?:^|(?<=[ \t]))[*_](?=[^\s*_])")


def _before_colon(cap: str, at: int, printed: Printed | None) -> str | None:
    """A caption before its colon and blank marker that ends in printed extras: a box code moves to the front, an
    italic hint or a box's decoration becomes a bracketed hint (`gain imposable *Après abattement* 1TZ:` ->
    `1TZ gain imposable (Après abattement):`). None when it ends in neither."""
    lead = LEAD.match(cap).end()
    code, body, moved_from = "", cap, len(cap)
    m = _CODE_END.search(cap)
    if m and m.start(1) > lead and _WORDS.search(EMPH.sub(" ", cap[lead : m.start(1)])):
        before = cap[: m.start(1)].rstrip(" \t")
        # a code: lettered (1TZ, D15) or set apart by emphasis; not the last number of a sum (`233 + 237`)
        apart = re.search(r"[*_>][ \t]*$", before) or re.search(r"[A-Z]", m.group(1))
        if apart and not re.search(r"[-+×=/][ \t*_]*$", before):
            code, body, moved_from = m.group(1), cap[: m.start(1)], m.start(1)
            first = re.match(r"[\[(*_ \t]*([0-9]{1,3}[a-zA-Z]?)\b", cap[lead:])
            if first and first.group(1) == code:
                code = ""  # the line's own number printed again at its box: dropped
    body = body.rstrip(" \t*_")
    hint = ""
    opens = list(_ITALIC_OPEN.finditer(body))
    if opens and opens[-1].start() > lead and _WORDS.search(EMPH.sub(" ", body[lead : opens[-1].start()])):
        moved_from = min(moved_from, opens[-1].start())
        hint = "(" + _clean(body[opens[-1].start() :]) + ")"
        body = body[: opens[-1].start()]
    elif (split := _deco(body, lead)) is not None and re.search(r"[(\[]", body[len(split[0]) :]):
        # a bracketed format hint (`(999) 999-9999`); a bare number ends the caption's own text (`Attach Form 4797`)
        moved_from = min(moved_from, len(split[0]))
        body, hint = split
    if not (code or hint) and moved_from == len(cap):
        return None
    body = body.rstrip()
    if body.count("**") % 2:
        body += "**"
    out = body[:lead] + (code + " " if code else "") + body[lead:].lstrip() + (" " + hint if hint else "")
    if " ".join(out.split()) == " ".join(cap.split()):
        return None
    if printed is not None and not printed(at + moved_from, at + len(cap)):
        return None
    return out


def _opens_field(text: str) -> bool:
    """Text after a blank marker that opens a field of its own: not a continuation (`, which ...`) nor a date part."""
    if re.match(r"[ \t]*[,.;:)\]/-]|[ \t]*[a-z]", text):
        return False
    return not _unit(re.split(r"[:：]", text)[0])


def blank_line(line: str, at: int = 0, printed: Printed | None = None) -> str:
    """A line with blank markers as one field per line, each marker its caption's slot (option `blanks`)."""
    groups: list[list[int]] = []
    for m in MARK.finditer(line):
        if groups and not re.search(r"\w", line[groups[-1][1] : m.start()]):
            groups[-1][1] = m.end()  # `[___]/[___]/[___]`: one slot
        else:
            groups.append([m.start(), m.end()])
    if not groups:
        return line
    out, last = "", 0
    for i, (g0, g1) in enumerate(groups):
        head = line[last:g0]
        new = _head(head, at + last, printed)
        out += ("\n" + new.lstrip() if i and _opens_field(head) else new) + line[g0:g1]
        last = g1
    rest = line[last:]
    moved = re.search(r"\w", rest) and _opens_field(rest)
    if moved and (printed is None or printed(at + last, at + len(line))):
        return out + "\n" + rest.lstrip()
    return out + rest


def hint_text(text: str) -> str | None:
    """Text ending in a box's decoration or a worded hint, with that end as a bracketed hint before a blank marker;
    None when it has no such end (options `cells` and `hints`)."""
    if not _WORDS.search(text) or BOXES.search(MARK.sub(" ", text)):
        return None
    if MARK.search(text):
        marks = list(MARK.finditer(text))
        head, rest = text[: marks[0].start()], MARK.sub(" ", text[marks[0].end() :])
        if re.search(r"\w", rest):
            head = head + " " + rest
        new = _head(head, 0, None)
        if new == head:
            if not re.search(r"\w", rest):
                return None
            new = head.rstrip() + ": "
        return new + "[___]"
    split = _deco(text)
    if split is None:
        return None
    body, hint = split
    body = re.sub(r"[ \t]*[:：]$", "", body.rstrip())
    if body.count("**") % 2:
        body = body.rstrip(" *") + "**"
    return body + " " + hint + ": [___]"


def _deco(text: str, lead: int = 0) -> tuple[str, str] | None:
    """(text before, bracketed hint) for text ending in a box's decoration (`5 ,00`, `(4)`, `(col. 7 minus col. 21) 9`)
    or in a worded hint (`(5 digits)`); None when it ends in neither."""
    m = _DECO.search(text)
    if not m or m.start() <= lead or not re.search(r"[0-9A-Za-z]", m.group(0)):
        return None
    if m.start() > 0 and text[m.start() - 1].isalnum() and not m.group(0)[0].isspace():
        return None  # part of a word: IHT401, W-2, assuré(e)
    body, deco = text[: m.start()], m.group(0)
    if not _WORDS.search(EMPH.sub(" ", body[lead:])):
        return None
    worded = [g for g in _BRACKET.finditer(deco) if _WORDS.search(g.group(0))]
    if worded:  # the last worded hint keeps its place (the caption may own it); what follows it joins it
        w = worded[-1]
        body = body + deco[: w.start()]
        if not re.search(r"[0-9A-Za-z]", deco[w.end() :]):
            return body, re.sub(r"[ \t]*[:：][ \t]*", " ", w.group(0))
        deco = deco[w.start() :]
    deco = _clean(deco)
    if not re.search(r"\w", deco) or _choice(deco):
        return None
    return body, "(" + deco + ")"


def colon_text(line: str) -> str | None:
    """A line `caption: short printed tail` as `caption (tail): [___]` (option `hints`)."""
    colons = list(LABEL_COLON.finditer(line))
    if not colons:
        return None
    tail = _clean(line[colons[-1].end() :])
    if not tail or len(tail.split()) > HINT_TAIL or _choice(tail):
        return None
    new = blank_line(line + " [___]")
    return new if new != line + " [___]" else None


class _Flags:
    """Per markdown token of a form page: typed (overlay) text, printed text."""

    def __init__(self, page: FormPage) -> None:
        al = page.al
        self.starts = [t.start for t in al.tokens]
        self.typed = [False] * len(al.tokens)
        self.printed = [False] * len(al.tokens)
        for k, p in enumerate(al.amatch):
            if p is not None:
                (self.typed if page.owner[al.p_word[p]] >= 0 else self.printed)[al.a_tok[k]] = True
        self.share = al.token_share()
        # a word the alignment left out (a repeated block) counts as printed when the page prints it and never types it
        printed_keys = {fold_word(w.text) for w in page.printed}
        typed_keys = {fold_word(w.text) for f in page.fills for w in f.words} | {f.key for f in page.fills}
        for t, tok in enumerate(al.tokens):
            key = fold_word(tok.text)
            if self.share[t] == 0 and key in printed_keys and key not in typed_keys:
                self.printed[t], self.share[t] = True, 1.0

    def span(self, lo: int, hi: int) -> range:
        return range(bisect.bisect_left(self.starts, lo), bisect.bisect_left(self.starts, hi))

    def any_typed(self, lo: int, hi: int) -> bool:
        return any(self.typed[t] for t in self.span(lo, hi))

    def all_printed(self, lo: int, hi: int) -> bool:
        return all(self.printed[t] and not self.typed[t] and self.share[t] >= 0.5 for t in self.span(lo, hi))


def slot_edits(page: FormPage, options: dict) -> list[tuple[int, int, str]]:
    """Line and table rewrites for the enabled options (`blanks`, `cells`, `hints`)."""
    md = page.markdown
    flags = _Flags(page)
    edits: list[tuple[int, int, str]] = []
    tables = [(m.start(), m.end()) for m in _TABLE.finditer(md)]
    if options.get("cells"):
        for start, end in tables:
            new = _table(md, start, end, flags)
            if new != md[start:end]:
                edits.append((start, end, new))
    if not (options.get("blanks") or options.get("hints")):
        return edits
    pos = 0
    for line in md.split("\n"):
        lo, hi = pos, pos + len(line)
        pos = hi + 1
        if (
            any(a < hi and lo < b for a, b in tables)
            or "|" in line
            or "<t" in line
            or BOXES.search(MARK.sub(" ", line))
            or flags.any_typed(lo, hi)
        ):
            continue
        new = None
        if MARK.search(line):
            if options.get("blanks"):
                new = blank_line(line, lo, flags.all_printed)
        elif options.get("hints") and _WORDS.search(line) and flags.all_printed(lo, hi):
            new = hint_text(line) or colon_text(line)
        if new is not None and new != line:
            edits.append((lo, hi, new))
    return edits


def _table(md: str, start: int, end: int, flags: _Flags) -> str:
    """The table at md[start:end] with its printed caption cells rewritten (option `cells`)."""
    from docparse.fixes.tables_html import _CELL, _ROW, parse_table

    raw = md[start:end]
    t = parse_table(raw)
    if t is None:
        return raw
    spans = []
    for rm in _ROW.finditer(md, start, end):
        body = rm.start() + re.match(r"<tr\b[^>]*>", rm.group(0), re.I).end()
        spans.append([(cm.start(3), cm.end(3)) for cm in _CELL.finditer(md, body, rm.end())])
    if [len(r.cells) for r in t.rows] != [len(s) for s in spans]:
        return raw
    state: dict[tuple[int, int], str] = {}  # grid slot -> empty / typed / printed
    for row, cells in zip(t.rows, spans, strict=True):
        for cell, (a, b) in zip(row.cells, cells, strict=True):
            kind = "empty" if not flags.span(a, b) else ("typed" if flags.any_typed(a, b) else "printed")
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    state.setdefault((cell.row + dr, cell.col + dc), kind)
    for row, cells in zip(t.rows, spans, strict=True):
        if any(BOXES.search(MARK.sub(" ", c.text)) for c in row.cells):
            continue  # option labels beside their boxes
        for cell, (a, b) in zip(row.cells, cells, strict=True):
            if state.get((cell.row, cell.col)) != "printed" or not flags.all_printed(a, b):
                continue
            right = state.get((cell.row, cell.col + cell.colspan), "none")
            below = state.get((cell.row + cell.rowspan, cell.col), "none")
            if right in ("none", "empty") and below == "printed":
                continue  # an emptied slot would read the printed cell below
            new = hint_text(cell.text)
            if new is not None:
                cell.inner, cell.raw = html.escape(new, quote=False), ""
                row.edited = t.changed = True
    return t.render() if t.changed else raw

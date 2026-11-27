"""Text corrections of the page reading from the PDF text layer (design §6, text). Each is one option:

    respell   a word the reading spells slightly differently from the text layer takes the layer's spelling, where
              the layer can be trusted: it is visible (not an OCR layer under a scan or in invisible text), its
              glyphs map to unicode, and it holds the reading's words (80% of its tokens occur in the reading).
              Only one-for-one swaps (up to two words in a row) anchored by two matching words on each side count,
              and only small ones: one edit in a short word, a quarter of the letters in a long one, one digit in a
              number of the same length. Edits where the layer shows its own artefacts are refused: a modifier
              letter, a capital inside a lower-case word, or a change at a ligature (fi, fl, ff, ft) the layer may
              have mis-mapped
    empty_read  a read with fewer words than 5% of a visible, well-mapped text layer of 50 words or more (the model
              wrote next to nothing for a page full of text) is replaced by the text layer as plain lines. Only the
              read's length counts: a long read that shares no words with the layer means a broken layer

A region the options do not change keeps its base candidate, so it is written back byte for byte.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from docparse.core.ir import Candidate, Page, Provenance, Region
from docparse.fixes.align_text import TOKEN, key, opcodes

# `picture_alt` is applied where the page is read (pipeline.py): a picture's description goes into its alt text
TEXT_OPTIONS = frozenset({"respell", "empty_read", "picture_alt"})
READER = "fix.text"
LAYER_SUPPORT = 0.8  # share of the layer's tokens the reading must hold
MAX_HIDDEN = 0.5
MAX_MAP_ERRORS = 0.01
CONTEXT = 2  # matching tokens needed on each side of a swap
MAX_SWAP = 2  # tokens swapped one for one in a row
MIN_LETTERS = 4
LETTER_RATE = 0.25
LIGATURES = ("fi", "fl", "ff", "ft")
_MARKUP = re.compile(
    r"\$\$[\s\S]*?\$\$"  # display math
    r"|\\\[[\s\S]*?\\\]"  # \[ ... \] display math
    r"|\\\([\s\S]*?\\\)"  # \( ... \) inline math
    r"|\$(?!\s)[^$\n]*?(?:\\[A-Za-z]+|[_^{}=<>])[^$\n]*?(?<!\s)\$"  # inline $...$ with math syntax (not currency)
    r"|\\[A-Za-z]+"  # LaTeX commands (\pmod, \equiv, \alpha, ...)
    r"|<[^>\n]*>"  # HTML tags
    r"|\]\([^)\n]*\)"  # markdown link targets
)


EMPTY_READ = 0.05  # a read with fewer words than this share of the layer's counts as empty


def needs_layer(options: dict) -> bool:
    return bool(options.get("respell") or options.get("empty_read"))


def empty_read(markdown: str, rendered) -> str | None:
    """The text layer as plain lines when the read is empty next to a visible, well-mapped layer; else None."""
    from docparse.fixes.fallback import MIN_TOKENS, text_dump
    from docparse.fixes.layout_refine import words_in_points

    if rendered.hidden is None or rendered.hidden > MAX_HIDDEN or (rendered.map_errors or 0) > MAX_MAP_ERRORS:
        return None
    words = words_in_points(rendered)
    layer = sum(len(TOKEN.findall(w.text)) for w in words)
    if len(words) < MIN_TOKENS or len(_text_tokens(markdown)) >= EMPTY_READ * layer:
        return None
    return text_dump(words)


def _select(region: Region, raw: str) -> None:
    cand = Candidate(
        id=f"{region.id}c{len(region.candidates)}",
        reader=READER,
        raw=raw,
        content=region.chosen().content,
        prov=[Provenance(source="pdf_text", ref=READER, detail="respell")],
    )
    region.candidates.append(cand)
    region.selected = cand.id


def _text_tokens(raw: str) -> list[tuple[int, int, str]]:
    """(start, end, surface) of the tokens of the text a reader sees: tags and link targets masked."""
    masked = _MARKUP.sub(lambda m: " " * len(m.group()), raw)
    return [(m.start(), m.end(), m.group()) for m in TOKEN.finditer(masked)]


def _distance(a: str, b: str) -> int:
    """Levenshtein distance (words are short; plain dynamic programming)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _artefact(read: str, layer: str) -> bool:
    """Would the swap copy a text-layer artefact rather than fix a misreading?"""
    if any(unicodedata.category(c) == "Lm" for c in layer):
        return True
    for x in range(1, len(layer)):
        if layer[x].isupper() and layer[x - 1].islower() and not (x < len(read) and read[x].isupper()):
            return True
    low = read.lower()
    spots = {p for lig in LIGATURES for m in re.finditer(lig, low) for p in range(m.start(), m.end())}
    if spots:
        for tag, i1, i2, _j1, _j2 in SequenceMatcher(None, low, layer.lower(), autojunk=False).get_opcodes():
            if tag != "equal" and spots.intersection(range(i1, max(i2, i1 + 1))):
                return True
    return False


def _close(read: str, layer: str) -> bool:
    if read.lower() == layer.lower() or _artefact(read, layer):
        return False
    distance = _distance(read.lower(), layer.lower())
    if any(c.isdigit() for c in read + layer):
        return len(read) == len(layer) >= 2 and distance <= 1
    longest = max(len(read), len(layer))
    return longest >= MIN_LETTERS and distance <= max(1, int(LETTER_RATE * longest))


def trusted(rendered, reading_keys: list[str], layer_keys: list[str]) -> bool:
    """Can the text layer correct the reading on this page?"""
    if rendered.hidden is None or rendered.hidden > MAX_HIDDEN or (rendered.map_errors or 0) > MAX_MAP_ERRORS:
        return False
    have, need = Counter(reading_keys), Counter(layer_keys)
    total = sum(need.values())
    return bool(total) and sum(min(n, have[k]) for k, n in need.items()) / total >= LAYER_SUPPORT


def refine_text(page: Page, rendered, options: dict) -> None:
    """Apply the enabled options to the page's regions (a changed region gets a new selected candidate)."""
    words = getattr(rendered, "text_words", None) if rendered is not None else None
    if not options.get("respell") or not words:
        return
    raws = [region.chosen().raw for region in page.regions]
    layer = [m.group() for w in words for m in TOKEN.finditer(w.text)]
    per_region = [_text_tokens(raw) for raw in raws]
    flat = [(r, t) for r, toks in enumerate(per_region) for t in toks]
    reading_keys, layer_keys = [key(t[2]) for _, t in flat], [key(s) for s in layer]
    if not trusted(rendered, reading_keys, layer_keys):
        return
    swaps: dict[int, list[tuple[int, int, str]]] = {}
    ops = opcodes(reading_keys, layer_keys)
    for x, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag != "replace" or i2 - i1 != j2 - j1 or i2 - i1 > MAX_SWAP:
            continue
        before = x > 0 and ops[x - 1][0] == "equal" and ops[x - 1][2] - ops[x - 1][1] >= CONTEXT
        after = x + 1 < len(ops) and ops[x + 1][0] == "equal" and ops[x + 1][2] - ops[x + 1][1] >= CONTEXT
        pairs = list(zip(range(i1, i2), range(j1, j2), strict=True))
        if before and after and all(_close(flat[i][1][2], layer[j]) for i, j in pairs):
            for i, j in pairs:
                r, (start, end, _surface) = flat[i]
                swaps.setdefault(r, []).append((start, end, layer[j]))
    for r, edits in swaps.items():
        raw = raws[r]
        for start, end, text in sorted(edits, reverse=True):
            raw = raw[:start] + text + raw[end:]
        _select(page.regions[r], raw)

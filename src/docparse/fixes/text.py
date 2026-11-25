"""Text corrections of the page reading from the PDF text layer (design §6, text). Each is one option:

    respell   a word the reading spells slightly differently from the text layer takes the layer's spelling, where
              the layer can be trusted: it is visible (not an OCR layer under a scan or in invisible text), its
              glyphs map to unicode, and it holds the reading's words (80% of its tokens occur in the reading).
              Only one-for-one swaps anchored by two matching words on each side count, and only small ones: one
              edit in a short word, a quarter of the letters in a long one, one digit in a number of the same
              length. Edits where the layer shows its own artefacts are refused: a modifier letter, a capital
              inside a lower-case word, or a change at a ligature (fi, fl, ff, ft) the layer may have mis-mapped

A region the options do not change keeps its base candidate, so it is written back byte for byte.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from rapidfuzz.distance import Levenshtein

from docparse.core.ir import Candidate, Page, Provenance, Region
from docparse.fixes.align import TOKEN, key

TEXT_OPTIONS = frozenset({"respell"})
READER = "fix.text"
LAYER_SUPPORT = 0.8  # share of the layer's tokens the reading must hold
MAX_HIDDEN = 0.5
MAX_MAP_ERRORS = 0.01
CONTEXT = 2  # matching tokens needed on each side of a swap
MIN_LETTERS = 4
LETTER_RATE = 0.25
LIGATURES = ("fi", "fl", "ff", "ft")
_MARKUP = re.compile(r"<[^>\n]*>|\]\([^)\n]*\)")  # tags and link targets are not text


def needs_layer(options: dict) -> bool:
    return bool(options.get("respell"))


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
        for tag, i1, i2, _j1, _j2 in Levenshtein.opcodes(low, layer.lower()):
            if tag != "equal" and spots.intersection(range(i1, max(i2, i1 + 1))):
                return True
    return False


def _close(read: str, layer: str) -> bool:
    if read.lower() == layer.lower() or _artefact(read, layer):
        return False
    distance = Levenshtein.distance(read.lower(), layer.lower())
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
    ops = SequenceMatcher(None, reading_keys, layer_keys, autojunk=False).get_opcodes()
    for x, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag != "replace" or i2 - i1 != 1 or j2 - j1 != 1:
            continue
        before = x > 0 and ops[x - 1][0] == "equal" and ops[x - 1][2] - ops[x - 1][1] >= CONTEXT
        after = x + 1 < len(ops) and ops[x + 1][0] == "equal" and ops[x + 1][2] - ops[x + 1][1] >= CONTEXT
        r, (start, end, surface) = flat[i1]
        if before and after and _close(surface, layer[j1]):
            swaps.setdefault(r, []).append((start, end, layer[j1]))
    for r, edits in swaps.items():
        raw = raws[r]
        for start, end, text in sorted(edits, reverse=True):
            raw = raw[:start] + text + raw[end:]
        _select(page.regions[r], raw)

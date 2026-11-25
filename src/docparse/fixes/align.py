"""Which markdown word is which PDF word.

A page read and the PDF text layer say mostly the same thing, in different orders (columns, sidebars, text an
editing tool added last) and with different word splits (kerning, one character per box). Both sides are folded
to lowercase letters and digits and aligned character by character: first in order (difflib), then each long
unmatched run of the markdown against the PDF characters still unused, in any order, where that text occurs
only once on the page. A character match ties a markdown token to a PDF word.

Visible text only: HTML tags, entities, check-box marks (`[x]`, `[ ]`) and image references are not words.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

from docparse.fixes.layout_refine import PtWord

_HIDDEN = re.compile(
    r"<[^<>\n]*>"  # HTML tags
    r"|&(?:#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,7});"  # entities
    r"|!\[[^\]\n]*\]\([^)\n]*\)"  # image references
    r"|\[[ \t]*(?:[xX✓✔✗✘☒☑*]|checked|unchecked|selected)?[ \t]*\]"  # check-box marks
)


def fold(ch: str) -> str:
    """A character as the aligner sees it: NFKC, lowercase, letters and digits only (may be empty or longer)."""
    return "".join(c for c in unicodedata.normalize("NFKC", ch).lower() if c.isalnum())


@dataclass(frozen=True)
class Token:
    """A word of the markdown's visible text: md[start:end]."""

    start: int
    end: int
    text: str


def visible_tokens(markdown: str) -> list[Token]:
    """Maximal runs of letters and digits outside tags, entities, box marks and image references."""
    masked = _HIDDEN.sub(lambda m: " " * len(m.group(0)), markdown)
    out: list[Token] = []
    start = None
    for i, ch in enumerate(masked):
        if fold(ch):
            if start is None:
                start = i
        elif start is not None:
            out.append(Token(start, i, markdown[start:i]))
            start = None
    if start is not None:
        out.append(Token(start, len(markdown), markdown[start:]))
    return out


@dataclass
class Alignment:
    tokens: list[Token]
    words: list[PtWord]
    a_chars: str  # folded markdown characters, token by token
    a_tok: list[int]  # token of each markdown character
    p_chars: str  # folded PDF characters, word by word
    p_word: list[int]  # word of each PDF character
    amatch: list[int | None] = field(default_factory=list)  # markdown character -> PDF character
    pused: list[bool] = field(default_factory=list)

    def match(self, a: int, p: int) -> None:
        self.amatch[a] = p
        self.pused[p] = True

    def token_words(self, t: int) -> list[int]:
        """PDF words that characters of token t matched, in the token's character order (no repeats)."""
        out: list[int] = []
        for k, owner in enumerate(self.a_tok):
            if owner == t and self.amatch[k] is not None:
                w = self.p_word[self.amatch[k]]
                if w not in out:
                    out.append(w)
        return out

    def word_tokens(self) -> list[list[int]]:
        """Per PDF word, the markdown tokens matched to it (in markdown order)."""
        out: list[list[int]] = [[] for _ in self.words]
        for k, p in enumerate(self.amatch):
            if p is not None:
                w, t = self.p_word[p], self.a_tok[k]
                if t not in out[w]:
                    out[w].append(t)
        return out

    def word_share(self) -> list[float]:
        """Per PDF word, the share of its folded characters matched to the markdown."""
        total = [0] * len(self.words)
        hit = [0] * len(self.words)
        for p, w in enumerate(self.p_word):
            total[w] += 1
            hit[w] += self.pused[p]
        return [h / n if n else 0.0 for h, n in zip(hit, total, strict=True)]

    def token_share(self) -> list[float]:
        """Per markdown token, the share of its folded characters matched to the PDF."""
        total = [0] * len(self.tokens)
        hit = [0] * len(self.tokens)
        for k, t in enumerate(self.a_tok):
            total[t] += 1
            hit[t] += self.amatch[k] is not None
        return [h / n if n else 0.0 for h, n in zip(hit, total, strict=True)]


def _chars(texts: list[str]) -> tuple[str, list[int]]:
    chars, owner = [], []
    for i, text in enumerate(texts):
        for ch in text:
            for c in fold(ch):
                chars.append(c)
                owner.append(i)
    return "".join(chars), owner


def _whole(owner: list[int], lo: int, hi: int) -> bool:
    """Does [lo, hi) start and end on owner boundaries (whole tokens or whole words)?"""
    return (lo == 0 or owner[lo - 1] != owner[lo]) and (hi == len(owner) or owner[hi] != owner[hi - 1])


def _pieces(a_tok: list[int], p_word: list[int], a0: int, p0: int, size: int) -> list[tuple[int, int]]:
    """A matching block cut wherever its markdown token or its PDF word changes (offsets inside the block)."""
    out, start = [], 0
    for k in range(1, size):
        if a_tok[a0 + k] != a_tok[a0 + k - 1] or p_word[p0 + k] != p_word[p0 + k - 1]:
            out.append((start, k))
            start = k
    out.append((start, size))
    return out


def phrases(words: list[PtWord]) -> list[list[int]]:
    """Runs of words consecutive in content order, on one line and close together: phrases as the PDF stores
    them."""
    out: list[list[int]] = []
    for i, w in enumerate(words):
        if out:
            p = words[out[-1][-1]]
            height = max(p.y1 - p.y0, w.y1 - w.y0)
            if abs(p.cy - w.cy) < 0.5 * height and -0.5 * height <= w.x0 - p.x1 <= 1.5 * height:
                out[-1].append(i)
                continue
        out.append([i])
    return out


def reading_order(a_chars: str, words: list[PtWord], min_key: int = 6) -> list[int]:
    """Word indices with the PDF's phrases sorted into the markdown's order: a phrase spelled exactly once in the
    markdown goes where the markdown has it; any other phrase follows the phrase before it in content order.
    (Some PDFs store their text in no reading order at all; an in-order alignment then finds little.)"""
    runs = phrases(words)
    place: list[float] = []
    last = -1.0
    for k, run in enumerate(runs):
        key = "".join(fold(ch) for i in run for ch in words[i].text)
        at = a_chars.find(key) if len(key) >= min_key else -1
        if at >= 0 and a_chars.find(key, at + 1) < 0:
            last = float(at)
        else:
            last += 1e-6
        place.append(last + k * 1e-12)
    return [i for k in sorted(range(len(runs)), key=lambda k: place[k]) for i in runs[k]]


def _firsts(owner: list[int], n: int) -> list[int]:
    """Index of the first character of each of n owners (owners in order), plus the end."""
    out = [len(owner)] * (n + 1)
    for k in range(len(owner) - 1, -1, -1):
        out[owner[k]] = k
    for i in range(n - 1, -1, -1):  # an owner without characters starts where the next one does
        out[i] = min(out[i], out[i + 1])
    return out


def align(
    markdown: str,
    words: list[PtWord],
    min_block: int = 3,
    seg_min: int = 8,
    passes: int = 3,
    reorder: bool = False,
) -> Alignment:
    """Align the markdown's visible tokens with PDF words given in content-stream order. With `reorder`, the
    PDF's phrases are first sorted into the markdown's order (`reading_order`)."""
    tokens = visible_tokens(markdown)
    a_chars, a_tok = _chars([t.text for t in tokens])
    order = reading_order(a_chars, words) if reorder else list(range(len(words)))
    p_chars, p_seq = _chars([words[i].text for i in order])
    p_word = [order[k] for k in p_seq]
    al = Alignment(tokens, words, a_chars, a_tok, p_chars, p_word, [None] * len(a_chars), [False] * len(p_chars))
    if not a_chars or not p_chars:
        return al

    matcher = difflib.SequenceMatcher(None, autojunk=False)

    def unused() -> None:
        matcher.set_seq2("".join("\x00" if used else c for c, used in zip(p_chars, al.pused, strict=True)))

    def take(a0: int, p0: int, size: int, out_of_order: bool) -> None:
        if out_of_order:
            if size < seg_min or p_chars.count(a_chars[a0 : a0 + size]) > 1:
                return  # repeated text: an out-of-order match cannot tell which copy
        elif size < min_block and not (_whole(a_tok, a0, a0 + size) and _whole(p_word, p0, p0 + size)):
            return  # a stray one- or two-letter match inside other words
        for s0, s1 in _pieces(a_tok, p_word, a0, p0, size):
            if s1 - s0 < min_block and not (_whole(a_tok, a0 + s0, a0 + s1) or _whole(p_word, p0 + s0, p0 + s1)):
                continue  # the block runs on into a neighbouring token by a letter or two
            for k in range(s0, s1):
                if al.amatch[a0 + k] is None and not al.pused[p0 + k]:
                    al.match(a0 + k, p0 + k)

    def run(lo: int, hi: int, out_of_order: bool) -> None:
        matcher.set_seq1("".join("\x01" if al.amatch[k] is not None else a_chars[k] for k in range(lo, hi)))
        for m in matcher.get_matching_blocks():
            if m.size:
                take(lo + m.a, m.b, m.size, out_of_order)

    # in order: whole tokens equal to whole PDF words first, then characters inside the gaps between them
    a_first = _firsts(a_tok, len(tokens))
    p_first = _firsts(p_seq, len(order))
    a_keys = [a_chars[a_first[t] : a_first[t + 1]] for t in range(len(tokens))]
    p_items = [k for k in range(len(order)) if p_first[k] < p_first[k + 1]]
    p_keys = [p_chars[p_first[k] : p_first[k + 1]] for k in p_items]
    anchors = [(-1, -1)]
    for m in difflib.SequenceMatcher(None, a_keys, p_keys, autojunk=False).get_matching_blocks():
        for k in range(m.size):
            t, w = m.a + k, p_items[m.b + k]
            for c in range(a_first[t + 1] - a_first[t]):
                al.match(a_first[t] + c, p_first[w] + c)
            anchors.append((t, w))
    anchors.append((len(tokens), len(order)))
    gap = difflib.SequenceMatcher(None, autojunk=False)
    for (t0, w0), (t1, w1) in zip(anchors, anchors[1:], strict=False):
        alo, ahi, plo, phi = a_first[t0 + 1], a_first[t1], p_first[w0 + 1], p_first[w1]
        if alo >= ahi or plo >= phi:
            continue
        gap.set_seqs(a_chars[alo:ahi], p_chars[plo:phi])
        for m in gap.get_matching_blocks():
            if m.size:
                take(alo + m.a, plo + m.b, m.size, False)
    for _ in range(passes):
        unused()  # one matcher per pass; a PDF character taken by an earlier segment of the pass stays taken
        segs, k = [], 0
        while k < len(a_chars):
            if al.amatch[k] is None:
                j = k
                while j < len(a_chars) and al.amatch[j] is None:
                    j += 1
                if j - k >= seg_min:
                    segs.append((k, j))
                k = j
            else:
                k += 1
        if not segs:
            break
        for lo, hi in sorted(segs, key=lambda s: s[0] - s[1]):
            run(lo, hi, True)
    return al

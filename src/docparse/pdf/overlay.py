"""Text that an editing tool drew on top of a finished page: typed-in form values and tick marks.

Fill tools wrap the page's own content in `q ... Q` and append one more content stream holding the new text
in fonts of its own. That is what `overlay_fonts` looks for: a last content stream with text, right after a
stream that only closes graphics states, in fonts no earlier stream uses. A page whose content merely comes in
several chunks fails the closing-`Q` test, and an AcroForm's values are not page content at all.
"""

from __future__ import annotations

import io
import logging
import re

_TF = re.compile(rb"/([^\s/\[\]<>()]+)\s+[-+]?[\d.]+\s+Tf\b")
_ONLY_Q = re.compile(rb"\s*(Q\s*)+")
_SUBSET = re.compile(r"^[A-Z]{6}\+")


def font_name(name: str) -> str:
    """`/ABCDEF+Arial-Bold` and `Arial-Bold` are the same font to both PDF libraries."""
    return _SUBSET.sub("", name.lstrip("/"))


def overlay_fonts(data: bytes, page_index: int = 0) -> frozenset[str]:
    """Base-font names used only by the overlay stream of the page; empty when there is no overlay."""
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # it warns about every repairable xref entry
    try:
        from pypdf import PdfReader
        from pypdf.generic import ArrayObject

        page = PdfReader(io.BytesIO(data)).pages[page_index]
        contents = page.get("/Contents")
        contents = contents.get_object() if contents is not None else None
        if not isinstance(contents, ArrayObject) or len(contents) < 3:
            return frozenset()
        streams = [s.get_object().get_data() for s in contents]
        if not _ONLY_Q.fullmatch(streams[-2]):
            return frozenset()
        last = set(_TF.findall(streams[-1]))
        earlier = {name for body in streams[:-1] for name in _TF.findall(body)}
        if not last or last & earlier:
            return frozenset()
        fonts = (page.get("/Resources") or {}).get("/Font") or {}
        names = set()
        for resource in last:
            font = fonts.get("/" + resource.decode("latin-1"))
            base = font.get_object().get("/BaseFont") if font is not None else None
            if base is None:
                return frozenset()
            names.add(font_name(str(base)))
        return frozenset(names)
    except Exception:  # an unreadable or unusual file simply has no detectable overlay
        return frozenset()

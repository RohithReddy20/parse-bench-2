"""The markdown writer: each region's selected candidate, in reading order, with the original separators."""

from __future__ import annotations

from docparse.core.ir import Document, Page

PAGE_SEPARATOR = "\n\n"


def write_page(page: Page) -> str:
    return page.leading + "".join(r.chosen().raw + r.sep_after for r in page.regions)


def write_document(document: Document) -> str:
    return PAGE_SEPARATOR.join(write_page(p) for p in document.pages)

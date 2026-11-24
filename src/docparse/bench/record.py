"""A parsed Document as the result record the ParseBench evaluator reads (the published-record shape).

Timestamps are constants: the same bytes and config must give byte-identical records (rename invariance).
`raw_output` holds only our version and config hash, nothing another layout adapter could match on.
"""

from __future__ import annotations

from docparse import __version__
from docparse.core.ir import Document
from docparse.write.layout_items import layout_page
from docparse.write.markdown import PAGE_SEPARATOR, write_page

EPOCH = "1970-01-01T00:00:00"


def result_record(example_id: str, pipeline_name: str, document: Document) -> dict:
    page_markdown = [write_page(p) for p in document.pages]
    layout_pages = [lp for p, md in zip(document.pages, page_markdown, strict=True) if (lp := layout_page(p, md))]
    return {
        "request": {"example_id": example_id, "product_type": "parse", "source_file_path": ""},
        "pipeline_name": pipeline_name,
        "product_type": "parse",
        "raw_output": {"docparse": __version__, "config_hash": document.config_hash},
        "output": {
            "task_type": "parse",
            "example_id": example_id,
            "pipeline_name": pipeline_name,
            "pages": [
                {"page_index": p.index, "markdown": md, "line_numbers": []}
                for p, md in zip(document.pages, page_markdown, strict=True)
            ],
            "layout_pages": layout_pages,
            "grounded_pages": [],
            "markdown": PAGE_SEPARATOR.join(page_markdown),
        },
        "started_at": EPOCH,
        "completed_at": EPOCH,
        "latency_in_ms": 0,
    }

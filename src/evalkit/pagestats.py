"""Per-page facts about the benchmark PDFs that the loss analysis splits by. Evaluation side only.

`text_layer_chars`: non-space characters in the page's PDF text layer. A page with almost none is a scan or
an image, where the PDF cannot correct anything and the model's reading stands alone.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from evalkit import paths
from evalkit.explode import split_example_id
from evalkit.standin import source_file

HAS_TEXT_LAYER = 50  # characters; below this the page is treated as having no usable text layer


def _chars(path: str) -> int:
    import pypdfium2 as pdfium

    if not path.lower().endswith(".pdf"):
        return 0
    try:
        pdf = pdfium.PdfDocument(path)
        try:
            text = pdf[0].get_textpage().get_text_range()
        finally:
            pdf.close()
    except Exception:
        return 0
    return sum(1 for ch in text if not ch.isspace() and ch not in "\x00￾")


def text_layer_chars(gt_dir: Path, example_ids: list[str], workers: int = 8) -> dict[str, int]:
    """example id -> characters in the text layer of its (first) page. Cached under the data root."""
    cache = paths.data_root() / "pagestats" / f"{gt_dir.name}.json"
    known: dict[str, int] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [i for i in example_ids if i not in known]
    if todo:
        files = []
        for example_id in todo:
            src = source_file(gt_dir, *split_example_id(example_id))
            files.append(str(src) if src else "")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            counts = list(pool.map(_chars, files, chunksize=32))
        known.update(dict(zip(todo, counts, strict=True)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, sort_keys=True))
    return {i: known[i] for i in example_ids}

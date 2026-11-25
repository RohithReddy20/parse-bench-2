"""Layout items for a page: detector regions filled with the words of OCRia-1's answer.

This is OCRia-1's own region-first method, run through its `ocria_grounding` package (the one piece of
downloaded code we execute, design §7): every word of the answer is placed by its PDF text-layer box, a word
goes to the smallest region holding its centre, and each region becomes one item with the answer's text.

What differs from the published pipeline: there is no attention pass. `ocria_grounding` uses the model's
attention to place words that have no PDF box (pages without a text layer) and image references. Here those
get no position: such words follow their neighbours' region, and pictures come from the detector alone. That
keeps layout CPU-only; whether it costs score is measured (plan Task 15).
"""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path

from docparse.core.ir import Box, LayoutItem
from docparse.models.adapters.attention import Attention
from docparse.models.adapters.heron import Detection
from docparse.pdf.render import RenderedPage

OCRIA_REVISION = "dae83aa51ecd0a8574d415d929a4d9e79348fd2a"
ATTENTION_CELL = 32  # pixels per attention cell (patch 16, merged 2 x 2); only the grid shape matters here


def ocria_code_dir() -> Path:
    """The pinned OCRia-1 snapshot holding `ocria_grounding/` (fetched by `python -m evalkit fetch ocria`)."""
    env = os.environ.get("DOCPARSE_OCRIA_DIR")
    if env:
        return Path(env)
    root = os.environ.get("DOCPARSE_DATA")
    base = Path(root) if root else Path.home() / ".cache" / "docparse-v2"
    return base / "ocria1" / OCRIA_REVISION


@lru_cache(maxsize=1)
def _grounding():
    """Import `ocria_grounding` from the pinned snapshot, and only from there."""
    code = ocria_code_dir()
    if not (code / "ocria_grounding" / "layout.py").exists():
        raise RuntimeError(f"OCRia-1 grounding code not found in {code}")
    sys.path.insert(0, str(code))
    try:
        layout = importlib.import_module("ocria_grounding.layout")
        ground = importlib.import_module("ocria_grounding.ground")
        blocks = importlib.import_module("ocria_grounding.blocks")
    finally:
        sys.path.remove(str(code))
    if Path(layout.__file__).resolve().parent != (code / "ocria_grounding").resolve():
        raise RuntimeError("ocria_grounding was imported from somewhere other than the pinned snapshot")
    return layout, ground, blocks


MIN_TEXT_LAYER_CHARS = 50  # fewer characters than this is a scan with a stray label, not a text layer


def has_text_layer(rendered: RenderedPage) -> bool:
    return sum(len(w.text) for w in rendered.words or []) >= MIN_TEXT_LAYER_CHARS


def unit_count(answer: str) -> int:
    """How many words ("units") OCRia-1's grounding code sees in an answer; an attention result has one map each."""
    _layout, ground_mod, blocks_mod = _grounding()
    return len(ground_mod.units_for(answer, blocks_mod.split_blocks(answer))[0]) if answer.strip() else 0


def ground(
    answer: str, rendered: RenderedPage, detections: list[Detection], attention: Attention | None = None
) -> list[LayoutItem]:
    """Layout items in PDF points for one page, in reading order of the answer. `attention` gives positions to
    words that have no PDF word box; without it such words follow their neighbours."""
    import numpy as np

    layout, ground_mod, blocks_mod = _grounding()
    if not answer.strip():
        return []
    blocks = blocks_mod.split_blocks(answer)
    units, per_block = ground_mod.units_for(answer, blocks)
    if not units:
        return []
    image = rendered.image
    grid = (max(1, image.height // ATTENTION_CELL), max(1, image.width // ATTENTION_CELL))
    maps = np.zeros((len(units), *grid), dtype=np.float32)  # no attention: nothing is placed by it
    has = np.zeros(len(units), dtype=bool)
    if attention is not None and len(attention.idx) == len(units):  # a result for another answer is ignored
        grid = attention.grid
        maps, has = attention.dense()
    words = [{"text": w.text, "box": list(w.box)} for w in rendered.words] if rendered.words else None
    info = layout.page_words(image, answer, blocks, units, per_block, maps, grid, has, words)
    regions = [{"label": d.label, "score": d.score, "box": list(d.box)} for d in detections]
    scale = rendered.scale
    items = []
    for it in layout.place(answer, info, regions):
        x0, y0, x1, y1 = (float(v) * scale for v in it["box"])
        if x1 <= x0 or y1 <= y0:
            continue
        items.append(LayoutItem(type=it["type"], label=it["label"], md=it["md"], bbox=Box(x0=x0, y0=y0, x1=x1, y1=y1)))
    return items

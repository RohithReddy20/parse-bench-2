"""KDL-Frontier-Parser-nano-v2 layout stage: build the call for a rendered page, read its regions.

The model (a 1.2B Qwen2-VL, open weights, no remote code) is asked "Layout Detection:" for the whole page and
answers one token string per region: `<|box_start|>x0 y0 x1 y1<|box_end|><|ref_start|>label<|ref_end|>`, with
coordinates on a 0-1000 grid of the page. Settings follow the `kdl_frontier_nano` provider in the ParseBench
harness: the page is resized to 1036 x 1036 and sent as JPEG quality 95, greedy decoding, up to 6,000 new
tokens, `no_repeat_ngram_size` 100. The raw answer is cached; reading it into regions happens here, so a
change in how regions are read never needs a model call.

Region reading follows the harness's `normalize_native_layout_items`: a `list` or `image_block` that holds
child regions is a container and is dropped (its children stay), and formula parts inside an `equation_block`
are dropped in favour of the block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docparse.core.hashing import sha256_hex
from docparse.models.adapters.heron import Detection
from docparse.models.calls import ModelCall, ModelRef, ModelResult

KDL = ModelRef(
    repo="KDLAI/KDL-Frontier-Parser-nano-v2", revision="5b5e0edda3c019563a2f6205ec49095288fafeb8", dtype="bf16"
)
RUNTIME = "transformers"
PROMPT_ID = "kdl-nano/layout"
PROMPT = "\nLayout Detection:\n"
PARAMS = {"size": 1036, "jpeg_quality": 95, "max_new_tokens": 6000, "no_repeat_ngram_size": 100}

CATEGORY = {
    "algorithm": "Text", "aside_text": "Text", "chart": "Chart", "code": "Text", "code_caption": "Caption",
    "equation": "Formula", "equation_block": "Formula", "footer": "Page-footer", "header": "Page-header",
    "image": "Picture", "image_block": "Picture", "image_caption": "Caption", "image_footnote": "Footnote",
    "inline_formula": "Formula", "list": "List-item", "list_item": "List-item", "page_footnote": "Footnote",
    "page_number": "Page-footer", "phonetic": "Text", "ref_text": "List-item", "table": "Table",
    "table_caption": "Caption", "table_footnote": "Footnote", "text": "Text", "title": "Title", "unknown": "Text",
}  # fmt: skip
_LIST_CHILDREN = frozenset({"text", "list_item", "ref_text"})
_IMAGE_CHILDREN = frozenset({"chart", "image", "image_caption", "image_footnote"})
_EQUATION_PARTS = frozenset({"equation", "inline_formula"})
_REGION = re.compile(
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>([^<]+?)<\|ref_end\|>", re.DOTALL
)


@dataclass(frozen=True)
class Region:
    raw: str  # the model's own label ("text", "image_block", ...)
    grid: tuple[int, int, int, int]  # x0, y0, x1, y1 on the 0-1000 grid
    in_list: bool = False  # inside a `list` container

    @property
    def label(self) -> str:
        """Docling class name, as the harness's provider maps it."""
        if self.raw in ("list_item", "ref_text") or (self.in_list and self.raw in _LIST_CHILDREN):
            return "List-item"
        return CATEGORY.get(self.raw, "Text")


def layout_call(page_hash: str, model: ModelRef = KDL) -> ModelCall:
    """The layout call for the page image whose `pixel_hash` is `page_hash` (its PNG is the input blob)."""
    return ModelCall(
        model=model,
        runtime=RUNTIME,
        prompt_id=PROMPT_ID,
        prompt_sha=sha256_hex(PROMPT.encode("utf-8")),
        params=dict(PARAMS),
        inputs=[page_hash],
    )


def parse_regions(text: str) -> list[Region]:
    out = []
    for m in _REGION.finditer(text or ""):
        x0, y0, x1, y1 = (int(m.group(i)) for i in range(1, 5))
        if max(x0, y0, x1, y1) > 1000:
            continue
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        if x1 > x0 and y1 > y0:
            out.append(Region(raw=m.group(5).strip().lower(), grid=(x0, y0, x1, y1)))
    return out


def _area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _contains(parent, child, share: float = 0.85) -> bool:
    inter = (max(parent[0], child[0]), max(parent[1], child[1]), min(parent[2], child[2]), min(parent[3], child[3]))
    return _area(child) > 0 and _area(inter) / _area(child) >= share


def read_regions(result: ModelResult) -> list[Region]:
    """The answer's regions in reading order, containers and formula parts removed."""
    regions = parse_regions(result.output.get("text") or "")

    def holds(parent: Region, kinds: frozenset) -> bool:
        return any(c is not parent and c.raw in kinds and _contains(parent.grid, c.grid) for c in regions)

    lists = [r for r in regions if r.raw == "list" and holds(r, _LIST_CHILDREN)]
    drop = {id(r) for r in lists}
    drop |= {id(r) for r in regions if r.raw == "image_block" and holds(r, _IMAGE_CHILDREN)}
    blocks = [r for r in regions if r.raw == "equation_block"]
    drop |= {id(c) for c in regions if c.raw in _EQUATION_PARTS and any(_contains(b.grid, c.grid) for b in blocks)}
    out = []
    for r in regions:
        if id(r) in drop:
            continue
        in_list = r.raw in _LIST_CHILDREN and any(_contains(p.grid, r.grid) for p in lists)
        out.append(Region(raw=r.raw, grid=r.grid, in_list=True) if in_list else r)
    return out


def read_detections(result: ModelResult) -> list[Detection]:
    """Regions as detections in pixels of the page image the call was made for (the model gives no score)."""
    w, h = float(result.output.get("width") or 0), float(result.output.get("height") or 0)
    out = []
    for r in read_regions(result):
        x0, y0, x1, y1 = r.grid
        box = (x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000)
        out.append(Detection(label=r.label, score=1.0, box=box))
    return out

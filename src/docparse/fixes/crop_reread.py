"""Layout-guided cropped high-res region re-reading and band stitching (general visual capability upgrade).

Uses cached `heron` layout detections to give hard regions (looped pages, dense/complex tables,
multi-column bodies, degraded scans, and charts) 2x-4x more visual tokens instead of relying on
one downscaled full-page VLM pass.

All model calls go through the same content-addressed `ModelClient` (batch manifests & live calls
share one code path).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from docparse.models.adapters.heron import Detection

CROP_OPTIONS = frozenset(
    {
        "loop_bands",  # split looped/truncated pages into overlapping horizontal bands and re-read
        "tables",  # crop detected table regions at high res and re-read
        "tables_min_area",  # min fraction of page area for table crop (default 0.03)
        "tables_max_area",  # max fraction of page area for table crop (default 0.92)
        "columns",  # crop clean 2-column bodies into left/right columns for higher res + order
        "scan_enhance",  # deskew + gentle contrast/background normalization on degraded scans
        "scan_bands",  # split degraded scans into 2 overlapping zoomed horizontal bands
        "charts_crop",  # crop chart pictures (+ caption/axis margin) at high res
        "target_long_side",  # upscale small crops toward this long side in px (default 1800)
        "max_upscale",  # maximum upscale factor for small crops (default 2.0)
    }
)


@dataclass(frozen=True)
class CropBox:
    """Pixel-space crop box `(x0, y0, x1, y1)` with semantic kind."""

    kind: str  # "table" | "chart" | "band" | "column"
    box: tuple[int, int, int, int]
    label: str = ""
    score: float = 1.0


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _merge_boxes(
    boxes: list[tuple[float, float, float, float]], iou_thr: float = 0.25
) -> list[tuple[float, float, float, float]]:
    """Merge overlapping or nested detections into clean outer boxes, sorted top-to-bottom."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[tuple[float, float, float, float]] = []
        used = [False] * len(merged)
        for i, a in enumerate(merged):
            if used[i]:
                continue
            cur = a
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                b = merged[j]
                # Merge on IoU or strong containment (one box mostly inside another)
                ix0, iy0 = max(cur[0], b[0]), max(cur[1], b[1])
                ix1, iy1 = min(cur[2], b[2]), min(cur[3], b[3])
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                min_area = min(
                    max(1.0, (cur[2] - cur[0]) * (cur[3] - cur[1])),
                    max(1.0, (b[2] - b[0]) * (b[3] - b[1])),
                )
                if _iou(cur, b) >= iou_thr or (inter / min_area) >= 0.60:
                    cur = (min(cur[0], b[0]), min(cur[1], b[1]), max(cur[2], b[2]), max(cur[3], b[3]))
                    used[j] = True
                    changed = True
            out.append(cur)
        merged = out
    return sorted(merged, key=lambda b: (b[1], b[0]))


def _clamp_int(box: tuple[float, float, float, float], w: int, h: int, pad: int = 0) -> tuple[int, int, int, int]:
    return (
        max(0, int(box[0]) - pad),
        max(0, int(box[1]) - pad),
        min(w, int(box[2]) + pad),
        min(h, int(box[3]) + pad),
    )


def table_crops(
    detections: list[Detection],
    w: int,
    h: int,
    min_area: float = 0.03,
    max_area: float = 0.92,
    min_score: float = 0.45,
) -> list[CropBox]:
    """Select tight, merged table crop boxes (expanding slightly to keep an adjacent caption/title)."""
    page_area = float(max(1, w * h))
    raw = [d.box for d in detections if d.label.lower() == "table" and d.score >= min_score]
    if not raw:
        return []
    merged = _merge_boxes(raw)
    captions = [d.box for d in detections if d.label.lower() in ("caption", "section_header") and d.score >= 0.35]
    crops: list[CropBox] = []
    for box in merged:
        area = ((box[2] - box[0]) * (box[3] - box[1])) / page_area
        if not (min_area <= area <= max_area):
            continue
        x0, y0, x1, y1 = box
        # Attach a caption sitting directly above the table within 3% of page height
        for cap in captions:
            horiz = min(x1, cap[2]) - max(x0, cap[0])
            if horiz > 0.3 * min(x1 - x0, cap[2] - cap[0]) and 0 <= y0 - cap[3] <= 0.035 * h:
                x0, y0, x1 = min(x0, cap[0]), min(y0, cap[1]), max(x1, cap[2])
        crops.append(CropBox(kind="table", box=_clamp_int((x0, y0, x1, y1), w, h, pad=24), label="table"))
    return crops


def chart_crops(
    detections: list[Detection],
    w: int,
    h: int,
    min_area: float = 0.04,
    min_score: float = 0.40,
) -> list[CropBox]:
    """Select chart/figure `picture` boxes expanded to include nearby titles, captions, axes, and legends."""
    page_area = float(max(1, w * h))
    raw = [d.box for d in detections if d.label.lower() == "picture" and d.score >= min_score]
    if not raw:
        return []
    merged = _merge_boxes(raw, iou_thr=0.15)
    captions = [
        d.box for d in detections if d.label.lower() in ("caption", "section_header", "title") and d.score >= 0.35
    ]
    crops: list[CropBox] = []
    for box in merged:
        bw, bh = box[2] - box[0], box[3] - box[1]
        area = (bw * bh) / page_area
        if area < min_area:
            continue
        # Expand around the picture so axis labels, tick values, and legends outside the plot box stay inside
        mx, my = max(24.0, 0.08 * bw), max(24.0, 0.08 * bh)
        x0, y0, x1, y1 = box[0] - mx, box[1] - my, box[2] + mx, box[3] + my
        for cap in captions:
            horiz = min(x1, cap[2]) - max(x0, cap[0])
            if horiz > 0.25 * min(bw, max(1.0, cap[2] - cap[0])):
                # Caption directly above or directly below within 5% page height
                if -0.02 * h <= box[1] - cap[3] <= 0.05 * h or -0.02 * h <= cap[1] - box[3] <= 0.05 * h:
                    x0, y0 = min(x0, cap[0] - 12), min(y0, cap[1] - 12)
                    x1, y1 = max(x1, cap[2] + 12), max(y1, cap[3] + 12)
        crops.append(CropBox(kind="chart", box=_clamp_int((x0, y0, x1, y1), w, h, pad=8), label="chart"))
    return crops


def column_crops(detections: list[Detection], w: int, h: int, min_score: float = 0.35) -> list[CropBox]:
    """Detect a crisp 2-column text layout from heron boxes and return ordered crops (top, left, right, bottom).
    Returns [] when the page is not a clear 2-column layout."""
    body_labels = {"text", "list_item", "section_header", "formula", "footnote"}
    boxes = [d.box for d in detections if d.label.lower() in body_labels and d.score >= min_score]
    if len(boxes) < 6:
        return []
    narrow = [b for b in boxes if (b[2] - b[0]) < 0.52 * w]
    wide = [b for b in boxes if (b[2] - b[0]) >= 0.56 * w]
    if len(narrow) < 6:
        return []

    # Try vertical gutter near center (42%..58% of page width)
    best_gutter = None
    best_score = -1
    for frac in (0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56):
        gx = frac * w
        left = [b for b in narrow if b[2] <= gx + 0.02 * w]
        right = [b for b in narrow if b[0] >= gx - 0.02 * w]
        crossing = [b for b in narrow if b[0] < gx - 0.02 * w and b[2] > gx + 0.02 * w]
        if len(left) >= 3 and len(right) >= 3 and len(crossing) <= 1:
            l_span = max(b[3] for b in left) - min(b[1] for b in left)
            r_span = max(b[3] for b in right) - min(b[1] for b in right)
            if l_span >= 0.25 * h and r_span >= 0.25 * h:
                score = len(left) + len(right) - 5 * len(crossing)
                if score > best_score:
                    best_score, best_gutter = score, (gx, left, right)
    if best_gutter is None:
        return []

    gx, left, right = best_gutter
    col_top = min(min(b[1] for b in left), min(b[1] for b in right))
    col_bot = max(max(b[3] for b in left), max(b[3] for b in right))

    # Do not split if a wide table or wide text sits INSIDE the column vertical span (complex interleaved layout)
    tables = [d.box for d in detections if d.label.lower() == "table" and d.score >= 0.45]
    for b in wide + tables:
        if (b[2] - b[0]) >= 0.56 * w and b[1] > col_top + 0.05 * h and b[3] < col_bot - 0.05 * h:
            return []

    crops: list[CropBox] = []
    top_boxes = [b for b in wide if b[3] <= col_top + 0.03 * h]
    if top_boxes:
        tb = (
            min(b[0] for b in top_boxes),
            min(b[1] for b in top_boxes),
            max(b[2] for b in top_boxes),
            max(b[3] for b in top_boxes),
        )
        crops.append(CropBox(kind="column", box=_clamp_int(tb, w, h, pad=20), label="top"))

    lb = (
        min(b[0] for b in left),
        min(b[1] for b in left),
        min(gx + 0.015 * w, max(b[2] for b in left) + 12),
        max(b[3] for b in left),
    )
    rb = (
        max(gx - 0.015 * w, min(b[0] for b in right) - 12),
        min(b[1] for b in right),
        max(b[2] for b in right),
        max(b[3] for b in right),
    )
    crops.append(CropBox(kind="column", box=_clamp_int(lb, w, h, pad=16), label="left"))
    crops.append(CropBox(kind="column", box=_clamp_int(rb, w, h, pad=16), label="right"))

    bot_boxes = [b for b in wide if b[1] >= col_bot - 0.03 * h]
    if bot_boxes:
        bb = (
            min(b[0] for b in bot_boxes),
            min(b[1] for b in bot_boxes),
            max(b[2] for b in bot_boxes),
            max(b[3] for b in bot_boxes),
        )
        crops.append(CropBox(kind="column", box=_clamp_int(bb, w, h, pad=20), label="bottom"))
    return crops


def render_crop(
    image, box: tuple[int, int, int, int], target_long_side: int = 1800, max_upscale: float = 2.0, border: int = 24
):
    """Crop `image` to `box`, upscale cleanly if smaller than `target_long_side`, and add white padding."""
    from PIL import Image, ImageOps

    crop = image.crop(box).convert("RGB")
    cw, ch = crop.size
    long_side = max(1, max(cw, ch))
    if long_side < target_long_side and max_upscale > 1.0:
        scale = min(max_upscale, target_long_side / long_side)
        if scale > 1.05:
            crop = crop.resize((max(1, int(cw * scale)), max(1, int(ch * scale))), Image.LANCZOS)
    if border > 0:
        crop = ImageOps.expand(crop, border=border, fill=(255, 255, 255))
    return crop


_NORM_LINE = re.compile(r"[^\w]+", re.UNICODE)


def _line_key(line: str) -> str:
    return _NORM_LINE.sub("", line.lower())


def stitch_bands(parts: list[str]) -> str:
    """Concatenate overlapping horizontal band readings, deduplicating identical/near-identical boundary lines."""
    clean = [p.strip() for p in parts if p and p.strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    acc_lines = clean[0].splitlines()
    for nxt in clean[1:]:
        nxt_lines = nxt.splitlines()
        # Look for matching anchor lines between the last 10 non-empty lines of acc and first 10 of nxt
        tail_nonempty = [
            (i, _line_key(ln))
            for i, ln in enumerate(acc_lines[-12:], max(0, len(acc_lines) - 12))
            if len(_line_key(ln)) >= 8
        ]
        head_nonempty = [(j, _line_key(ln)) for j, ln in enumerate(nxt_lines[:12]) if len(_line_key(ln)) >= 8]
        drop_head = 0
        for _i, tk in reversed(tail_nonempty):
            for j, hk in head_nonempty:
                if tk == hk or (min(len(tk), len(hk)) >= 14 and (tk in hk or hk in tk)):
                    # Drop overlap in nxt up through line j
                    drop_head = j + 1
                    break
            if drop_head:
                break
        remainder = nxt_lines[drop_head:]
        if remainder:
            acc_lines.append("")
            acc_lines.extend(remainder)
    return "\n".join(acc_lines)


def _table_tokens(raw_html: str) -> set[str]:
    text = re.sub(r"<[^>]+>", " ", raw_html)
    return {t for t in re.findall(r"[a-z0-9]{2,}", text.lower())}


def match_cropped_table(crop_table_html: str, page_tables: list[tuple[str, str]]) -> str | None:
    """Return the region_id of the full-page table best matching `crop_table_html`, or None."""
    if not page_tables:
        return None
    ctoks = _table_tokens(crop_table_html)
    if not ctoks:
        return None
    best_id, best_score = None, 0.0
    for rid, raw in page_tables:
        ptoks = _table_tokens(raw)
        if not ptoks:
            continue
        inter = len(ctoks & ptoks)
        overlap = inter / max(1, min(len(ctoks), len(ptoks)))
        jaccard = inter / max(1, len(ctoks | ptoks))
        score = 0.6 * overlap + 0.4 * jaccard
        if score > best_score:
            best_id, best_score = rid, score
    return best_id if best_score >= 0.35 else None


def strip_furniture_markdown(
    markdown: str, detections: list[Detection], words: list[Any] | None, w: int, h: int
) -> str:
    """Remove lines matching `page_header` / `page_footer` heron boxes (works on both digital and scanned pages)."""
    hf_boxes = [d.box for d in detections if d.label.lower() in ("page_header", "page_footer") and d.score >= 0.35]
    if not hf_boxes:
        return markdown
    # Collect normalized strings from PDF words inside header/footer boxes (when digital)
    hf_keys: set[str] = set()
    if words:
        for b in hf_boxes:
            box_words = [
                wd.text
                for wd in words
                if (wd.box[0] + wd.box[2]) / 2 >= b[0] - 4
                and (wd.box[0] + wd.box[2]) / 2 <= b[2] + 4
                and (wd.box[1] + wd.box[3]) / 2 >= b[1] - 4
                and (wd.box[1] + wd.box[3]) / 2 <= b[3] + 4
            ]
            if box_words:
                k = _line_key(" ".join(box_words))
                if k:
                    hf_keys.add(k)
    lines = markdown.splitlines()
    # Allow stripping bare top/bottom page-number lines when heron detected header/footer there
    has_top_hf = any(b[1] < 0.15 * h for b in hf_boxes)
    has_bot_hf = any(b[3] > 0.85 * h for b in hf_boxes)
    nonempty_idx = [i for i, ln in enumerate(lines) if ln.strip()]
    if not nonempty_idx:
        return markdown
    top_edge = set(nonempty_idx[:2])
    bot_edge = set(nonempty_idx[-2:])
    out = []
    for i, ln in enumerate(lines):
        k = _line_key(ln)
        if not k:
            out.append(ln)
            continue
        if k in hf_keys and (i in top_edge or i in bot_edge or len(k) >= 6):
            continue
        if not words and ((has_top_hf and i in top_edge) or (has_bot_hf and i in bot_edge)):
            # On scans without text words, only strip short standalone page-number lines (< 80 chars)
            s = ln.strip()
            if (
                len(s) <= 80
                and not s.startswith("<table")
                and (s.isdigit() or re.fullmatch(r"[-—–\s]*\d{1,4}[-—–\s]*", s))
            ):
                continue
        out.append(ln)
    return "\n".join(out).strip() + ("\n" if markdown.endswith("\n") else "")

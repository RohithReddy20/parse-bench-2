"""Compare two runs on a fixed set of pages: used for the pilot, where only a sample is scored.

`gt_view` builds a ground-truth folder holding only the chosen cases (symlinks), so the harness scores just
those pages. `compare_runs` puts our page scores next to a baseline run's scores for the same pages.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

from evalkit.explode import split_example_id
from evalkit.ledger import Ledger
from evalkit.losses import slot_weights


def gt_view(gt_dir: Path, example_ids: list[str], dest: Path) -> Path:
    """`dest/<subset>/` with links to every file of the chosen cases. Nothing is copied or opened."""
    for example_id in example_ids:
        subset, stem = split_example_id(example_id)
        (dest / subset).mkdir(parents=True, exist_ok=True)
        for src in (gt_dir / subset).glob("*"):
            if src.name.startswith(stem + ".") and src.name[len(stem) :] in (
                ".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".test.json",
            ):  # fmt: skip
                link = dest / subset / src.name
                if not link.exists():
                    link.symlink_to(src)
    return dest


def compare_runs(ledger: Ledger, run_id: str, baseline_id: str, example_ids: list[str]) -> dict:
    """Per (subset, metric) and per dimension: mean score of both runs on `example_ids`."""
    wanted = set(example_ids)
    ours = {(c.example_id, c.metric): c for c in ledger.cases(run_id) if c.example_id in wanted and c.counted}
    base = {(c.example_id, c.metric): c for c in ledger.cases(baseline_id) if c.example_id in wanted and c.counted}
    keys = sorted(set(ours) & set(base))
    slots: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    pages = []
    for key in keys:
        a, b = ours[key].score or 0.0, base[key].score or 0.0
        slots[(ours[key].subset, key[1])].append((a, b))
        pages.append({"example_id": key[0], "metric": key[1], "ours": a, "baseline": b, "delta": a - b})
    weights = slot_weights()
    rows = []
    dims: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for (subset, metric), pairs in sorted(slots.items()):
        a = sum(p[0] for p in pairs) / len(pairs)
        b = sum(p[1] for p in pairs) / len(pairs)
        dim, w = weights[(subset, metric)]
        rows.append({"dimension": dim, "subset": subset, "metric": metric, "pages": len(pairs),
                     "ours": 100 * a, "baseline": 100 * b, "delta": 100 * (a - b)})  # fmt: skip
        dims[dim][0] += w * a
        dims[dim][1] += w * b
        dims[dim][2] += w
    dimensions = {d: {"ours": 100 * v[0] / v[2], "baseline": 100 * v[1] / v[2]} for d, v in dims.items()}
    return {"slots": rows, "dimensions": dimensions, "pages": sorted(pages, key=lambda p: p["delta"]),
            "only_ours": len(set(ours) - set(base)), "only_baseline": len(set(base) - set(ours))}  # fmt: skip


_WS = re.compile(r"\s+")


def markdown_similarity(a: str, b: str) -> float:
    """0..1 similarity of two page readings on whitespace-normalised words."""
    wa, wb = _WS.sub(" ", a).split(), _WS.sub(" ", b).split()
    if not wa and not wb:
        return 1.0
    return difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()


def read_markdown(run_output_root: Path, pipeline: str, example_id: str) -> str | None:
    subset, stem = split_example_id(example_id)
    path = run_output_root / subset / pipeline / f"{stem}.result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["output"]["markdown"]

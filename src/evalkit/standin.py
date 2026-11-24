"""Import OCRia-1's published outputs into the model-call cache as stand-in page readings.

Each published page is stored under the key of the call our pipeline makes for the same rendered page, with
the `PUBLISHED` model ref. The pipeline then runs end to end on all pages with a ReplayClient and no GPU.
Published records carry post-processed markdown only (their `raw_output` is empty).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from docparse.core.errors import ParseError
from docparse.models.adapters import ocria
from docparse.models.cache import Cache
from docparse.models.calls import ModelCall, ModelResult
from docparse.pdf.render import pixel_hash, render_pages
from evalkit.explode import SUBSETS, read_records

SOURCE_SUFFIXES = (".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass
class StandinReport:
    imported: int = 0
    cached: int = 0
    collisions: list[tuple[str, str]] = field(default_factory=list)  # same pixels, different published markdown
    missing_source: list[str] = field(default_factory=list)
    page_mismatch: list[str] = field(default_factory=list)  # published page count != rendered page count
    unreadable: list[str] = field(default_factory=list)


def source_file(gt_dir: Path, subset: str, stem: str) -> Path | None:
    for suffix in SOURCE_SUFFIXES:
        path = gt_dir / subset / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def published_call(page_hash: str) -> ModelCall:
    return ModelCall(
        model=ocria.PUBLISHED,
        runtime=ocria.PUBLISHED_RUNTIME,
        prompt_id=ocria.PROMPT_ID,
        prompt_sha="published",
        params=dict(ocria.PARAMS),
        inputs=[page_hash],
    )


def _hash_pages(path: str) -> list[str] | None:
    try:
        return [pixel_hash(p.image) for p in render_pages(Path(path).read_bytes())]
    except ParseError:
        return None


def import_published(published_dir: Path, gt_dir: Path, cache: Cache, workers: int = 8) -> StandinReport:
    report = StandinReport()
    jobs: list[tuple[str, Path, dict]] = []
    for subset in SUBSETS:
        path = published_dir / f"{subset}.jsonl"
        if not path.exists():
            continue
        for _number, _subset, stem, record in read_records(path):
            example_id = f"{subset}/{stem}"
            src = source_file(gt_dir, subset, stem)
            if src is None:
                report.missing_source.append(example_id)
            else:
                jobs.append((example_id, src, record))
    paths = [str(src) for _id, src, _rec in jobs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            hashes = list(pool.map(_hash_pages, paths, chunksize=8))
    else:
        hashes = [_hash_pages(p) for p in paths]
    owner: dict[str, tuple[str, str]] = {}  # call key -> (example id, markdown)
    for (example_id, _src, record), page_hashes in zip(jobs, hashes, strict=True):
        if page_hashes is None:
            report.unreadable.append(example_id)
            continue
        output = record.get("output") or {}
        pages = output.get("pages") or []
        if len(pages) != len(page_hashes):
            report.page_mismatch.append(example_id)
        layout_by_number = {lp.get("page_number"): lp for lp in output.get("layout_pages") or []}
        for i, page_hash in enumerate(page_hashes):
            markdown = pages[i].get("markdown", "") if i < len(pages) else ""
            if len(page_hashes) == 1 and len(pages) != 1:
                markdown = output.get("markdown", "")
            call = published_call(page_hash)
            key = call.key()
            if key in owner:
                if owner[key][1] != markdown:
                    report.collisions.append((owner[key][0], example_id))
                continue
            owner[key] = (example_id, markdown)
            body = {"markdown": markdown, "layout_page": layout_by_number.get(i + 1), "source": "published"}
            stored = cache.put(ModelResult(key=key, output=body), call, {"example_id": example_id, "page": i})
            if stored:
                report.imported += 1
            else:
                report.cached += 1
    return report


def report_json(report: StandinReport) -> str:
    return json.dumps(report.__dict__, indent=2)

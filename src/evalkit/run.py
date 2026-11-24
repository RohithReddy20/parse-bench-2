"""Run the parser over benchmark cases and write result records in the layout the evaluator reads.

This is the only place where example ids and files meet: the parser gets the bytes, and the record gets the
example id afterwards. Parsing runs with answer keys forbidden.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from docparse.bench.record import result_record
from docparse.core.config import Config
from docparse.core.errors import ParseError
from docparse.models.cache import Cache
from docparse.models.calls import CacheMiss, ModelCall, Pending
from docparse.models.client import ReplayClient
from docparse.models.manifest import ManifestWriter, _check_hash
from docparse.pipeline import parse
from docparse.write.lint import lint_page
from evalkit.explode import SUBSETS, metadata_for, split_example_id
from evalkit.guards import forbid_answer_keys
from evalkit.standin import source_file


@dataclass
class RunSummary:
    written: int = 0
    pending: int = 0  # documents waiting on model calls
    new_calls: int = 0  # calls added to the manifest
    errors: list[tuple[str, str]] = field(default_factory=list)
    lint: Counter = field(default_factory=Counter)


def list_examples(gt_dir: Path, subsets: tuple[str, ...] | None = None) -> list[str]:
    """Every `<subset>/<stem>` that has a test case. File names only; no answer key is opened."""
    out = []
    for subset in subsets or SUBSETS:
        out += sorted(f"{subset}/{p.name.removesuffix('.test.json')}" for p in (gt_dir / subset).glob("*.test.json"))
    return out


class _PlanningClient:
    """Cache first; a miss stores its blob in the manifest folder and hands the call back to the parent."""

    def __init__(self, cache: Cache, manifest_dir: Path):
        self.cache, self.dir, self.misses = cache, manifest_dir, []

    def get(self, call: ModelCall, blobs, meta=None):
        hit = self.cache.get(call.key())
        if hit is not None:
            return hit
        for name in call.inputs:
            path = self.dir / "blobs" / _check_hash(name)
            if not path.exists():
                tmp = path.with_name(f".{name}.{id(self)}.tmp")
                tmp.write_bytes(blobs[name])
                tmp.replace(path)
        self.misses.append((call.model_dump(mode="json"), meta or {}))
        return Pending(key=call.key())


def _one(job: tuple[str, str, str, str, str, str, str | None]) -> dict:
    example_id, src, config_json, cache_root, out_root, pipeline_name, manifest_dir = job
    subset, stem = split_example_id(example_id)
    config = Config.model_validate_json(config_json)
    cache = Cache(Path(cache_root))
    client = _PlanningClient(cache, Path(manifest_dir)) if manifest_dir else ReplayClient(cache)
    data = Path(src).read_bytes()
    try:
        with forbid_answer_keys():
            result = parse(data, config, client)
    except (ParseError, CacheMiss) as exc:
        return {"id": example_id, "error": f"{type(exc).__name__}: {exc}"}
    if result.document is None:
        return {"id": example_id, "pending": True, "misses": client.misses}
    record = result_record(example_id, pipeline_name, result.document)
    target = Path(out_root) / subset / pipeline_name / f"{stem}.result.json"
    target.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    lint = Counter()
    layouts = {lp["page_number"]: lp for lp in record["output"]["layout_pages"]}
    for page in record["output"]["pages"]:
        number = page["page_index"] + 1
        lint.update(f.check for f in lint_page(page["markdown"], layouts.get(number), number))
    return {"id": example_id, "lint": dict(lint)}


def run_pipeline(
    config: Config,
    pipeline_name: str,
    out_root: Path,
    gt_dir: Path,
    cache_root: Path,
    example_ids: list[str] | None = None,
    manifest_dir: Path | None = None,
    workers: int = 8,
) -> RunSummary:
    """Parse every case and write `<out_root>/<subset>/<pipeline_name>/<stem>.result.json`.

    With `manifest_dir`, cache misses become manifest lines (for a GPU session) instead of errors.
    """
    ids = example_ids if example_ids is not None else list_examples(gt_dir)
    summary = RunSummary()
    jobs = []
    for example_id in ids:
        subset, stem = split_example_id(example_id)
        src = source_file(gt_dir, subset, stem)
        if src is None:
            summary.errors.append((example_id, "no source file"))
            continue
        (out_root / subset / pipeline_name).mkdir(parents=True, exist_ok=True)
        jobs.append((example_id, str(src), config.model_dump_json(), str(cache_root), str(out_root), pipeline_name,
                     str(manifest_dir) if manifest_dir else None))  # fmt: skip
    writer = ManifestWriter(manifest_dir) if manifest_dir else None
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, jobs, chunksize=4))
    else:
        results = [_one(job) for job in jobs]
    for res in results:
        if "error" in res:
            summary.errors.append((res["id"], res["error"]))
        elif res.get("pending"):
            summary.pending += 1
            for call_json, meta in res["misses"]:
                summary.new_calls += writer.add(ModelCall.model_validate(call_json), {}, meta)
        else:
            summary.written += 1
            summary.lint.update(res["lint"])
    for subset in {split_example_id(i)[0] for i in ids}:
        folder = out_root / subset / pipeline_name
        if folder.is_dir():
            (folder / "_metadata.json").write_text(json.dumps(metadata_for(pipeline_name, "docparse"), indent=2))
    return summary

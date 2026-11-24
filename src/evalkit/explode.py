"""Published results JSONL -> the per-document layout the harness evaluator reads.

Our own code: the results repo ships `tools/explode.py`, which is downloaded data and is never run. Layout
(ParseBench 2): `<output_root>/<subset>/<pipeline>/<stem>.result.json` plus a `_metadata.json` per folder.
Records are untrusted, so every `example_id` is validated before a path is built from it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from parse_bench.evaluation.next_score import NEXT_SUBSETS

SUBSETS: tuple[str, ...] = tuple(NEXT_SUBSETS)
MAX_LINE_BYTES = 64 * 1024 * 1024
_PIPELINE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def metadata_for(pipeline: str, source: str = "published results") -> dict:
    return {
        "pipeline": {
            "pipeline_name": pipeline,
            "provider_name": pipeline,
            "product_type": "parse",
            "config": {},
        },
        "source": source,
    }


def split_example_id(example_id: object) -> tuple[str, str]:
    """`<subset>/<stem>` -> (subset, stem); raises ValueError on anything that could leave the output root."""
    if not isinstance(example_id, str):
        raise ValueError(f"example_id is not a string: {example_id!r}")
    subset, sep, stem = example_id.partition("/")
    if not sep or subset not in SUBSETS:
        raise ValueError(f"example_id must be '<subset>/<stem>' with a known subset: {example_id!r}")
    if not stem or "/" in stem or "\\" in stem or "\x00" in stem or stem in (".", "..") or stem.startswith("."):
        raise ValueError(f"unsafe stem in example_id: {example_id!r}")
    return subset, stem


def read_records(path: Path):
    """(line number, record) for every non-blank line; ValueError names the line on any malformed record."""
    with path.open("rb") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            if len(line) > MAX_LINE_BYTES:
                raise ValueError(f"{path.name}:{number}: line of {len(line)} bytes exceeds the limit")
            try:
                record = json.loads(line)
                example_id = record["request"]["example_id"]
                subset, stem = split_example_id(example_id)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path.name}:{number}: {exc}") from exc
            if subset != path.stem:
                raise ValueError(f"{path.name}:{number}: record of subset {subset!r} in the wrong file")
            yield number, subset, stem, record


def explode(results_dir: Path, output_root: Path, pipeline: str) -> dict[str, int]:
    """Write every record of `results_dir/<subset>.jsonl`; returns documents written per subset.

    All files are validated before anything is written, so a bad record leaves the output root untouched.
    """
    if not _PIPELINE.fullmatch(pipeline):
        raise ValueError(f"not a pipeline name: {pipeline!r}")
    files = [results_dir / f"{s}.jsonl" for s in SUBSETS if (results_dir / f"{s}.jsonl").exists()]
    if not files:
        raise ValueError(f"no <subset>.jsonl files in {results_dir}")
    for path in files:  # validation pass
        seen: set[str] = set()
        for number, _subset, stem, _record in read_records(path):
            if stem in seen:
                raise ValueError(f"{path.name}:{number}: duplicate example {stem!r}")
            seen.add(stem)
    written: dict[str, int] = {}
    root = output_root.resolve()
    for path in files:
        folder = root / path.stem / pipeline
        folder.mkdir(parents=True, exist_ok=True)
        count = 0
        for _number, _subset, stem, record in read_records(path):
            target = folder / f"{stem}.result.json"
            if target.resolve().parent != folder:
                raise ValueError(f"{path.name}: {stem!r} resolves outside {folder}")
            target.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            count += 1
        (folder / "_metadata.json").write_text(json.dumps(metadata_for(pipeline), indent=2), encoding="utf-8")
        written[path.stem] = count
    return written

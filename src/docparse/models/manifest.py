"""Request manifests: the calls a batch GPU session has to run (architecture plan §1.5).

A manifest is a directory holding `manifest.jsonl` (one call per line, with its metadata) and `blobs/<hash>`
(each input once). `plan` turns it into the table shown to the owner before any GPU is started.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docparse.models.cache import Cache
from docparse.models.calls import ModelCall

MANIFEST = "manifest.jsonl"
_HEX = set("0123456789abcdef")


def _check_hash(name: str) -> str:
    if not name or len(name) > 128 or not set(name) <= _HEX:
        raise ValueError(f"not a blob hash: {name!r}")
    return name


class ManifestWriter:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        (self.dir / "blobs").mkdir(parents=True, exist_ok=True)
        self._keys = {call.key() for call, _meta in read_manifest(self.dir)}

    def add(self, call: ModelCall, blobs: dict[str, bytes], meta: dict[str, Any] | None = None) -> bool:
        """Record a call once. False when it is already in this manifest."""
        key = call.key()
        if key in self._keys:
            return False
        missing = [h for h in call.inputs if h not in blobs and not (self.dir / "blobs" / _check_hash(h)).exists()]
        if missing:
            raise ValueError(f"call {key[:12]} has no blob for input(s) {missing}")
        for name, data in blobs.items():
            path = self.dir / "blobs" / _check_hash(name)
            if not path.exists():
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
        with (self.dir / MANIFEST).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "call": call.model_dump(mode="json"), "meta": meta or {}}) + "\n")
        self._keys.add(key)
        return True

    def __len__(self) -> int:
        return len(self._keys)


def read_manifest(directory: Path) -> Iterator[tuple[ModelCall, dict[str, Any]]]:
    path = Path(directory) / MANIFEST
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                yield ModelCall.model_validate(row["call"]), row.get("meta") or {}


def load_blobs(directory: Path, call: ModelCall) -> dict[str, bytes]:
    return {h: (Path(directory) / "blobs" / _check_hash(h)).read_bytes() for h in call.inputs}


@dataclass(frozen=True)
class Rate:
    seconds_per_call: float  # measured or estimated GPU seconds per call
    usd_per_hour: float  # 0 for free compute


@dataclass(frozen=True)
class PlanRow:
    model: str
    new_calls: int
    cached_calls: int
    est_gpu_seconds: float
    est_usd: float


def plan(directory: Path, cache: Cache, rates: dict[str, Rate] | None = None) -> list[PlanRow]:
    """Per model: calls still to run, calls already cached, and the estimated GPU time and cost of the new ones."""
    new: dict[str, int] = {}
    cached: dict[str, int] = {}
    for call, _meta in read_manifest(directory):
        name = f"{call.model.repo}@{call.model.revision[:12]}"
        bucket = cached if cache.has(call.key()) else new
        bucket[name] = bucket.get(name, 0) + 1
    rows = []
    for name in sorted(set(new) | set(cached)):
        rate = (rates or {}).get(name.split("@")[0], Rate(0.0, 0.0))
        seconds = new.get(name, 0) * rate.seconds_per_call
        rows.append(PlanRow(name, new.get(name, 0), cached.get(name, 0), seconds, seconds / 3600 * rate.usd_per_hour))
    return rows


def format_plan(rows: list[PlanRow]) -> str:
    lines = [f"{'model':58}{'new':>7}{'cached':>8}{'GPU h':>8}{'USD':>8}"]
    for r in rows:
        lines.append(f"{r.model:58}{r.new_calls:7d}{r.cached_calls:8d}{r.est_gpu_seconds / 3600:8.2f}{r.est_usd:8.2f}")
    total_new = sum(r.new_calls for r in rows)
    total_usd = sum(r.est_usd for r in rows)
    lines.append(f"{'total':58}{total_new:7d}{sum(r.cached_calls for r in rows):8d}"
                 f"{sum(r.est_gpu_seconds for r in rows) / 3600:8.2f}{total_usd:8.2f}")  # fmt: skip
    return "\n".join(lines)

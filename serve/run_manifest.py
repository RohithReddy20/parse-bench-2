"""The one resumable manifest runner, used on Modal, Colab, Kaggle and locally.

    run_manifest(manifest_dir, results_path, executor)   # executes the calls not yet in results_path
    import_results(manifest_dir, results_path, cache)    # stores finished calls in the model-call cache

`results_path` is append-only JSONL, one line per finished call, flushed as it goes, so a killed session
loses at most the calls that were in flight. An executor error is recorded and the run continues; failed
calls are retried by the next run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from docparse.models.cache import Cache
from docparse.models.calls import ModelCall
from docparse.models.client import result_from_output
from docparse.models.manifest import load_blobs, read_manifest

Executor = Callable[[ModelCall, dict[str, bytes]], dict[str, Any]]


@dataclass
class RunStats:
    total: int = 0
    skipped: int = 0  # already finished in results_path
    done: int = 0
    failed: int = 0
    seconds: float = 0.0


def _read_results(results_path: Path) -> dict[str, dict]:
    """key -> last line for that key (a later success replaces an earlier failure). Torn last lines are ignored."""
    out: dict[str, dict] = {}
    if not results_path.exists():
        return out
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "key" in row:
                if row.get("ok") or row["key"] not in out or not out[row["key"]].get("ok"):
                    out[row["key"]] = row
    return out


def run_manifest(manifest_dir: Path, results_path: Path, executor: Executor, max_workers: int = 1) -> RunStats:
    stats = RunStats()
    finished = {k for k, row in _read_results(results_path).items() if row.get("ok")}
    todo = []
    for call, _meta in read_manifest(manifest_dir):
        stats.total += 1
        if call.key() in finished:
            stats.skipped += 1
        else:
            todo.append(call)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    started = time.monotonic()
    with results_path.open("a", encoding="utf-8") as fh:

        def one(call: ModelCall) -> None:
            t0 = time.monotonic()
            try:
                row = {"key": call.key(), "ok": True, "output": executor(call, load_blobs(manifest_dir, call))}
            except Exception as exc:  # one bad page must not stop the session
                row = {"key": call.key(), "ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
            row["seconds"] = round(time.monotonic() - t0, 3)
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if row["ok"]:
                    stats.done += 1
                else:
                    stats.failed += 1

        if max_workers <= 1:
            for call in todo:
                one(call)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(one, todo))
    stats.seconds = time.monotonic() - started
    return stats


def import_results(manifest_dir: Path, results_path: Path, cache: Cache) -> int:
    """Put every finished call of the manifest into the cache; returns how many were new."""
    rows = _read_results(results_path)
    added = 0
    for call, meta in read_manifest(manifest_dir):
        row = rows.get(call.key())
        if row and row.get("ok"):
            added += cache.put(result_from_output(call, row["output"]), call, meta)
    return added

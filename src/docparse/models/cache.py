"""The model-call cache: `calls/<aa>/<key>.json.zst` plus a SQLite index (architecture plan §1.6).

Nothing is ever deleted or overwritten: the first complete write of a key wins. Entries are written to a
temporary file and renamed, so an interrupted write is never readable. The index is a rebuildable view used
for cost reporting; the files are the truth.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import zstandard

from docparse.core.hashing import sha256_hex
from docparse.models.calls import ModelCall, ModelResult

_INDEX = """
CREATE TABLE IF NOT EXISTS calls (
    key TEXT PRIMARY KEY, model TEXT, revision TEXT, runtime TEXT, prompt_id TEXT,
    tokens_in INTEGER, tokens_out INTEGER, gpu_seconds REAL, gpu_type TEXT, created_at TEXT, meta TEXT
);
"""


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "calls").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.corrupt: list[str] = []  # keys whose file could not be read (reported, treated as absent)
        with self._index() as db:
            db.executescript(_INDEX)

    def _index(self) -> sqlite3.Connection:
        return sqlite3.connect(self.root / "index.sqlite", timeout=60)

    def _path(self, key: str) -> Path:
        if len(key) != 64 or not all(c in "0123456789abcdef" for c in key):
            raise ValueError(f"not a call key: {key!r}")
        return self.root / "calls" / key[:2] / f"{key}.json.zst"

    def get(self, key: str) -> ModelResult | None:
        path = self._path(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            result = ModelResult.model_validate_json(zstandard.ZstdDecompressor().decompress(data))
            if result.key != key:
                raise ValueError("key mismatch")
            return result
        except Exception:
            if key not in self.corrupt:
                self.corrupt.append(key)
            return None

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def put(self, result: ModelResult, call: ModelCall, meta: dict[str, Any] | None = None) -> bool:
        """Store `result` for `call`. False, with nothing changed, when the key is already present."""
        key = call.key()
        if result.key != key:
            raise ValueError("result.key does not match the call")
        path = self._path(key)
        with self._lock:
            if self.get(key) is not None:
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{key}.{uuid.uuid4().hex}.tmp")
            try:
                tmp.write_bytes(zstandard.ZstdCompressor(level=9).compress(result.model_dump_json().encode("utf-8")))
                if key in self.corrupt:  # replace an unreadable entry
                    os.replace(tmp, path)
                    self.corrupt.remove(key)
                else:
                    try:
                        os.link(tmp, path)  # atomic and fails if another process got there first
                    except FileExistsError:
                        return False
            finally:
                tmp.unlink(missing_ok=True)
            with self._index() as db:
                db.execute(
                    "INSERT OR IGNORE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        key, call.model.repo, call.model.revision, call.runtime, call.prompt_id, result.tokens_in,
                        result.tokens_out, result.gpu_seconds, result.gpu_type, result.created_at,
                        json.dumps(meta or {}, sort_keys=True),
                    ),
                )  # fmt: skip
        return True

    def keys(self) -> Iterator[str]:
        for path in sorted((self.root / "calls").glob("??/*.json.zst")):
            yield path.name.removesuffix(".json.zst")

    def snapshot_hash(self) -> str:
        """Identifies the set of cached calls a score was computed from."""
        return sha256_hex("\n".join(self.keys()).encode("ascii"))

    def usage(self, model_repo: str | None = None) -> dict[str, float]:
        sql = "SELECT COUNT(*), SUM(tokens_in), SUM(tokens_out), SUM(gpu_seconds) FROM calls"
        args: tuple = ()
        if model_repo is not None:
            sql += " WHERE model = ?"
            args = (model_repo,)
        with self._index() as db:
            n, tin, tout, gpu = db.execute(sql, args).fetchone()
        return {"calls": n or 0, "tokens_in": tin or 0, "tokens_out": tout or 0, "gpu_seconds": gpu or 0.0}

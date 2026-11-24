"""The three model clients (architecture plan §1.5). All read the cache first.

- CacheOnlyClient: a miss becomes a manifest line and `Pending`; nothing is spent. Batch planning.
- ReplayClient: a miss is an error. Every reported score is produced with this client.
- EndpointClient: a miss calls an executor (an online endpoint) and stores the answer. The viewer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from docparse.models.cache import Cache
from docparse.models.calls import CacheMiss, ModelCall, ModelResult, Pending
from docparse.models.manifest import ManifestWriter

Executor = Callable[[ModelCall, dict[str, bytes]], dict[str, Any]]


class CacheOnlyClient:
    def __init__(self, cache: Cache, manifest: ManifestWriter):
        self.cache, self.manifest = cache, manifest

    def get(self, call: ModelCall, blobs: dict[str, bytes], meta: dict[str, Any] | None = None):
        hit = self.cache.get(call.key())
        if hit is not None:
            return hit
        self.manifest.add(call, blobs, meta)
        return Pending(key=call.key())


class ReplayClient:
    def __init__(self, cache: Cache):
        self.cache = cache

    def get(self, call: ModelCall, blobs: dict[str, bytes], meta: dict[str, Any] | None = None):
        hit = self.cache.get(call.key())
        if hit is None:
            raise CacheMiss(call.key())
        return hit


def result_from_output(call: ModelCall, output: dict[str, Any]) -> ModelResult:
    """An executor's answer as a cache entry. Usage numbers travel in reserved `_`-prefixed keys."""
    usage = {k: output.get(k) for k in ("_tokens_in", "_tokens_out", "_gpu_seconds", "_gpu_type")}
    body = {k: v for k, v in output.items() if k not in usage}
    return ModelResult(
        key=call.key(),
        output=body,
        tokens_in=usage["_tokens_in"],
        tokens_out=usage["_tokens_out"],
        gpu_seconds=usage["_gpu_seconds"],
        gpu_type=usage["_gpu_type"],
    )


class EndpointClient:
    def __init__(self, cache: Cache, executor: Executor):
        self.cache, self.executor = cache, executor

    def get(self, call: ModelCall, blobs: dict[str, bytes], meta: dict[str, Any] | None = None):
        hit = self.cache.get(call.key())
        if hit is not None:
            return hit
        result = result_from_output(call, self.executor(call, blobs))
        self.cache.put(result, call, meta)
        return self.cache.get(call.key()) or result  # first write wins if another process raced us

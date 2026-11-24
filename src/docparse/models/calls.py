"""Model calls as content-addressed values (architecture plan §1.5).

A call's key is the hash of everything that can change its answer: model revision and dtype, runtime, prompt,
parameters (sample index and seed included) and the hashes of its inputs. Where a page came from (file, page
number, DPI) is metadata and never part of the key.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from docparse.core.hashing import canonical_json, sha256_hex


class ModelRef(BaseModel, frozen=True):
    repo: str  # Hugging Face repo id
    revision: str  # commit sha, or a named stand-in such as "published-497e243f68cd"
    dtype: str  # "bf16", "fp16", "fp32", ...


class ModelCall(BaseModel, frozen=True):
    model: ModelRef
    runtime: str  # "vllm-0.31.0", "transformers-5.22.2", "published"
    prompt_id: str  # which request template, e.g. "ocria-1/page"
    prompt_sha: str  # sha256 of the fully rendered request minus its inputs
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)  # hash of each input blob

    def key(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


class ModelResult(BaseModel):
    key: str
    output: dict[str, Any]
    tokens_in: int | None = None
    tokens_out: int | None = None
    gpu_seconds: float | None = None
    gpu_type: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class Pending(BaseModel, frozen=True):
    """The call is not in the cache; it was written to a manifest for a later GPU session."""

    key: str


class CacheMiss(LookupError):
    def __init__(self, key: str):
        super().__init__(f"model call {key} is not in the cache")
        self.key = key


class ModelClient(Protocol):
    def get(
        self, call: ModelCall, blobs: dict[str, bytes], meta: dict[str, Any] | None = None
    ) -> ModelResult | Pending: ...

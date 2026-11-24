"""OCRia-1 page reads: build the call for a rendered page, and read its result.

Two model refs share the code path. `OWN` is the real model at its pinned revision. `PUBLISHED` holds
LlamaIndex's published outputs for the same pages, imported as stand-in readings (design §2); its key differs,
so a stand-in can never be mistaken for, or block, a real answer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from docparse.base.postprocess import ocria_postprocess
from docparse.models.calls import ModelCall, ModelRef, ModelResult
from docparse.pdf.render import pixel_hash, png_bytes

REPO = "llamaindex/OCRia-1"
OWN = ModelRef(repo=REPO, revision="dae83aa51ecd0a8574d415d929a4d9e79348fd2a", dtype="bf16")
PUBLISHED = ModelRef(repo=REPO, revision="published-497e243f68cd", dtype="bf16")
PUBLISHED_RUNTIME = "published"
PROMPT_ID = "ocria-1/page"
MAX_TOKENS = 24576
PARAMS = {"temperature": 0.0, "max_tokens": MAX_TOKENS, "enable_thinking": False}


class LazyBlobs(Mapping):
    """Input blobs encoded only if someone reads them: a cache hit never pays for PNG encoding."""

    def __init__(self, image, name: str):
        self._image, self._name, self._data = image, name, None

    def __getitem__(self, key: str) -> bytes:
        if key != self._name:
            raise KeyError(key)
        if self._data is None:
            self._data = png_bytes(self._image)
        return self._data

    def __iter__(self) -> Iterator[str]:
        return iter([self._name])

    def __len__(self) -> int:
        return 1


def build_page_call(image, model: ModelRef, runtime: str, prompt_sha: str) -> tuple[ModelCall, LazyBlobs]:
    """The one-image, system-prompt-only request the harness's `ocria_1` pipeline sends, as a keyed call."""
    name = pixel_hash(image)
    call = ModelCall(
        model=model, runtime=runtime, prompt_id=PROMPT_ID, prompt_sha=prompt_sha, params=dict(PARAMS), inputs=[name]
    )
    return call, LazyBlobs(image, name)


@dataclass
class PageRead:
    markdown: str  # post-processed, as scored
    raw_answer: str | None  # the model's raw answer; None for published stand-ins
    layout_page: dict | None  # a ParseBench layout page (PDF points), when the reading carries one
    finish_reason: str | None = None


def read_page_result(result: ModelResult) -> PageRead:
    out = result.output
    if "answer" in out:
        raw = out.get("answer") or ""
        return PageRead(ocria_postprocess(raw), raw, out.get("layout_page"), out.get("finish_reason"))
    return PageRead(out.get("markdown") or "", None, out.get("layout_page"), out.get("finish_reason"))

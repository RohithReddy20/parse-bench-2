"""LightOnOCR-3-4B page transcription: a third small reader, used for pages without a text layer.

The model is called as its card says: the page image alone (empty prompt = transcription mode), longest edge
2048 px, temperature 0.2, top_p 0.9, up to 4096 new tokens. It answers in Markdown with HTML tables and LaTeX math.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from docparse.models.adapters.ocria import LazyBlobs
from docparse.models.adapters.vlm_specialists import RUNTIME
from docparse.models.calls import ModelCall, ModelRef, ModelResult
from docparse.pdf.render import pixel_hash

LIGHTON3_4B = ModelRef(
    repo="lightonai/LightOnOCR-3-4B",
    revision="b71010f095dc3735de7ee6b9969bff8f8f50b39a",
    dtype="bf16",
)
PROMPT_ID = "lightonocr-3/transcribe"
MAX_SIDE = 2048
PARAMS: dict[str, Any] = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 4096}


def build_page_call(
    image: Image.Image, model: ModelRef = LIGHTON3_4B, runtime: str = RUNTIME
) -> tuple[ModelCall, LazyBlobs]:
    if max(image.size) > MAX_SIDE:
        s = MAX_SIDE / max(image.size)
        image = image.resize(
            (max(1, round(image.width * s)), max(1, round(image.height * s))), Image.Resampling.LANCZOS
        )
    name = pixel_hash(image)
    call = ModelCall(
        model=model, runtime=runtime, prompt_id=PROMPT_ID, prompt_sha="", params=dict(PARAMS), inputs=[name]
    )
    return call, LazyBlobs(image, name)


def read_page(result: ModelResult) -> tuple[str, bool]:
    """`(markdown, usable)`: a cut-off answer is not usable."""
    out = result.output
    text = (out.get("answer") or "").strip()
    return text, bool(text) and out.get("finish_reason") == "stop"

"""Docling heron (RT-DETRv2 layout detector): build the call for a rendered page, read its detections.

Settings follow `ocria_grounding.detector.Detector` in the OCRia-1 repository: the page is downscaled to at
most 1,600 px on its long side, detections below 0.2 are dropped, and boxes are reported in pixels of the
page image the call was made for.
"""

from __future__ import annotations

from dataclasses import dataclass

from docparse.models.calls import ModelCall, ModelRef, ModelResult

HERON = ModelRef(
    repo="docling-project/docling-layout-heron", revision="8f39ad3c0b4c58e9c2d2c84a38465abf757272d8", dtype="fp32"
)
RUNTIME = "transformers"
PROMPT_ID = "heron/detect"
PARAMS = {"threshold": 0.2, "max_side": 1600}


@dataclass(frozen=True)
class Detection:
    label: str  # Docling class name as the model reports it ("Text", "Section-header", "Picture", ...)
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels of the rendered page


def detect_call(page_hash: str, model: ModelRef = HERON) -> ModelCall:
    """The detection call for the page image whose `pixel_hash` is `page_hash` (its PNG is the input blob)."""
    return ModelCall(
        model=model, runtime=RUNTIME, prompt_id=PROMPT_ID, prompt_sha="", params=dict(PARAMS), inputs=[page_hash]
    )


def read_detections(result: ModelResult) -> list[Detection]:
    out = []
    for d in result.output.get("detections") or []:
        x0, y0, x1, y1 = (float(v) for v in d["box"])
        if x1 > x0 and y1 > y0:
            out.append(Detection(label=str(d["label"]), score=float(d["score"]), box=(x0, y0, x1, y1)))
    return out

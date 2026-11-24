"""Run a layout detector locally (CPU or Apple MPS) through the manifest runner. No cloud GPU, no cost.

    uv run --group detect python -m serve.local_detector --manifest DIR --cache DIR

The executor mirrors `ocria_grounding.detector.Detector` (OCRia-1 repository) so that our detections are the
ones OCRia-1's published layout was built from.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Any

from docparse.models.calls import ModelCall


def heron_executor(device: str | None = None):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    state: dict[str, Any] = {}
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def execute(call: ModelCall, blobs: dict[str, bytes]) -> dict[str, Any]:
        if "model" not in state:
            repo, revision = call.model.repo, call.model.revision
            state["proc"] = AutoImageProcessor.from_pretrained(repo, revision=revision)
            state["model"] = AutoModelForObjectDetection.from_pretrained(repo, revision=revision).to(device).eval()
        proc, model = state["proc"], state["model"]
        started = time.monotonic()
        image = Image.open(io.BytesIO(blobs[call.inputs[0]])).convert("RGB")
        w, h = image.size
        s = min(1.0, call.params["max_side"] / max(w, h))
        small = image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS) if s < 1 else image
        sx, sy = w / small.width, h / small.height
        with torch.no_grad():
            out = model(**proc(images=[small], return_tensors="pt").to(device))
        r = proc.post_process_object_detection(
            out, target_sizes=[(small.height, small.width)], threshold=call.params["threshold"]
        )[0]
        detections = [
            {
                "label": model.config.id2label[int(lab)],
                "score": round(float(sc), 3),
                "box": [
                    round(float(b[0]) * sx, 1),
                    round(float(b[1]) * sy, 1),
                    round(float(b[2]) * sx, 1),
                    round(float(b[3]) * sy, 1),
                ],
            }  # fmt: skip
            for sc, lab, b in zip(r["scores"], r["labels"], r["boxes"], strict=True)
        ]
        return {"detections": detections, "width": w, "height": h, "_gpu_seconds": round(time.monotonic() - started, 3),
                "_gpu_type": f"local-{device}"}  # fmt: skip

    return execute


def main(argv: list[str] | None = None) -> int:
    from docparse.models.cache import Cache
    from serve.run_manifest import import_results, run_manifest

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    results = args.manifest / "results.jsonl"
    stats = run_manifest(args.manifest, results, heron_executor(args.device))
    added = import_results(args.manifest, results, Cache(args.cache))
    print(
        f"{stats.done} done, {stats.failed} failed, {stats.skipped} skipped, {stats.seconds:.0f}s; {added} newly cached"
    )
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

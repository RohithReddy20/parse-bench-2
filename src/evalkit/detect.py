"""Build a layout-detection manifest for the page images of an existing page-read manifest.

The detector reads the same page PNG as OCRia-1, so its call is keyed by the same blob and the two manifests
share one `blobs/` folder (a symlink). Detection runs locally (`serve/local_detector.py`) at no cost.
"""

from __future__ import annotations

from pathlib import Path

from docparse.models.adapters.heron import detect_call
from docparse.models.manifest import ManifestWriter, read_manifest


def build_detect_manifest(page_manifest: Path, dest: Path) -> int:
    """One heron call per page image of `page_manifest`; returns the number of calls in `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    blobs = dest / "blobs"
    if not blobs.exists():
        blobs.symlink_to((page_manifest / "blobs").resolve(), target_is_directory=True)
    writer = ManifestWriter(dest)
    for call, meta in read_manifest(page_manifest):
        writer.add(detect_call(call.inputs[0]), {}, meta)
    return len(writer)

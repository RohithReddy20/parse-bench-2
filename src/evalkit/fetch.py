"""Pinned downloads into $DOCPARSE_DATA. Everything fetched here is data: it is read, never executed
(OCRia-1's own grounding code is the one exception, design §7)."""

from __future__ import annotations

import re
import time
from pathlib import Path

from evalkit import paths, pins

ATTEMPTS = 5
WORKERS = 4
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
OCRIA_CODE_PATTERNS = [
    "ocria_grounding/*",
    "examples/*",
    "prompt.txt",
    "*.json",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
]


def _snapshot(dest: Path, **kwargs) -> Path:
    """snapshot_download into `dest`, once: a `.complete` marker holding the revision ends the work."""
    from huggingface_hub import snapshot_download

    marker = dest / ".complete"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            snapshot_download(local_dir=str(dest), max_workers=WORKERS, **kwargs)
            break
        except Exception as exc:  # the Hub drops bursts (HTTP 429/499, resets); files on disk are kept
            if attempt == ATTEMPTS:
                raise
            wait = 15 * 2 ** (attempt - 1)
            print(f"download failed ({type(exc).__name__}: {str(exc)[:120]}); retry in {wait}s", flush=True)
            time.sleep(wait)
    marker.write_text(kwargs["revision"])
    return dest


def fetch_ground_truth() -> Path:
    return _snapshot(paths.gt_dir(), repo_id=pins.DATA_REPO, repo_type="dataset", revision=pins.DATA_REVISION)


def fetch_published(pipeline: str = pins.OCRIA_PIPELINE, revision: str = pins.RESULTS_REVISION) -> Path:
    """Folder holding `<subset>.jsonl` for one published pipeline."""
    if not _NAME.fullmatch(pipeline):
        raise ValueError(f"not a pipeline name: {pipeline!r}")
    dest = paths.data_root() / "published" / revision / pipeline
    _snapshot(
        dest,
        repo_id=pins.RESULTS_REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"results/{pipeline}/*"],
    )
    return dest / "results" / pipeline


def fetch_ocria_code() -> Path:
    """OCRia-1's prompt, configs and grounding code. No weights."""
    return _snapshot(
        paths.ocria_dir(),
        repo_id=pins.OCRIA_REPO,
        repo_type="model",
        revision=pins.OCRIA_REVISION,
        allow_patterns=OCRIA_CODE_PATTERNS,
    )

"""Shared fixtures. Tests marked `data` need the pinned downloads under $DOCPARSE_DATA and skip without them."""

import json
from pathlib import Path

import pytest

from evalkit import paths, pins


def _published() -> Path:
    return (
        paths.data_root() / "published" / pins.RESULTS_REVISION / pins.OCRIA_PIPELINE / "results" / pins.OCRIA_PIPELINE
    )


@pytest.fixture(scope="session")
def published_dir() -> Path:
    path = _published()
    if not (path.parent.parent / ".complete").exists():
        pytest.skip("published OCRia-1 outputs not downloaded (python -m evalkit fetch published)")
    return path


@pytest.fixture(scope="session")
def gt_dir() -> Path:
    path = paths.gt_dir()
    if not (path / ".complete").exists():
        pytest.skip("benchmark data not downloaded (python -m evalkit fetch gt)")
    return path


@pytest.fixture(scope="session")
def published_markdown(published_dir) -> dict[str, str]:
    """example id -> OCRia-1's published page markdown, for all 13 subsets."""
    out: dict[str, str] = {}
    for path in sorted(published_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    out[record["request"]["example_id"]] = record["output"]["markdown"]
    return out

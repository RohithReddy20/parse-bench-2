"""Where downloaded data, the model-call cache and run outputs live: always outside the repo."""

import os
from pathlib import Path

from evalkit import pins


def data_root() -> Path:
    env = os.environ.get("DOCPARSE_DATA")
    return Path(env) if env else Path.home() / ".cache" / "docparse-v2"


def gt_dir() -> Path:
    return data_root() / "gt" / pins.DATA_REVISION


def published_dir() -> Path:
    return data_root() / "published" / pins.RESULTS_REVISION


def ocria_dir() -> Path:
    return data_root() / "ocria1" / pins.OCRIA_REVISION


def cache_dir() -> Path:
    return data_root() / "cache"


def runs_dir() -> Path:
    return data_root() / "runs"


def ledger_path() -> Path:
    return data_root() / "ledger.sqlite"

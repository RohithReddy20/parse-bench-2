"""Pipeline configuration. Everything that can change an output is here, and `config_hash` identifies it."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from docparse.core.hashing import canonical_json, sha256_hex
from docparse.models.calls import ModelRef

FIXES = ("layout", "forms", "tables", "charts", "text", "formatting")
LAYOUT_OPTIONS = frozenset({"tighten", "tighten_pad", "native_text", "native_support", "split_gap"})


class Config(BaseModel):
    page_model: ModelRef
    page_runtime: str
    prompt_sha: str  # sha256 of OCRia-1's prompt.txt ("published" for stand-in readings)
    dpi: int = 300
    # where layout items come from: carried by the page read (published stand-ins); built here from detector
    # regions and PDF words with OCRia-1's grounding code ("ground"); or none at all
    layout: Literal["from_page_read", "ground", "none"] = "from_page_read"
    # refinements of the layout items on pages with a text layer (fixes/layout_refine.py); empty = none
    layout_options: dict[str, float | bool] = Field(default_factory=dict)
    fixes: dict[str, bool] = Field(default_factory=lambda: dict.fromkeys(FIXES, False))

    def config_hash(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))[:16]

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> Config:
        """`overrides`: `key=value` or `section.key=value` strings (values parsed as YAML), for experiments."""
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        for item in overrides or []:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"override must be key=value: {item!r}")
            target = data
            *parents, leaf = key.split(".")
            for name in parents:
                target = target.setdefault(name, {})
            target[leaf] = yaml.safe_load(value)
        if set(data.get("layout_options") or {}) - LAYOUT_OPTIONS:
            raise ValueError(f"unknown layout option(s): {sorted(set(data['layout_options']) - LAYOUT_OPTIONS)}")
        unknown = set(data.get("fixes") or {}) - set(FIXES)
        if unknown:
            raise ValueError(f"unknown fix flag(s): {sorted(unknown)}")
        return cls.model_validate(data)

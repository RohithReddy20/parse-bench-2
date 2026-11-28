"""Pipeline configuration. Everything that can change an output is here, and `config_hash` identifies it."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from docparse.core.hashing import canonical_json, sha256_hex
from docparse.models.calls import ModelRef

FIXES = ("layout", "forms", "tables", "charts", "text", "formatting")
LAYOUT_OPTIONS = frozenset(
    {
        "tighten", "tighten_pad", "native_text", "native_support", "split_gap",
        "orphans", "orphan_gap", "orphan_min", "orphan_ink",
        "kdl_replace", "kdl_words", "kdl_add", "kdl_split", "kdl_children", "kdl_pictures", "kdl_picbox",
        "kdl_arbiter", "kdl_keep_pictures", "kdl_keep_tables",
        "pp_arbiter", "pp_keep_pictures", "pp_add", "pp_min_score", "pp_scan_arbiter", "blank_drop",
        "chandra_arbiter", "chandra_add", "chandra_fallback",
    }
)  # fmt: skip
TABLE_OPTIONS = frozenset(
    {
        "regrid",
        "regrid_wide",
        "complete",
        "merge",
        "extend",
        "health",
        "unspan",
        "unspan_mode",
        "dedupe",
        "upright",
        "split",
        "absorb",
        "leaf",
        "leaf_mode",
        "complete_extent",
        "table_geo",
        # round three (PDF-grounded column and merge repairs)
        "squeeze",
        "sign",
        "head_fill",
        "last_column",
        "merge_apart",
        "merge_strict",
    }
)  # fixes/tables.py
FORM_OPTIONS = frozenset(
    {"combs", "boxes", "outlines", "ids", "codes", "grids", "trailing", "blanks", "cells", "hints", "blank_forms"}
    # the separate forms round-two line ("Neon", 2026-11-25-forms-neon-integration.md); `n_blank_forms` is its own
    # blank-form repair (forms_blank.py), not `blank_forms`
    | {
        "mini_tables", "table_rows", "box_grids", "bare_ticks", "field_lines", "table_clean", "acroform",
        "comb_noise", "field_notes", "field_slots", "table_grids", "field_clean", "acro_clean", "blank_hints",
        "n_blank_forms", "checkbox_ink", "route_blanks",
    }
)  # fmt: skip
# chart tables corrected from the PDF's vector drawing (fixes/charts.py)
CHART_OPTIONS = frozenset(
    {
        "measure",
        "after_tier2",  # the vector measurements also correct a Tier-2 chart read
        "agree",
        "agree_tol",
        "replace_min",
        "coverage",
        "bar_mode",
        "legend",
        "legend_max",
        "legend_corr",
        "zero_reach",
        "years",
        "bent",
        "signs",
        "upright",
        "reverse",
        "labels",
        "sides",
        "snap",
        "snap_tol",
        "panels",
        "anchors",
        "lost",  # round two: tick labels with leading digits missing from the text layer
        "rotated",  # round two: rotated category labels ("ordered" or "any")
        "pos_marks",  # round two: markers placed on categories by their labels
        "titles",  # round two: legend binding finds a series' panel by the title printed above it
        "legend_min",  # round two: fewest cells a legend-bound series may have (default 3)
        "legend_override",  # round two: a binding whose colour contradicts the legend takes the legend's series
        "pitch",  # round two: bars at a regular pitch with categories missing between them
        "elim",  # round two: the last unbound series of a plot takes its last unused drawn series
        "panel_override",  # round two: ... and a binding to another panel than the titled one moves to it
    }
)
# Tier-2 reads with Qwen3.5/3.8 (pipeline.py, models/adapters/qwen_tier2.py); `model` is a `TIER2_MODELS` alias
# ("27b", "122b", "q38") or repo id, Qwen3.5-27B-FP8 when unset
TIER2_CHART_OPTIONS = frozenset(
    {
        "model", "enable_thinking", "temperature", "top_p", "top_k", "presence_penalty", "max_tokens",
        "use_ppdoclayout", "pp_chart_score", "per_chart_crops", "per_chart_min_union", "per_chart_max_boxes",
        "hint_max_chars", "hint_max_series", "prompt", "crop_zoom_to", "hint_chars", "fallback_plain", "samples",
        "seed",
    }
)  # fmt: skip
# Tier-2 draft-guided form refinement
TIER2_FORM_OPTIONS = frozenset(
    {
        "model", "enable_thinking", "temperature", "top_p", "top_k", "presence_penalty", "max_tokens",
        "min_word_coverage", "min_checkboxes", "min_acro_widgets",
        "prompt_version", "two_strips", "strip_overlap", "prompt", "layer", "layer_chars", "wide", "fallback_plain",
        "any_reader",
    }
)  # fmt: skip
# Tier-2 strip re-read of dense table pages
TIER2_TABLE_OPTIONS = frozenset(
    {
        "model", "n_strips", "strip_overlap", "max_tokens", "temperature", "top_p", "presence_penalty",
        "min_recall_trigger", "min_recall_gain",
    }
)  # fmt: skip


class Config(BaseModel):
    page_model: ModelRef
    page_runtime: str
    prompt_sha: str  # sha256 of OCRia-1's prompt.txt ("published" for stand-in readings)
    # Primary page reader engine:
    # - "ocria": single-model OCRia-1 page read (default)
    # - "chandra": single-model Chandra-OCR-2 layout-HTML read (with OCRia-1 fallback on length/loop)
    # - "chandra_hybrid": Chandra-OCR-2 primary + OCRia-1 fallback on length/loop & vector-chart Figure fusion
    reader: Literal["ocria", "chandra", "chandra_hybrid"] = "ocria"
    # choosing between the page's independent readings (fixes/select.py): `layer_margin` = how much better the
    # other reading must agree with a healthy PDF text layer to replace the primary one; empty = no selection
    select_options: dict[str, Any] = Field(default_factory=dict)
    chandra_model: ModelRef | None = None
    # Drop Page-Header and Page-Footer blocks when the reader tags page furniture (e.g. Chandra layout blocks)
    strip_furniture: bool = False
    # Verify Chandra reads against a healthy PDF text layer (low word recall or flattened table -> secondary check)
    verify_pdf: bool = False
    # Chandra readers: a page with a chart (a Chandra chart Figure, or a PP-DocLayoutV3 chart box) is read once more
    # by Qwen3.5, whose chart tables replace Chandra's (or are added when only PP-DocLayoutV3 saw the chart); with
    # `per_chart_crops`, a page whose 2..`per_chart_max_boxes` chart boxes cover `per_chart_min_union` of it is
    # read once per chart, each with its own zoomed crop
    tier2_charts: bool = False
    # sampling and trigger settings of that read (TIER2_CHART_OPTIONS); empty = the defaults in pipeline.py
    tier2_chart_options: dict[str, Any] = Field(default_factory=dict)
    # Chandra readers: a form page (no math; a Chandra `Form` block, dense checkboxes or AcroForm widgets) that took
    # no Tier-2 chart read has its draft checked against the page by Qwen3.5; the refined copy replaces it
    # only when it keeps `min_word_coverage` of the draft's words
    tier2_forms: bool = False
    # sampling and trigger settings of that read (TIER2_FORM_OPTIONS); empty = the defaults in pipeline.py
    tier2_form_options: dict[str, Any] = Field(default_factory=dict)
    # a tabular page without math whose read hit the token limit, or (healthy text layer) holds under
    # `min_recall_trigger` of its PDF words, is re-read by Qwen3.5 as `n_strips` overlapping horizontal strips; the
    # stitched copy is taken only when it recovers PDF words (a scan: only in place of a cut-off read). Chandra
    # pages already escalated for a chart or form are never re-read
    tier2_tables: bool = False
    # model, sampling, strip and gate settings of that read (TIER2_TABLE_OPTIONS); empty = the defaults in pipeline.py
    tier2_table_options: dict[str, Any] = Field(default_factory=dict)
    dpi: int = 300
    # where layout items come from: carried by the page read (published stand-ins); built here from detector
    # regions and PDF words with OCRia-1's grounding code ("ground"); or none at all
    layout: Literal["from_page_read", "ground", "none"] = "from_page_read"
    # with layout "ground": OCRia-1's attention (one more cached model call per page) places what the PDF text
    # layer cannot: every word on pages without a text layer ("no_text_layer"), and picture references on any
    # page ("always"). "never" places such things by their neighbours only
    layout_attention: Literal["never", "no_text_layer", "always"] = "never"
    # refinements of the layout items on pages with a text layer (fixes/layout_refine.py); empty = none
    # a page read that looped or hit the token limit, and holds under this share of the PDF text layer's words,
    # is replaced by the text layer (0 = off)
    fallback_below: float = 0.0
    # a page whose text layer runs sideways or upside down is turned upright before it is read and corrected
    upright_read: bool = False
    # a page read that hit the token limit is read once more with these extra request parameters, and the second
    # read is used when it ends normally (empty = no second read). Keys: repetition_penalty, image_scale
    reread_truncated: dict[str, float] = Field(default_factory=dict)
    layout_options: dict[str, float | bool] = Field(default_factory=dict)
    # corrections of HTML tables from the PDF text layer (fixes/tables.py); empty = none
    table_options: dict[str, float | bool] = Field(default_factory=dict)
    # repairs of form pages from the text a fill tool typed on top of the page (fixes/forms.py); empty = none
    form_options: dict[str, float | bool] = Field(default_factory=dict)
    # chart values measured from vector bars, lines and markers against tick-label axes; empty = off
    chart_options: dict[str, float | bool | str] = Field(default_factory=dict)
    # formatting corrections of the page reading from the PDF (fixes/formatting.py); empty = none
    format_options: dict[str, float | bool] = Field(default_factory=dict)
    # text corrections of the page reading from the PDF text layer (fixes/text.py); empty = none
    text_options: dict[str, float | bool] = Field(default_factory=dict)
    fixes: dict[str, bool] = Field(default_factory=lambda: dict.fromkeys(FIXES, False))

    def config_hash(self) -> str:
        # the Tier-2 fields enter the hash only when set, so configs that predate them keep their hash
        tier2 = (
            "tier2_charts", "tier2_chart_options", "tier2_forms", "tier2_form_options",
            "tier2_tables", "tier2_table_options", "select_options",
        )  # fmt: skip
        unset = {name for name in tier2 if not getattr(self, name)}
        return sha256_hex(canonical_json(self.model_dump(mode="json", exclude=unset)))[:16]

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
        if set(data.get("table_options") or {}) - TABLE_OPTIONS:
            raise ValueError(f"unknown table option(s): {sorted(set(data['table_options']) - TABLE_OPTIONS)}")
        if set(data.get("form_options") or {}) - FORM_OPTIONS:
            raise ValueError(f"unknown form option(s): {sorted(set(data['form_options']) - FORM_OPTIONS)}")
        if set(data.get("chart_options") or {}) - CHART_OPTIONS:
            raise ValueError(f"unknown chart option(s): {sorted(set(data['chart_options']) - CHART_OPTIONS)}")
        from docparse.fixes.formatting import FORMAT_OPTIONS

        if set(data.get("format_options") or {}) - FORMAT_OPTIONS:
            raise ValueError(f"unknown format option(s): {sorted(set(data['format_options']) - FORMAT_OPTIONS)}")
        from docparse.fixes.text import TEXT_OPTIONS

        if set(data.get("text_options") or {}) - TEXT_OPTIONS:
            raise ValueError(f"unknown text option(s): {sorted(set(data['text_options']) - TEXT_OPTIONS)}")
        unknown = set(data.get("tier2_chart_options") or {}) - TIER2_CHART_OPTIONS
        if unknown:
            raise ValueError(f"unknown tier2 chart option(s): {sorted(unknown)}")
        unknown = set(data.get("tier2_form_options") or {}) - TIER2_FORM_OPTIONS
        if unknown:
            raise ValueError(f"unknown tier2 form option(s): {sorted(unknown)}")
        unknown = set(data.get("tier2_table_options") or {}) - TIER2_TABLE_OPTIONS
        if unknown:
            raise ValueError(f"unknown tier2 table option(s): {sorted(unknown)}")
        unknown = set(data.get("fixes") or {}) - set(FIXES)
        if unknown:
            raise ValueError(f"unknown fix flag(s): {sorted(unknown)}")
        return cls.model_validate(data)

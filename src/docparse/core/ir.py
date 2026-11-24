"""The intermediate representation (architecture plan §1.3, revised for design revision 2).

A page is a list of regions in reading order. Each region keeps every candidate reading it received and the
id of the selected one. OCRia-1's answer is the base candidate of every region; correctors add candidates.

Lossless write-back: `Candidate.raw` is the exact markdown the candidate writes and `Region.sep_after` the
whitespace up to the next block, so a page nobody corrected writes back byte for byte.

Coordinates are PDF points, origin top-left, in the displayed page frame (CropBox and /Rotate applied).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Style = Literal["b", "i", "u", "s", "sup", "sub", "mark", "code"]
RegionKind = Literal["heading", "paragraph", "list", "table", "chart", "form", "code", "image_ref", "opaque"]
Trust = Literal["native", "native_partial", "ocr_layer", "pixels_only", "unknown"]


class Box(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class Provenance(BaseModel):
    source: Literal["pdf_text", "pdf_fill", "pdf_widget", "pdf_drawing", "model", "rule"]
    ref: str  # evidence id, model call key, or rule id
    detail: str = ""


class Confidence(BaseModel):
    value: float = 0.0  # 0..1, ranked rather than calibrated
    basis: list[str] = Field(default_factory=list)


class Span(BaseModel):
    text: str
    bbox: Box | None = None
    style: list[Style] = Field(default_factory=list)
    prov: Provenance | None = None
    critical: bool = False  # number, id, unit, negator


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    lines: list[list[Span]] = Field(default_factory=list)
    role: Literal["paragraph", "heading", "list_item", "caption", "footnote", "furniture", "code"] = "paragraph"
    heading_level: int | None = None
    list_depth: int | None = None


class Cell(BaseModel):
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    header: bool = False
    spans: list[Span] = Field(default_factory=list)
    word_ids: list[str] = Field(default_factory=list)


class TableContent(BaseModel):
    type: Literal["table"] = "table"
    cells: list[Cell] = Field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    title: str | None = None


class ChartPoint(BaseModel):
    category: str
    series: str
    value: str  # the string that will be printed
    numeric: float | None = None
    prov: Provenance | None = None
    tolerance_margin: float | None = None


class ChartContent(BaseModel):
    type: Literal["chart"] = "chart"
    title: str | None = None
    panel_title: str | None = None
    chart_type: str | None = None
    unit: str | None = None
    categories: list[str] = Field(default_factory=list)
    series: list[str] = Field(default_factory=list)
    points: list[ChartPoint] = Field(default_factory=list)
    diagnostics: dict[str, float] = Field(default_factory=dict)


class FormField(BaseModel):
    caption: list[Span] = Field(default_factory=list)
    printed_id: str | None = None
    kind: Literal["text", "checkbox", "signature"] = "text"
    value: list[Span] | None = None  # None means blank
    checked: bool | None = None
    grid: tuple[int, int] | None = None
    prov: Provenance | None = None


class FormContent(BaseModel):
    type: Literal["form"] = "form"
    fields: list[FormField] = Field(default_factory=list)
    tables: list[TableContent] = Field(default_factory=list)


class OpaqueContent(BaseModel):
    """Markdown the IR does not model (an unclosed table, a truncated answer): carried and written verbatim."""

    type: Literal["opaque"] = "opaque"
    reason: str = ""


Content = TextContent | TableContent | ChartContent | FormContent | OpaqueContent


class CheckResult(BaseModel):
    check: str
    passed: bool
    level: Literal["hard", "soft"] = "soft"
    evidence: str = ""


class Candidate(BaseModel):
    id: str
    reader: str  # "ocria.base", "fix.tables.words", ...
    raw: str  # exact markdown this candidate writes
    content: Content | None = Field(default=None, discriminator="type")
    prov: list[Provenance] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)


class RepairRecord(BaseModel):
    check: str
    action: str
    call_key: str | None = None
    outcome: str = ""


class Finding(BaseModel):
    region: str | None = None
    check: str
    level: Literal["hard", "soft"] = "soft"
    evidence: str = ""
    actions: list[str] = Field(default_factory=list)


class LayoutItem(BaseModel):
    """One item as the layout scorer reads it: a box, a canonical segment label, and its own text."""

    type: str  # "Text", "Section-header", "Table", "Picture", ...
    label: str  # segment label ("text", "table", "image", ...)
    md: str
    bbox: Box
    region: str | None = None  # id of the region it belongs to, when known


class Region(BaseModel):
    id: str
    kind: RegionKind
    order: int
    bbox: Box | None = None
    segments: list[Box] = Field(default_factory=list)
    trust: Trust = "unknown"
    proposals: list[Provenance] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    selected: str | None = None
    repairs: list[RepairRecord] = Field(default_factory=list)
    sep_after: str = ""  # whitespace between this block and the next one in the page markdown

    def chosen(self) -> Candidate:
        for c in self.candidates:
            if c.id == self.selected:
                return c
        raise ValueError(f"region {self.id}: no selected candidate")


class CostRecord(BaseModel):
    calls: int = 0
    cached_calls: int = 0
    gpu_seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


class Page(BaseModel):
    index: int
    width: float = 0.0  # points
    height: float = 0.0
    rotation: int = 0
    page_class: Literal["born_digital", "scanned", "mixed", "unknown"] = "unknown"
    leading: str = ""  # whitespace before the first block
    regions: list[Region] = Field(default_factory=list)
    layout: list[LayoutItem] | None = None  # None: no layout for this page
    findings: list[Finding] = Field(default_factory=list)
    cost: CostRecord = Field(default_factory=CostRecord)


class Document(BaseModel):
    sha256: str
    n_pages: int
    pages: list[Page]
    config_hash: str = ""
    code_version: str = ""

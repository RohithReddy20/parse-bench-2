"""Plain OCRia-1 against OCRia-1 + every kept correction on forms no rule was written with: RealDocBench's QA set.

RealDocBench (Extend, arXiv 2606.07401) ships 581 single-page regulated documents (mortgage, finance, supply
chain, medical) with 1,356 questions, each with a typed gold answer (`gold_dict`). Its harness scores a parser by
having Gemini answer each question from the parser's markdown and comparing the answer to `gold_dict`; that path
needs a paid model API and is not run here. Instead, both candidates are scored deterministically on the same
gold values:

    value found        the gold value appears in the document's markdown, as a whole token (text), as a number
                       equal to it (any of `12800`, `12,800.00`, `$12,800`), or as a date that parses to it
    value found, loose the same after forgiving spacing and punctuation (`1 2 8 0 0`, `Smith,John`), the
                       spacing the harness's own extractor is told to collapse
    question complete  every measurable field of the question found

Check-box states (booleans), choices among printed options and absent values (null) cannot be judged by
presence and are counted but not scored. Run with `python -I` (the downloaded data is untrusted):

    uv run python -I -m evalkit.external.forms_outside plan    # manifest of missing page reads, both configs
    uv run python -I -m evalkit.external.forms_outside write   # once the reads are cached: markdown per page
    uv run python -I -m evalkit.external.forms_outside score   # both candidates side by side
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evalkit.paths import cache_dir, data_root, runs_dir

REPO = "Extend-AI/RealDoc-Bench"
REVISION = "08dde9d20d844943fc328b161f9d332c7e8bdee4"
HARNESS = "https://github.com/extend-hq/realdoc-bench"
HARNESS_COMMIT = "9a352990ec675405c021d01eee27322e3ba726bc"
SEED = 20261125  # documents are sampled with this seed when the set holds more than CAP_PAGES pages
CAP_PAGES = 600
CONFIGS = {"plain": "configs/ocria_own.yaml", "corrected": "configs/own_best.yaml"}
OVERRIDES = ["layout=none"]
SCORED = ("text", "number", "date")
# questions whose capability tags name a form trait (field-value pairs, check boxes, blanks, handwriting)
FORM_TAG = re.compile(r"form|field|check|blank|handwrit|bubble|circled")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return data_root() / "external" / "realdocbench" / f"qa-{REVISION[:7]}"


def out_dir() -> Path:
    return runs_dir() / "ext-realdoc"


# ── which gold fields can be judged from the markdown ──────────────────────────────────────────────────────────
_PAIR = re.compile(r"([^=<>;|]+)=<([^>]+)>")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def response_tokens(response_format: str) -> dict[str, str]:
    """`Return exactly: a=<number>; b=<Yes|No|null>` -> {"a": "number", "b": "Yes|No|null"}."""
    body = re.sub(r"^\s*Return exactly[^:]*:\s*", "", response_format or "")
    return {k.strip(): t.strip() for k, t in _PAIR.findall(body)}


def field_kind(token: str | None, value: Any) -> str:
    """absent | state | choice | list | date | number | text. Only date, number and text are scored."""
    if value is None:
        return "absent"
    if isinstance(value, bool):
        return "state"
    if isinstance(value, list):
        return "list"
    options = [o.strip().lower() for o in (token or "").split("|") if o.strip() and o.strip().lower() != "null"]
    if len(options) > 1:
        return "choice"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str) and _ISO.match(value) and "date" in (token or "").lower():
        return "date"
    return "text"


# ── the markdown as a haystack ─────────────────────────────────────────────────────────────────────────────────
_DASH = re.compile(r"[‐‑‒–—―−]")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "′": "'", "“": '"', "”": '"', "„": '"', "″": '"'})
_TAG = re.compile(r"<br\s*/?>|</?[a-z][^>]*>", re.I)
_STYLE = re.compile(r"\*\*|__|~~|`")
_AROUND_SEP = re.compile(r"\s*([/\-,:])\s*")
_NUMBER = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?!\d)")
_DIGIT_GAP = re.compile(r"(?<=\d) (?=\d)|(?<=\d) ?([.,]) ?(?=\d)")
_NUMERIC_DATE = re.compile(r"\d{1,4}[/.\-]\d{1,2}[/.\-]\d{2,4}")
_WORDED_DATE = re.compile(r"[a-z]{3,9}\.? ?\d{1,2}(?:, ?| )\d{4}|\d{1,2} [a-z]{3,9}\.? \d{4}")
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%d/%m/%Y", "%d-%m-%Y",
                 "%d.%m.%Y", "%m/%d/%y", "%m-%d-%y", "%d/%m/%y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
                 "%B %d %Y", "%b %d %Y")  # fmt: skip


def normalize(text: str) -> str:
    """Casefolded, styling and tags removed, dashes and quotes unified, whitespace collapsed (around separators too)."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = _STYLE.sub("", _TAG.sub(" ", _DASH.sub("-", text)))
    text = _AROUND_SEP.sub(r"\1", text.replace("|", " | "))
    return re.sub(r"\s+", " ", text).strip().casefold()


def fold(text: str) -> str:
    """Letters and digits only."""
    return re.sub(r"[\W_]+", "", normalize(text))


def close_digit_gaps(text: str) -> str:
    """`1 2 8 0 0` -> `12800`, `1 , 234 . 50` -> `1,234.50`: OCR character spacing collapsed."""
    return _DIGIT_GAP.sub(lambda m: m.group(1) or "", text)


def numbers_in(text: str) -> set[float]:
    out = set()
    for token in _NUMBER.findall(text):
        try:
            out.add(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def _iso_dates(candidate: str) -> set[str]:
    candidate = re.sub(r"\s*,\s*", ", ", re.sub(r"\bsept\b", "sep", candidate.rstrip(".,;:")))
    candidate = re.sub(r"^([a-z]{3,9})\.?(?=\d)", r"\1 ", candidate).replace(". ", " ")
    out = set()
    for fmt in _DATE_FORMATS:
        try:
            out.add(datetime.strptime(candidate, fmt).strftime("%Y-%m-%d"))
        except ValueError:
            continue
    return out


def dates_in(text: str) -> set[str]:
    """Every ISO date a date-like string of the (normalized) text can be read as, day-first and month-first."""
    out: set[str] = set()
    for candidate in _NUMERIC_DATE.findall(text) + _WORDED_DATE.findall(text):
        out |= _iso_dates(candidate)
    return out


@dataclass
class Haystack:
    text: str
    loose_text: str
    folded: str
    numbers: set[float] = field(default_factory=set)
    loose_numbers: set[float] = field(default_factory=set)
    dates: set[str] = field(default_factory=set)
    loose_dates: set[str] = field(default_factory=set)

    @classmethod
    def of(cls, markdown: str) -> Haystack:
        text = normalize(markdown)
        loose = close_digit_gaps(text)
        return cls(text, loose, fold(markdown), numbers_in(text), numbers_in(loose), dates_in(text), dates_in(loose))


def _whole(needle: str, text: str) -> bool:
    return bool(needle) and re.search(r"(?<![^\W_])" + re.escape(needle) + r"(?![^\W_])", text) is not None


def found(kind: str, value: Any, hay: Haystack) -> tuple[bool, bool]:
    """(found, found loosely) for a scored field."""
    if kind == "number":
        target = abs(float(value))
        return target in hay.numbers, target in hay.loose_numbers or target in hay.numbers
    if kind == "date":
        return value in hay.dates, value in hay.loose_dates or value in hay.dates
    needle = normalize(str(value))
    exact = _whole(needle, hay.text)
    folded = fold(str(value))
    if len(folded) < 4:  # too short to find loosely without matching inside other words
        return exact, exact or _whole(needle, hay.loose_text)
    return exact, exact or folded in hay.folded


# ── sample, bootstrap ──────────────────────────────────────────────────────────────────────────────────────────
def sample_documents(pages: dict[str, int], cap: int = CAP_PAGES, seed: int = SEED) -> list[str]:
    """All documents when their pages fit under `cap`; else a seeded random set of whole documents that does."""
    if sum(pages.values()) <= cap:
        return sorted(pages)
    order = sorted(pages)
    random.Random(seed).shuffle(order)
    chosen, total = [], 0
    for stem in order:
        if total + pages[stem] <= cap:
            chosen.append(stem)
            total += pages[stem]
    return sorted(chosen)


def bootstrap_gain(per_doc: list[tuple[int, int, int]], samples: int = 2000, seed: int = 7) -> tuple[float, float]:
    """per_doc: (found by A, found by B, fields). Gain of B over A in points, and P(gain > 0) over documents."""
    per_doc = [d for d in per_doc if d[2]]
    if not per_doc:
        return 0.0, 0.0

    def gain(docs) -> float:
        n = sum(d[2] for d in docs)
        return 100.0 * (sum(d[1] for d in docs) - sum(d[0] for d in docs)) / n if n else 0.0

    rng = random.Random(seed)
    wins = sum(gain([rng.choice(per_doc) for _ in per_doc]) > 0 for _ in range(samples))
    return gain(per_doc), wins / samples


# ── data ───────────────────────────────────────────────────────────────────────────────────────────────────────
def load_items() -> list[dict]:
    return json.loads((data_dir() / "qa_bank.json").read_text(encoding="utf-8"))["items"]


def documents() -> list[Path]:
    return sorted((data_dir() / "docs").glob("*.pdf"))


def _config(name: str):
    from docparse.core.config import Config

    return Config.load(_REPO_ROOT / CONFIGS[name], OVERRIDES)


def _page_counts(pdfs: list[Path]) -> dict[str, int]:
    import pypdfium2 as pdfium

    return {p.stem: len(pdfium.PdfDocument(p.read_bytes())) for p in pdfs}


def chosen_documents() -> list[Path]:
    pdfs = documents()
    keep = set(sample_documents(_page_counts(pdfs)))
    return [p for p in pdfs if p.stem in keep]


def profile(pdfs: list[Path]) -> dict[str, dict]:
    """Per document, what the corrections could act on: text layer, the typed-in overlay the form repairs need."""
    import pypdfium2 as pdfium

    from docparse.pdf.overlay import overlay_fonts

    out = {}
    for p in pdfs:
        data = p.read_bytes()
        pdf = pdfium.PdfDocument(data)
        chars = [pdf[i].get_textpage().count_chars() for i in range(len(pdf))]
        out[p.stem] = {
            "pages": len(pdf),
            "text_pages": sum(c >= 50 for c in chars),
            "overlay_pages": sum(bool(overlay_fonts(data, i)) for i in range(len(pdf))),
        }
    return out


# ── commands ───────────────────────────────────────────────────────────────────────────────────────────────────
def cmd_plan() -> int:
    from docparse.core.errors import ParseError
    from docparse.models.cache import Cache
    from docparse.models.client import CacheOnlyClient
    from docparse.models.manifest import ManifestWriter
    from docparse.pipeline import parse

    pdfs = chosen_documents()
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    cache, writer = Cache(cache_dir()), ManifestWriter(out / "manifest")
    before = len(writer)
    errors: dict[str, str] = {}
    for name in CONFIGS:
        cfg, pending = _config(name), 0
        for p in pdfs:
            try:
                pending += len(parse(p.read_bytes(), cfg, CacheOnlyClient(cache, writer)).pending)
            except ParseError as exc:
                errors[p.stem] = str(exc)
        print(f"{name:10} pending page reads {pending}")
    prof = profile(pdfs)
    (out / "profile.json").write_text(json.dumps(prof, indent=1))
    (out / "plan.json").write_text(json.dumps({
        "dataset": f"{REPO}@{REVISION}", "documents": [p.stem for p in pdfs], "seed": SEED, "cap_pages": CAP_PAGES,
        "configs": CONFIGS, "overrides": OVERRIDES, "manifest_calls": len(writer), "new_calls": len(writer) - before,
        "errors": errors}, indent=1))  # fmt: skip
    pages = sum(v["pages"] for v in prof.values())
    print(f"documents {len(pdfs)}  pages {pages}  with a text layer {sum(v['text_pages'] for v in prof.values())}  "
          f"with a typed-in overlay {sum(v['overlay_pages'] for v in prof.values())}")  # fmt: skip
    new = len(writer) - before
    print(f"new calls {new}  manifest {len(writer)} calls  at {out / 'manifest'}  errors {len(errors)}")
    return 0


def cmd_write() -> int:
    from docparse.core.errors import ParseError
    from docparse.models.cache import Cache
    from docparse.models.client import ReplayClient
    from docparse.pipeline import parse
    from docparse.write.markdown import write_page

    pdfs, cache, out = chosen_documents(), Cache(cache_dir()), out_dir()
    for name in CONFIGS:
        cfg, errors = _config(name), {}
        for p in pdfs:
            try:
                doc = parse(p.read_bytes(), cfg, ReplayClient(cache)).document
            except ParseError as exc:
                errors[p.stem] = str(exc)
                continue
            d = out / name / p.stem
            d.mkdir(parents=True, exist_ok=True)
            for page in doc.pages:
                (d / f"p{page.index:03d}.md").write_text(write_page(page), encoding="utf-8")
            findings = [[pg.index, f.check, f.evidence[:200]] for pg in doc.pages for f in pg.findings]
            (d / "findings.json").write_text(json.dumps(findings))
        (out / f"{name}_errors.json").write_text(json.dumps(errors, indent=1))
        print(f"{name:10} written {len(pdfs) - len(errors)} documents, {len(errors)} unreadable")
    return 0


def _pages(name: str, stem: str) -> list[str]:
    d = out_dir() / name / stem
    return [p.read_text(encoding="utf-8") for p in sorted(d.glob("p*.md"))] if d.is_dir() else []


def score_items(items: list[dict], markdown: dict[str, dict[str, str]]) -> dict[str, list[dict]]:
    """One row per (candidate, question): counts of scored fields found, exact and loose, and of unscored ones."""
    rows: dict[str, list[dict]] = {}
    for name, docs in markdown.items():
        hays = {stem: Haystack.of(md) for stem, md in docs.items()}
        out = rows[name] = []
        for it in items:
            hay = hays.get(it["source_file"]) or Haystack.of("")
            tokens = response_tokens(it.get("response_format", ""))
            row = {"qid": it["question_id"], "doc": it["source_file"], "domain": it["domain"], "fields": 0,
                   "found": 0, "loose": 0, "unscored": {}}  # fmt: skip
            row["form"] = any(FORM_TAG.search(c) for c in it.get("capabilities", []))
            for key, value in it["gold_dict"].items():
                kind = field_kind(tokens.get(key), value)
                if kind not in SCORED:
                    row["unscored"][kind] = row["unscored"].get(kind, 0) + 1
                    continue
                exact, loose = found(kind, value, hay)
                row["fields"] += 1
                row["found"] += exact
                row["loose"] += loose
            out.append(row)
    return rows


def _summary(rows: list[dict]) -> dict[str, float]:
    fields = sum(r["fields"] for r in rows)
    asked = [r for r in rows if r["fields"]]
    pct = lambda a, b: round(100.0 * a / b, 2) if b else 0.0  # noqa: E731
    return {
        "questions": len(asked),
        "fields": fields,
        "values found": pct(sum(r["found"] for r in rows), fields),
        "values found, loose": pct(sum(r["loose"] for r in rows), fields),
        "questions complete": pct(sum(r["found"] == r["fields"] for r in asked), len(asked)),
        "questions complete, loose": pct(sum(r["loose"] == r["fields"] for r in asked), len(asked)),
    }


def cmd_score() -> int:
    items, prof = load_items(), json.loads((out_dir() / "profile.json").read_text())
    stems = sorted(prof)
    items = [it for it in items if it["source_file"] in prof]
    pages = {n: {s: _pages(n, s) for s in stems} for n in CONFIGS}
    rows = score_items(items, {n: {s: "\n\n".join(p) for s, p in docs.items()} for n, docs in pages.items()})
    a, b = rows["plain"], rows["corrected"]
    groups = {
        "all": lambda r: True,
        "form questions": lambda r: r["form"],
        "page with a text layer": lambda r: prof[r["doc"]]["text_pages"] > 0,
        "page without a text layer": lambda r: prof[r["doc"]]["text_pages"] == 0,
        **{f"domain {d}": (lambda r, d=d: r["domain"] == d) for d in sorted({r["domain"] for r in a})},
    }
    report: dict[str, Any] = {"dataset": f"{REPO}@{REVISION}", "groups": {}}
    for title, keep in groups.items():
        sa, sb = _summary([r for r in a if keep(r)]), _summary([r for r in b if keep(r)])
        report["groups"][title] = {"plain": sa, "corrected": sb}
        print(f"\n{title}: {sa['questions']} questions, {sa['fields']} scored fields")
        for measure in list(sa)[2:]:
            print(f"  {measure:28}{sa[measure]:8.2f}{sb[measure]:8.2f}{sb[measure] - sa[measure]:+8.2f}")
    differ = sum(pa != pb for s in stems for pa, pb in zip(pages["plain"][s], pages["corrected"][s], strict=False))
    differ += sum(len(pages["plain"][s]) != len(pages["corrected"][s]) for s in stems)
    per_doc: dict[str, list[int]] = {}
    only_a = only_b = 0
    for ra, rb in zip(a, b, strict=True):
        acc = per_doc.setdefault(ra["doc"], [0, 0, 0])
        acc[0] += ra["loose"]
        acc[1] += rb["loose"]
        acc[2] += ra["fields"]
        only_a += max(ra["loose"] - rb["loose"], 0)
        only_b += max(rb["loose"] - ra["loose"], 0)
    gain, p = bootstrap_gain([tuple(v) for v in per_doc.values()])
    unscored: dict[str, int] = {}
    for r in a:
        for k, n in r["unscored"].items():
            unscored[k] = unscored.get(k, 0) + n
    report.update({"pages": sum(len(v) for v in pages["plain"].values()), "pages_differ": differ,
                   "questions_more_found_plain": only_a, "questions_more_found_corrected": only_b,
                   "loose_gain": round(gain, 3), "p_gain": p, "unscored_fields": unscored})  # fmt: skip
    print(f"\npages {report['pages']}, markdown differs on {differ}")
    print(f"loose fields found by one only: plain {only_a}, corrected {only_b}; gain {gain:+.3f}, P(gain>0) {p:.2f}")
    print(f"fields not scored (presence cannot judge them): {unscored}")
    (out_dir() / "score.json").write_text(json.dumps({"report": report, "rows": rows}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forms_outside", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["plan", "write", "score"])
    args = parser.parse_args(argv)
    return {"plan": cmd_plan, "write": cmd_write, "score": cmd_score}[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())

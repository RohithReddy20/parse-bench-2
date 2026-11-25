"""olmOCR-bench (Allen AI) as an outside test: plain OCRia-1 against OCRia-1 plus the PDF corrections.

    python -m evalkit.external.olmocr_bench fetch      # pinned dataset into <data root>/external/olmocr-bench/data
    python -m evalkit.external.olmocr_bench describe   # PDFs, pages, text layers and tests per category
    python -m evalkit.external.olmocr_bench plan       # manifest of page reads not in the cache, both configs
    python -m evalkit.external.olmocr_bench write      # candidate folders docparse_plain and docparse_corrected
    python -m evalkit.external.olmocr_bench textlayer  # a trivial candidate: each page's PDF text layer as text
    python -m evalkit.external.olmocr_bench score      # the pinned scorer on both candidates, side by side

The dataset is data: it is read, never executed. The scorer is `olmocr.bench.benchmark` from the pinned
`olmocr` wheel, run with `python -I` from its own venv, set up once by hand (E = <data root>/external/olmocr-bench):

    uv venv --python 3.12 $E/venv
    uv pip install --python $E/venv/bin/python --no-deps olmocr==0.4.27
    uv pip install --python $E/venv/bin/python beautifulsoup4==4.15.0 fuzzysearch==0.8.1 numpy==2.5.5 \
        pillow==13.0.0 pypdf==7.1.1 rapidfuzz==3.14.6 tqdm==4.70.1 playwright==1.63.0  # what olmocr.bench imports

The bench code imports Playwright at load time; the math files render equations with KaTeX in headless Chromium,
and `score` leaves them out when no Chromium for the pinned Playwright is already on the machine (no install).
The scorer reads `<dir>/*.jsonl`, `<dir>/pdfs/**.pdf` and one sub-folder per candidate, so it runs on a view
folder `runs/ext-olmocr/bench` that links to the data and holds the candidates: the download is never written to.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from evalkit import paths

REPO = "allenai/olmOCR-bench"
REVISION = "54a96a6fb6a2bd3b297e59869491db4d3625b711"  # main on 2026-11-25 (last commit 2026-02-19)
OLMOCR_VERSION = "0.4.27"  # PyPI wheel sha256 4c54b77d1e5dd487bcd60be26728dc07aa74f7fb3947bae4c697776bd500f480
PLAYWRIGHT_VERSION = "1.63.0"  # math tests render with KaTeX in headless Chromium (build 1243)
MATH_CATEGORIES = ("arxiv_math", "old_scans_math")  # the files whose tests render equations in a browser
OVERRIDES = ["layout=none"]  # layout boxes are not scored here
CANDIDATES = {"docparse_plain": "ocria_own.yaml", "docparse_corrected": "own_best.yaml"}
HAS_TEXT_LAYER = 50  # non-space characters, as in evalkit.pagestats

ROOT = Path(__file__).resolve().parents[3]


def ext_dir() -> Path:
    return paths.data_root() / "external" / "olmocr-bench"


def data_dir() -> Path:
    return ext_dir() / "data" / REVISION / "bench_data"


def scorer_python() -> Path:
    return ext_dir() / "venv" / "bin" / "python"


def out_dir() -> Path:
    return paths.runs_dir() / "ext-olmocr"


# ---- pure helpers -------------------------------------------------------------------------------------------


def md_name(pdf_rel: str, page: int, repeat: int = 1) -> str:
    """Candidate file for one page, as the scorer matches it: `<pdf path without .pdf>_pg<page>_repeat<n>.md`.

    `pdf_rel` is the PDF's path under `pdfs/` (`tables/x.pdf`); `page` counts from 1.
    """
    if not pdf_rel.endswith(".pdf"):
        raise ValueError(f"not a pdf path: {pdf_rel!r}")
    return f"{pdf_rel[:-4]}_pg{page}_repeat{repeat}.md"


def category_dirs(rows: list[dict]) -> set[str]:
    """The `pdfs/` sub-folders a category file's tests point at."""
    return {row["pdf"].split("/")[0] for row in rows}


_SUMMARY = "Final Summary with 95% Confidence Intervals:"
_CANDIDATE = re.compile(r"^(\S+)\s*: Average Score: (.+?) \(average of per-JSONL scores\)$")
_FILE = re.compile(r"^\s+(\S+)\s*: ([\d.]+)% \((\d+)/(\d+) tests\)$")
_FAIL = re.compile(r"^\s+\[FAIL\] Test (.+?) on .+? page \d+ average pass ratio: ([\d.]+) ")


def parse_summary(log: str) -> dict[str, dict]:
    """Candidate -> {"files": {jsonl name: (passed, total)}, "printed": the scorer's own overall line}."""
    if _SUMMARY not in log:
        raise ValueError("no final summary in the scorer output")
    out: dict[str, dict] = {}
    current = None
    for line in log.split(_SUMMARY, 1)[1].splitlines():
        if m := _CANDIDATE.match(line):
            current = out.setdefault(m.group(1), {"files": {}, "printed": m.group(2).strip()})
        elif current is not None and (m := _FILE.match(line)):
            current["files"][m.group(1)] = (int(m.group(3)), int(m.group(4)))
    return out


def failed_tests(log: str) -> set[str]:
    """Ids of the tests the scorer reports as failed (one repeat per page, so a ratio below 1 is a fail)."""
    return {m.group(1) for line in log.splitlines() if (m := _FAIL.match(line)) and float(m.group(2)) < 0.5}


def overall(files: dict[str, tuple[int, int]]) -> float:
    """The scorer's overall score: the mean of the per-file pass rates (the per-PDF baseline file included)."""
    rates = [p / t for p, t in files.values() if t]
    return 100 * sum(rates) / len(rates) if rates else 0.0


def side_by_side(results: dict[str, dict[str, tuple[int, int]]]) -> str:
    """Per-file pass rates (and tests passed) of each candidate, overall first; with two, the change."""
    names = list(results)
    two = len(names) == 2
    lines = [f"{'category':24}{'tests':>7}" + "".join(f"{n:>24}" for n in names) + (f"{'change':>9}" if two else "")]
    totals = [overall(results[n]) for n in names]
    lines.append(f"{'overall':24}{'':>7}" + "".join(f"{v:24.2f}" for v in totals)
                 + (f"{totals[1] - totals[0]:+9.2f}" if two else ""))  # fmt: skip
    for file in sorted({f for r in results.values() for f in r}):
        cells = [results[n].get(file, (0, 0)) for n in names]
        rates = [100 * p / t if t else 0.0 for p, t in cells]
        lines.append(f"{file.removesuffix('.jsonl'):24}{max(t for _, t in cells):7d}"
                     + "".join(f"{r:16.2f} ({p:>5d})" for r, (p, _) in zip(rates, cells, strict=True))
                     + (f"{rates[1] - rates[0]:+9.2f}" if two else ""))  # fmt: skip
    return "\n".join(lines)


def changed_pages(a: Path, b: Path) -> tuple[int, int]:
    """(pages whose markdown differs, pages in both) between two candidate folders."""
    fa = {p.relative_to(a) for p in a.rglob("*.md")}
    fb = {p.relative_to(b) for p in b.rglob("*.md")}
    both = fa & fb
    return sum((a / r).read_bytes() != (b / r).read_bytes() for r in both), len(both)


# ---- data ---------------------------------------------------------------------------------------------------


def fetch() -> Path:
    from evalkit.fetch import _snapshot

    patterns = ["bench_data/*.jsonl", "bench_data/pdfs/**/*.pdf", "README.md", "eval.yaml"]
    _snapshot(data_dir().parent, repo_id=REPO, repo_type="dataset", revision=REVISION, allow_patterns=patterns,
              token=False)  # fmt: skip
    return data_dir()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def categories(data: Path, math: bool = True) -> dict[str, list[dict]]:
    """Category (jsonl stem) -> its test rows."""
    out = {p.stem: read_jsonl(p) for p in sorted(data.glob("*.jsonl"))}
    return {k: v for k, v in out.items() if math or k not in MATH_CATEGORIES}


def list_pdfs(data: Path, dirs: set[str] | None = None) -> list[str]:
    pdfs = data / "pdfs"
    rels = sorted(p.relative_to(pdfs).as_posix() for p in pdfs.rglob("*.pdf"))
    return [r for r in rels if dirs is None or r.split("/")[0] in dirs]


def _page_facts(path: str) -> list[int]:
    """Non-space characters in each page's text layer."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        counts = []
        for page in pdf:
            text = page.get_textpage().get_text_range()
            counts.append(sum(1 for ch in text if not ch.isspace() and ch not in "\x00￾"))
        return counts
    finally:
        pdf.close()


def describe(data: Path, workers: int = 8) -> dict:
    cats = categories(data)
    pdfs = list_pdfs(data)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        facts = dict(zip(pdfs, pool.map(_page_facts, [str(data / "pdfs" / r) for r in pdfs], chunksize=16),
                         strict=True))  # fmt: skip
    out = {"revision": REVISION, "pdfs": len(pdfs), "categories": {}}
    for name, rows in cats.items():
        dirs = category_dirs(rows)
        mine = [r for r in pdfs if r.split("/")[0] in dirs]
        types: dict[str, int] = {}
        for row in rows:
            types[row["type"]] = types.get(row["type"], 0) + 1
        pages = [len(facts[r]) for r in mine]
        out["categories"][name] = {
            "pdf_dirs": sorted(dirs),
            "pdfs": len(mine),
            "pdfs_with_tests": len({row["pdf"] for row in rows}),
            "pages": sum(pages),
            "multi_page_pdfs": sum(n > 1 for n in pages),
            "pdfs_with_text_layer": sum(any(c >= HAS_TEXT_LAYER for c in facts[r]) for r in mine),
            "tests": len(rows),
            "tests_by_type": dict(sorted(types.items())),
            "tests_on_page_gt_1": sum(row.get("page", 1) > 1 for row in rows),
        }
    out["pages"] = sum(len(v) for v in facts.values())
    out["pdfs_with_text_layer"] = sum(any(c >= HAS_TEXT_LAYER for c in v) for v in facts.values())
    out["tests"] = sum(c["tests"] for c in out["categories"].values())
    return out


# ---- the parser, through the cache --------------------------------------------------------------------------


def _configs():
    from docparse.core.config import Config

    return {name: Config.load(ROOT / "configs" / file, OVERRIDES) for name, file in CANDIDATES.items()}


def _plan_one(job: tuple[str, list[str], str, str]) -> dict:
    from docparse.core.config import Config
    from docparse.core.errors import ParseError
    from docparse.models.cache import Cache
    from docparse.pipeline import parse
    from evalkit.run import _PlanningClient

    path, config_json, cache_root, manifest = job
    client = _PlanningClient(Cache(Path(cache_root)), Path(manifest))
    data = Path(path).read_bytes()
    out = {"pdf": path, "pending": 0}
    for cfg in config_json:
        try:
            out["pending"] += len(parse(data, Config.model_validate_json(cfg), client).pending)
        except ParseError as exc:
            out["error"] = f"ParseError: {exc}"
    out["misses"] = client.misses
    return out


def plan(data: Path, manifest: Path, workers: int = 8) -> dict:
    """Write every page read either config needs and the cache lacks into one manifest (each call once)."""
    from docparse.models.calls import ModelCall
    from docparse.models.manifest import ManifestWriter

    writer = ManifestWriter(manifest)
    before = len(writer)
    configs = [c.model_dump_json() for c in _configs().values()]
    pdfs = list_pdfs(data)
    jobs = [(str(data / "pdfs" / r), configs, str(paths.cache_dir()), str(manifest)) for r in pdfs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_plan_one, jobs, chunksize=4))
    new = 0
    errors = []
    for res in results:
        if "error" in res:
            errors.append((res["pdf"], res["error"]))
        for call_json, meta in res["misses"]:
            new += writer.add(ModelCall.model_validate(call_json), {}, meta)
    pending_docs = sum(res["pending"] > 0 for res in results)
    return {"pdfs": len(pdfs), "pdfs_waiting": pending_docs, "new_calls": new, "manifest_calls": len(writer),
            "manifest_calls_before": before, "errors": errors}  # fmt: skip


def _write_one(job: tuple[str, str, dict[str, str], str]) -> dict:
    from docparse.core.config import Config
    from docparse.core.errors import ParseError
    from docparse.models.cache import Cache
    from docparse.models.calls import CacheMiss
    from docparse.models.client import ReplayClient
    from docparse.pipeline import parse
    from docparse.write.markdown import write_page

    path, rel, config_json, bench = job
    data = Path(path).read_bytes()
    client = ReplayClient(Cache(paths.cache_dir()))
    out: dict = {"pdf": rel}
    for name, cfg in config_json.items():
        try:
            doc = parse(data, Config.model_validate_json(cfg), client).document
        except CacheMiss:
            out.setdefault("missing", []).append(name)
            continue
        except ParseError as exc:  # an unreadable PDF scores as an empty page rather than stopping the scorer
            out.setdefault("errors", []).append([name, str(exc)])
            target = Path(bench) / name / md_name(rel, 1)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("", encoding="utf-8")
            continue
        for page in doc.pages:
            target = Path(bench) / name / md_name(rel, page.index + 1)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(write_page(page), encoding="utf-8")
    return out


def write(data: Path, bench: Path, workers: int = 8) -> dict:
    configs = {name: c.model_dump_json() for name, c in _configs().items()}
    jobs = [(str(data / "pdfs" / r), r, configs, str(bench)) for r in list_pdfs(data)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_write_one, jobs, chunksize=4))
    missing = [r["pdf"] for r in results if r.get("missing")]
    errors = [r for r in results if r.get("errors")]
    (out_dir() / "write_errors.json").write_text(json.dumps({"missing": missing, "errors": errors}, indent=1))
    return {"pdfs": len(jobs), "missing_reads": len(missing), "parse_errors": len(errors)}


def _textlayer_one(job: tuple[str, str, str]) -> None:
    import pypdfium2 as pdfium

    path, rel, folder = job
    pdf = pdfium.PdfDocument(path)
    try:
        for i, page in enumerate(pdf):
            text = page.get_textpage().get_text_range().replace("\r\n", "\n").replace("\x00", "")
            target = Path(folder) / md_name(rel, i + 1)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    finally:
        pdf.close()


def textlayer(data: Path, folder: Path, workers: int = 8) -> int:
    pdfs = list_pdfs(data)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_textlayer_one, [(str(data / "pdfs" / r), r, str(folder)) for r in pdfs], chunksize=16))
    return len(pdfs)


# ---- scoring ------------------------------------------------------------------------------------------------


def browser_ok() -> bool:
    """Can the scorer's Playwright start its headless Chromium? (Nothing is installed if it cannot.)"""
    code = "from playwright.sync_api import sync_playwright as s\nwith s() as p: p.chromium.launch().close()"
    try:
        run = subprocess.run([str(scorer_python()), "-I", "-c", code], capture_output=True, timeout=120,
                             cwd=out_dir())  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return False
    return run.returncode == 0


def build_view(data: Path, bench: Path, math: bool) -> list[str]:
    """Link the category files and their PDF folders into `bench` (candidates live beside them)."""
    cats = categories(data, math)
    (bench / "pdfs").mkdir(parents=True, exist_ok=True)
    for old in [*bench.glob("*.jsonl"), *(bench / "pdfs").iterdir()]:
        if old.is_symlink():
            old.unlink()
    for name, rows in cats.items():
        (bench / f"{name}.jsonl").symlink_to(data / f"{name}.jsonl")
        for d in category_dirs(rows):
            if not (bench / "pdfs" / d).exists():
                (bench / "pdfs" / d).symlink_to(data / "pdfs" / d, target_is_directory=True)
    return sorted(cats)


def run_scorer(bench: Path, candidate: str, log: Path) -> str:
    cmd = [str(scorer_python()), "-I", "-m", "olmocr.bench.benchmark", "--dir", str(bench), "--candidate", candidate,
           "--bootstrap_samples", "1000"]  # fmt: skip
    run = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir())
    log.write_text(run.stdout + "\n--- stderr ---\n" + run.stderr, encoding="utf-8")
    if run.returncode != 0:
        raise RuntimeError(f"scorer failed on {candidate} ({run.returncode}); see {log}")
    return run.stdout


def score(data: Path, bench: Path, names: tuple[str, ...], math: bool | None) -> int:
    if math is None:
        math = browser_ok()
        if not math:
            print(f"no headless Chromium for Playwright {PLAYWRIGHT_VERSION}: {', '.join(MATH_CATEGORIES)} left out")
    cats = build_view(data, bench, math)
    print("categories:", ", ".join(cats))
    logs = out_dir() / "scores"
    logs.mkdir(parents=True, exist_ok=True)
    results, fails = {}, {}
    for name in names:
        text = run_scorer(bench, name, logs / f"{name}.log")
        results[name] = parse_summary(text)[name]
        fails[name] = failed_tests(text)
    print(side_by_side({n: results[n]["files"] for n in names}))
    for name in names:
        print(f"scorer's own line, {name}: {results[name]['printed']}")
    if len(names) == 2:
        a, b = fails[names[0]], fails[names[1]]
        print(f"tests failed by {names[0]} only: {len(a - b)}; by {names[1]} only: {len(b - a)}")
        changed, both = changed_pages(bench / names[0], bench / names[1])
        print(f"pages whose markdown differs: {changed} of {both}")
    summary = {"math": math, "categories": cats, "results": results}
    (logs / f"summary-{'-vs-'.join(names)}.json").write_text(json.dumps(summary, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit.external.olmocr_bench")
    parser.add_argument("command", choices=["fetch", "describe", "plan", "write", "textlayer", "score"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--candidates", default=",".join(CANDIDATES), help="score: candidate folders to compare")
    parser.add_argument("--math", choices=["auto", "yes", "no"], default="auto",
                        help="score: include the browser-rendered math files (auto: when Chromium starts)")  # fmt: skip
    args = parser.parse_args(argv)
    out_dir().mkdir(parents=True, exist_ok=True)
    bench = out_dir() / "bench"
    if args.command == "fetch":
        print(fetch())
        return 0
    data = data_dir()
    if not data.is_dir():
        print(f"no data at {data}; run `fetch` first", file=sys.stderr)
        return 1
    if args.command == "describe":
        facts = describe(data, args.workers)
        (out_dir() / "describe.json").write_text(json.dumps(facts, indent=1))
        print(json.dumps(facts, indent=1))
    elif args.command == "plan":
        res = plan(data, out_dir() / "manifest", args.workers)
        folder = out_dir() / "manifest"
        size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        print(f"{res['pdfs']} PDFs, {res['pdfs_waiting']} waiting on page reads; {res['new_calls']} new calls "
              f"(manifest now {res['manifest_calls']} calls, {size / 1e6:.1f} MB, at {folder})")  # fmt: skip
        for pdf, err in res["errors"][:20]:
            print("  error", pdf, err)
    elif args.command == "write":
        res = write(data, bench, args.workers)
        print(res)
        return 1 if res["missing_reads"] else 0
    elif args.command == "textlayer":
        print(textlayer(data, bench / "textlayer", args.workers), "PDFs written to", bench / "textlayer")
    else:
        names = tuple(args.candidates.split(","))
        math = None if args.math == "auto" else args.math == "yes"
        return score(data, bench, names, math)
    return 0


if __name__ == "__main__":
    sys.exit(main())

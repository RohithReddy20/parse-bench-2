"""Self-contained Colab GPU runner for full-benchmark `datalab-to/chandra-ocr-2` evaluation via direct HF download.

Reads `/content/jobs.jsonl` (lightweight ~100-500 KB job table with exact local cache keys),
downloads only the needed PDFs/images directly from Hugging Face onto Colab at datacenter speed,
renders each page at 300 DPI (with exact upright rotation + 2048px LANCZOS resize), serves
`datalab-to/chandra-ocr-2` via vLLM 0.34.0, and writes `/content/results.jsonl.gz`.

Usage (via `colab exec --env HF_REPO=... --env HF_REV=... --env HF_PATTERN=... -f serve/colab_hf_chandra.py`):
"""

from __future__ import annotations

import base64
import glob
import gzip
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = "datalab-to/chandra-ocr-2"
REVISION = "af93b47dba1b47b6640c86ccf487ed2260ab9a09"
CHANDRA_MAX_SIDE = 2048
MAX_PIXELS = 80_000_000

_CHANDRA_TAGS = (
    "['math', 'br', 'i', 'b', 'u', 'del', 'sup', 'sub', 'table', 'tr', 'td', 'p', 'th', 'div', 'pre', "
    "'h1', 'h2', 'h3', 'h4', 'h5', 'ul', 'ol', 'li', 'input', 'a', 'span', 'img', 'hr', 'tbody', "
    "'small', 'caption', 'strong', 'thead', 'big', 'code', 'chem']"
)
_CHANDRA_ATTRS = (
    "['class', 'colspan', 'rowspan', 'display', 'checked', 'type', 'border', 'value', 'style', "
    "'href', 'alt', 'align', 'data-bbox', 'data-label']"
)
CHANDRA_PROMPT = (
    "OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the "
    "data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are "
    "normalized 0-1000. The data-label attribute is the label for the block.\n\n"
    "Use the following labels:\n"
    "- Caption\n- Footnote\n- Equation-Block\n- List-Group\n- Page-Header\n- Page-Footer\n- Image\n"
    "- Section-Header\n- Table\n- Text\n- Complex-Block\n- Code-Block\n- Form\n- Table-Of-Contents\n"
    "- Figure\n- Chemical-Block\n- Diagram\n- Bibliography\n- Blank-Page\n\n"
    f"Only use these tags {_CHANDRA_TAGS}, and these attributes {_CHANDRA_ATTRS}.\n\n"
    "Guidelines:\n"
    "* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in "
    "KaTeX-compatible LaTeX. Use display for block math.\n"
    "* Tables: Use colspan and rowspan attributes to match table structure.\n"
    "* Formatting: Maintain consistent formatting with the image, including spacing, indentation, "
    "subscripts/superscripts, and special characters.\n"
    "* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out "
    "the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, "
    "and convert diagrams to mermaid.\n"
    "* Forms: Mark checkboxes and radio buttons properly.\n"
    "* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line "
    "breaks within paragraphs, but only when absolutely necessary to maintain meaning.\n"
    "* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.\n"
    "* Lists: Preserve indents and proper list markers.\n"
    "* Use the simplest possible HTML structure that accurately represents the content of the block.\n"
    "* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should "
    "be correct and natural."
)

_IMG_FRAC = (0.7, 0.5, 0.35)
_OVER_TOTAL_RE = re.compile(r"passed (\d+) input tokens and requested (\d+) output")
_PDFIUM_LOCK = threading.Lock()


def _install_deps() -> None:
    print("=== GPU ===", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=True)
    print("=== Installing vLLM 0.34.0 + pypdfium2 + deps ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "vllm==0.34.0",
            "pypdfium2",
            "httpx",
            "pillow",
            "huggingface_hub",
        ],
        check=True,
    )


def _download_datasets(jobs: list[dict[str, Any]], data_root: Path) -> dict[tuple[str, str], Path]:
    from huggingface_hub import snapshot_download

    chunks = sorted(glob.glob("/content/data_part_*"))
    local_dir = Path("/content/local_data")
    if chunks:
        tar_path = Path("/content/local_data.tar")
        if not local_dir.exists():
            print(f"=== Reassembling {len(chunks)} uploaded data chunks -> {tar_path} ===", flush=True)
            with tar_path.open("wb") as out:
                for c in chunks:
                    out.write(Path(c).read_bytes())
            local_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(local_dir)
            print(f"Extracted local data ({tar_path.stat().st_size / 1e6:.1f} MB) -> {local_dir}", flush=True)
        return {(j["hf_repo"], j["hf_rev"]): local_dir for j in jobs}

    repos = sorted({(j["hf_repo"], j["hf_rev"], j.get("hf_pattern", "*")) for j in jobs})
    roots: dict[tuple[str, str], Path] = {}
    for repo, rev, pattern in repos:
        dest = data_root / repo.replace("/", "__") / rev[:8]
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / ".done"
        if not marker.exists():
            t0 = time.monotonic()
            patterns = [p.strip() for p in pattern.split(",") if p.strip()]
            print(f"=== Downloading {repo}@{rev[:8]} ({patterns}) ===", flush=True)
            for attempt in range(1, 5):
                try:
                    snapshot_download(
                        repo_id=repo,
                        repo_type="dataset",
                        revision=rev,
                        local_dir=str(dest),
                        allow_patterns=patterns,
                        max_workers=8,
                        token=False,
                    )
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise
                    print(f"  retry {attempt} after error: {str(exc)[:120]}", flush=True)
                    time.sleep(10 * attempt)
            marker.write_text("ok", encoding="utf-8")
            print(f"  downloaded in {time.monotonic() - t0:.1f}s -> {dest}", flush=True)
        roots[(repo, rev)] = dest
    return roots


def _render_job_png(path: Path, page_idx: int, turn: int, dpi: int = 300) -> bytes:
    import pypdfium2 as pdfium
    from PIL import Image

    data = path.read_bytes()
    if data.lstrip()[:5].startswith(b"%PDF-"):
        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(data)
            try:
                page = pdf[page_idx]
                w_pt, h_pt = page.get_size()
                scale = dpi / 72.0
                if w_pt * scale * h_pt * scale > MAX_PIXELS:
                    scale = (MAX_PIXELS / max(1.0, w_pt * h_pt)) ** 0.5
                im = page.render(scale=scale).to_pil().convert("RGB")
            finally:
                pdf.close()
    else:
        im = Image.open(io.BytesIO(data)).convert("RGB")

    if turn in (90, 180, 270):
        how = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[turn]
        im = im.transpose(how)
    if max(im.width, im.height) > CHANDRA_MAX_SIDE:
        s = CHANDRA_MAX_SIDE / max(im.width, im.height)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _downscale_png(png: bytes, factor: float) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    im = im.resize((max(1, int(im.width * factor)), max(1, int(im.height * factor))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _wait_ready(proc: subprocess.Popen, base: str, timeout: float = 1200.0) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with code {proc.returncode} before ready")
        try:
            if httpx.get(base + "/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise RuntimeError("vllm did not become ready in time")


def _stop_vllm(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> int:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    jobs_file = Path(os.environ.get("JOBS_FILE", "/content/jobs.jsonl"))
    out_file = Path(os.environ.get("RESULTS_FILE", "/content/results.jsonl"))
    out_gz = Path(os.environ.get("RESULTS_GZ", "/content/results.jsonl.gz"))
    data_root = Path("/content/hf_data")

    assert jobs_file.exists(), f"Missing {jobs_file}"
    _install_deps()

    import httpx

    jobs = [json.loads(ln) for ln in jobs_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    done_keys: set[str] = set()
    if out_file.exists():
        for ln in out_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.strip():
                try:
                    row = json.loads(ln)
                    if "key" in row and "output" in row:
                        done_keys.add(row["key"])
                except json.JSONDecodeError:
                    pass
    todo = [j for j in jobs if j["key"] not in done_keys]
    print(f"=== Chandra-OCR-2 HF batch: total={len(jobs)} done={len(done_keys)} todo={len(todo)} ===", flush=True)

    roots = _download_datasets(todo, data_root)

    gpu_mem_mb = int(
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[0]
    )
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=True
    ).stdout.strip()

    max_len = 16384 if gpu_mem_mb >= 32000 else 12288
    workers = 32 if gpu_mem_mb >= 32000 else 20
    extra_vllm = ["--gpu-memory-utilization", "0.92", "--max-num-seqs", str(workers)]
    cmd = [
        "vllm",
        "serve",
        REPO,
        "--revision",
        REVISION,
        "--served-model-name",
        "vlm",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_len),
        "--limit-mm-per-prompt",
        '{"image": 1}',
        "--port",
        "8000",
        *extra_vllm,
    ]
    log_path = Path("/content/vllm_chandra.log")
    log_fh = log_path.open("w", encoding="utf-8")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    completed = 0
    failed = 0
    lock = threading.Lock()
    cap_mt = 8192 if gpu_mem_mb >= 32000 else 6144

    try:
        try:
            _wait_ready(proc, "http://localhost:8000")
        except Exception:
            log_fh.flush()
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:])
            print(f"vLLM startup failed; tail:\n{tail}", flush=True)
            raise
        print(f"vLLM ready in {time.monotonic() - t0:.1f}s on {gpu_name} (workers={workers})", flush=True)
        client = httpx.Client(base_url="http://localhost:8000/v1", timeout=600.0)

        def _call_one(job: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
            key = job["key"]
            path = roots[(job["hf_repo"], job["hf_rev"])] / job["rel"]
            if not path.exists():
                return key, None, f"missing file {path}"
            try:
                original_png = _render_job_png(
                    path, int(job["page_idx"]), int(job.get("turn", 0)), int(job.get("dpi", 300))
                )
            except Exception as exc:
                return key, None, f"render error: {exc}"
            png = original_png
            mt = cap_mt
            scale = 1.0
            started = time.monotonic()
            for attempt in range(4):
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
                    },
                    {"type": "text", "text": CHANDRA_PROMPT},
                ]
                body = {
                    "model": "vlm",
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.0,
                    "max_tokens": mt,
                }
                try:
                    resp = client.post("/chat/completions", json=body)
                except Exception as exc:
                    if attempt < 3:
                        time.sleep(2)
                        continue
                    return key, None, str(exc)
                if resp.status_code == 400 and attempt < 3:
                    m_total = _OVER_TOTAL_RE.search(resp.text)
                    if m_total:
                        inp = int(m_total.group(1))
                        if max_len - int(inp * 1.4) < 512:
                            scale = _IMG_FRAC[attempt]
                            png = _downscale_png(original_png, scale)
                        mt = max(512, min(mt // 2, max_len - int(inp * 1.4)))
                    else:
                        scale = _IMG_FRAC[attempt]
                        png = _downscale_png(original_png, scale)
                        mt = max(512, mt // 2)
                    continue
                if resp.status_code != 200:
                    return key, None, f"HTTP {resp.status_code}: {resp.text[:300]}"
                data = resp.json()
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                out = {
                    "answer": choice["message"].get("content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "max_tokens": mt,
                    "image_scale": scale,
                    "_tokens_in": usage.get("prompt_tokens"),
                    "_tokens_out": usage.get("completion_tokens"),
                    "_gpu_seconds": round(time.monotonic() - started, 3),
                    "_gpu_type": gpu_name,
                }
                return key, out, None
            return key, None, "exhausted retries"

        run_t0 = time.monotonic()
        with out_file.open("a", encoding="utf-8") as fh:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_call_one, j) for j in todo]
                for fut in as_completed(futs):
                    key, out, err = fut.result()
                    with lock:
                        if err is not None:
                            failed += 1
                            print(f"  [FAIL] {key[:12]}: {err}", flush=True)
                        else:
                            fh.write(json.dumps({"key": key, "ok": True, "output": out}, ensure_ascii=False) + "\n")
                            fh.flush()
                            completed += 1
                            if completed % 50 == 0 or completed == len(todo):
                                dt = time.monotonic() - run_t0
                                rate = completed / max(dt, 1e-6)
                                msg = (
                                    f"  [chandra2] {completed}/{len(todo)} done "
                                    f"({failed} fail) in {dt:.1f}s ({rate:.2f} pg/s)"
                                )
                                print(msg, flush=True)
        client.close()
    finally:
        _stop_vllm(proc)
        log_fh.close()

    with out_file.open("rb") as src, gzip.open(out_gz, "wb") as dst:
        dst.write(src.read())
    print(
        f"ALL DONE: completed={completed} failed={failed} gz={out_gz} ({out_gz.stat().st_size / 1e6:.2f} MB)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Self-contained Colab H100/A100-80GB runner for the Frontier 2026 VLM Pilot-175 bakeoff.

Runs 3 orthogonal frontier open-weight models sequentially with clean vLLM teardown between models:
  1. lightonai/LightOnOCR-3-4B (BF16): `grounding` on all 175 pages + `""` plain on 60 olmOCR pages
  2. infly/Infinity-Parser2-Pro (35B-A3B MoE, FP8 on H100): official `doc2md` on all 175 pages
  3. Qwen/Qwen3.5-27B-FP8 (dense 27B FP8, reasoning):
       - 40 chart pages (full page + zoomed Figure crop + PDF vector measurements + thinking)
       - 40 form pages (25 PB2 dev forms + 15 RealDoc scanned forms + thinking)
       - 60 hard pages (30 olmOCR old_scans + 30 PB2 dev hard tables/text)

All outputs are flushed incrementally to `/content/out/*.jsonl` and bundled into `/content/frontier_results.tar.gz`.
"""

from __future__ import annotations

import base64
import gc
import glob
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

WORK = Path("/content/pilot175")
OUT = Path("/content/out")

CHART_SYSTEM = (
    "You are an expert document data analyst extracting exact, complete structured HTML tables from charts and "
    "figures on document pages."
)
CHART_PROMPT = """\
Extract the exact data table(s) for EVERY chart and EVERY subplot/panel shown on this document page.
You are provided with:
- The full page image (and, when present, a zoomed high-resolution crop of the chart region).
- Exact measurements extracted directly from the PDF's vector drawing paths (trust these over eyeballing pixels \
wherever they correspond to a plotted bar, line, or marker):
<measurements>
{hint}
</measurements>
- Current draft HTML table(s) from an initial parser (may have missing panels, missing/swapped labels, or eyeballed \
numbers — verify everything visually against the image and vector measurements):
<draft_tables>
{draft}
</draft_tables>

Rules:
1. Preserve EVERY chart, EVERY panel/subplot, EVERY category, and EVERY series shown on the page. Never drop a panel \
or truncate categories.
2. Wherever <measurements> gives exact vector values for a plotted series, use those exact values (with the chart's \
printed units and signs). Read the image for any values, labels, titles, units, or categories not in <measurements>.
3. Match legend colors to series names carefully. Use exact printed labels, axis units, and chart title(s). Never use \
placeholders like "Series 1" or "Value" when an axis label or legend name is printed.
4. Do NOT invent Total, Average, or Sum rows/columns that are not actually plotted in the chart.
5. Output ONLY the exact chart title line(s) and clean HTML <table>...</table> (using <table>, <tr>, <th>, <td>, \
colspan/rowspan as needed). No markdown code fences or commentary."""

FORM_PROMPT = """\
Transcribe this document form page completely and accurately into clean Markdown (using HTML <table> for multi-column \
tabular sections or grids).

Strict rules:
1. Preserve every section heading, field label, printed instruction, and filled-in value (typed or handwritten) \
exactly as written, paired cleanly with its corresponding field label.
2. Mark every checkbox or radio button accurately from visual inspection: `[x]` ONLY if checked/ticked/filled/crossed, \
`[ ]` if unchecked/empty. Double-check every single box carefully.
3. Keep blank/unfilled fields explicitly blank — NEVER hallucinate or invent values for empty fields, and NEVER shift \
a value into the wrong field.
4. For character-box / comb fields (e.g., postal codes, dates, ID numbers written one character per box), read the \
complete assembled value cleanly (e.g., `M2 9YT`, `2025-04-12`) instead of splitting characters into separate cells.
5. Do not add commentary or code fences; output only the faithful document transcription."""

HARD_OCR_PROMPT = """\
Transcribe this document page completely and faithfully into clean Markdown in natural reading order.

Rules:
1. Convert every table into clean, complete HTML `<table>` using `<thead>`, `<tbody>`, `<th>`, `<td>`, `colspan`, and \
`rowspan` accurately matching every column header and cell value. Never dump a table as plain text.
2. Convert mathematical equations to clean LaTeX (`$...$` inline, `$$...$$` display).
3. For degraded, faint, handwritten, or historical scans: transcribe every legible word accurately in natural reading \
order without skipping lines, hallucinating, or repeating text.
4. Omit decorative image descriptions (do not write paragraphs describing logos or photographs).
5. Output only the clean document transcription with no code fences or commentary."""

DEFAULT_INFPARSER2_PROMPT = (
    "You are an AI assistant specialized in converting PDF images to Markdown format. "
    "Please convert the entire document page accurately into clean Markdown, using HTML tables for all tables "
    "and LaTeX for mathematical equations. Preserve natural reading order."
)

_OVER_TOTAL_RE = re.compile(r"passed (\d+) input tokens and requested (\d+) output")
_IMG_FRAC = (0.75, 0.55, 0.4)


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check)


def _gpu_info() -> tuple[str, int]:
    name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=True
    ).stdout.strip()
    mem = int(
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()[0]
    )
    return name, mem


def _unpack_payload() -> list[dict[str, Any]]:
    if not (WORK / "jobs.json").exists():
        chunks = sorted(glob.glob("/content/pilot175_part_*"))
        assert chunks, "No /content/pilot175_part_* files uploaded!"
        tar_path = Path("/content/pilot175.tar")
        with tar_path.open("wb") as out:
            for c in chunks:
                out.write(Path(c).read_bytes())
        WORK.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(WORK)
        print(f"Unpacked {len(chunks)} chunks ({tar_path.stat().st_size/1e6:.1f} MB) -> {WORK}", flush=True)
    return json.loads((WORK / "jobs.json").read_text(encoding="utf-8"))


def _fetch_infparser2_prompt() -> str:
    """Try to fetch the official verbatim doc2md prompt from infly-ai/INF-MLLM; fall back cleanly."""
    repo_dir = Path("/content/INF-MLLM")
    try:
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/infly-ai/INF-MLLM.git", str(repo_dir)],
                check=True,
                timeout=90,
            )
        # Search for doc2md prompt definition
        for py in sorted((repo_dir / "Infinity-Parser2").rglob("*.py")):
            txt = py.read_text(encoding="utf-8", errors="replace")
            if "doc2md" in txt.lower() and ("prompt" in txt.lower() or "markdown" in txt.lower()):
                print(f"[infparser2] inspecting {py.relative_to(repo_dir)} ({len(txt)} chars)", flush=True)
                # Look for triple-quoted string assigned near doc2md
                for m in re.finditer(r'(?:DOC2MD|doc2md)[^=\n]*[:=]\s*[rfb]*("""|\'\'\')([\s\S]{40,3000}?)\1', txt):
                    cand = m.group(2).strip()
                    print(f"[infparser2] FOUND official doc2md prompt ({len(cand)} chars): {cand[:200]!r}", flush=True)
                    return cand
                # Also look for dict entry "doc2md": "..."
                for m in re.finditer(r'["\']doc2md["\']\s*:\s*["\']([^"\']{30,2000})["\']', txt):
                    cand = m.group(1).encode("utf-8").decode("unicode_escape").strip()
                    print(f"[infparser2] FOUND official doc2md prompt ({len(cand)} chars): {cand[:200]!r}", flush=True)
                    return cand
                print(f"[infparser2] snippet from {py.name}:\n{txt[:1500]}", flush=True)
    except Exception as exc:
        print(f"[infparser2] Could not fetch INF-MLLM prompt: {exc}", flush=True)
    return DEFAULT_INFPARSER2_PROMPT


def _wait_ready(proc: subprocess.Popen, base: str, timeout: float = 1200.0) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with code {proc.returncode}")
        try:
            if httpx.get(base + "/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise RuntimeError("vllm startup timeout")


def _stop_vllm(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    gc.collect()
    time.sleep(6)


def _img_b64(path: Path, scale: float = 1.0) -> str:
    raw = path.read_bytes()
    if scale == 1.0:
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    from PIL import Image

    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.strip():
            try:
                r = json.loads(ln)
                if r.get("ok") and "uid" in r:
                    done.add(r["uid"])
            except Exception:
                pass
    return done


def _run_batch(
    tag: str,
    repo: str,
    vllm_args: list[str],
    requests: list[dict[str, Any]],
    out_file: Path,
    workers: int = 16,
    max_model_len: int = 24576,
    gpu_name: str = "",
) -> dict[str, Any]:
    import httpx
    from huggingface_hub import HfApi

    done = _load_done(out_file)
    todo = [r for r in requests if r["uid"] not in done]
    print(
        f"\n==== [{tag}] repo={repo} total={len(requests)} done={len(done)} todo={len(todo)} ====",
        flush=True,
    )
    if not todo:
        return {"tag": tag, "done": 0, "skipped": len(done), "failed": 0}

    try:
        sha = HfApi().model_info(repo).sha
    except Exception:
        sha = "unknown"
    print(f"[{tag}] resolved HF sha={sha}", flush=True)

    cmd = [
        "vllm",
        "serve",
        repo,
        "--served-model-name",
        "vlm",
        "--port",
        "8000",
        "--max-model-len",
        str(max_model_len),
        *vllm_args,
    ]
    log_path = OUT / f"vllm_{tag}.log"
    log_fh = log_path.open("w", encoding="utf-8")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    completed = 0
    failed = 0
    lock = threading.Lock()

    try:
        try:
            _wait_ready(proc, "http://localhost:8000", timeout=1200.0)
        except Exception as exc:
            log_fh.flush()
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
            print(f"[{tag}] vLLM startup FAILED: {exc}\n{tail}", flush=True)
            return {"tag": tag, "repo": repo, "sha": sha, "error": str(exc), "done": 0, "failed": len(todo)}

        print(f"[{tag}] vLLM ready in {time.monotonic()-t0:.1f}s on {gpu_name} (workers={workers})", flush=True)
        client = httpx.Client(base_url="http://localhost:8000/v1", timeout=600.0)

        def _call_one(req: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
            uid = req["uid"]
            imgs = [WORK / p for p in req["images"]]
            prompt = req.get("prompt", "")
            sys_prompt = req.get("system", "")
            mt = int(req.get("max_tokens", 8192))
            temp = float(req.get("temperature", 0.0))
            enable_thinking = req.get("enable_thinking", None)
            scale = 1.0
            started = time.monotonic()

            for attempt in range(4):
                content: list[dict[str, Any]] = [
                    {"type": "image_url", "image_url": {"url": _img_b64(p, scale)}} for p in imgs
                ]
                if prompt:
                    content.append({"type": "text", "text": prompt})
                messages: list[dict[str, Any]] = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": content})
                body: dict[str, Any] = {
                    "model": "vlm",
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": mt,
                }
                if enable_thinking is not None:
                    body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
                try:
                    resp = client.post("/chat/completions", json=body)
                except Exception as exc:
                    if attempt < 3:
                        time.sleep(2)
                        continue
                    return uid, None, str(exc)
                if resp.status_code == 400 and attempt < 3:
                    m = _OVER_TOTAL_RE.search(resp.text)
                    scale = _IMG_FRAC[attempt]
                    if m:
                        inp = int(m.group(1))
                        mt = max(1024, min(mt, max_model_len - int(inp * 1.15)))
                    else:
                        mt = max(1024, mt // 2)
                    continue
                if resp.status_code != 200:
                    return uid, None, f"HTTP {resp.status_code}: {resp.text[:300]}"
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]
                usage = data.get("usage") or {}
                out = {
                    "answer": msg.get("content") or "",
                    "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "tokens_in": usage.get("prompt_tokens"),
                    "tokens_out": usage.get("completion_tokens"),
                    "gpu_seconds": round(time.monotonic() - started, 3),
                    "repo": repo,
                    "sha": sha,
                    "bucket": req.get("bucket"),
                    "item_id": req.get("item_id"),
                    "subtask": req.get("subtask", tag),
                }
                return uid, out, None
            return uid, None, "exhausted retries"

        run_t0 = time.monotonic()
        with out_file.open("a", encoding="utf-8") as fh:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_call_one, r) for r in todo]
                for fut in as_completed(futs):
                    uid, out, err = fut.result()
                    with lock:
                        if err is not None:
                            failed += 1
                            print(f"  [FAIL {tag}] {uid}: {err}", flush=True)
                        else:
                            fh.write(json.dumps({"uid": uid, "ok": True, "output": out}, ensure_ascii=False) + "\n")
                            fh.flush()
                            completed += 1
                            if completed % 25 == 0 or completed == len(todo):
                                dt = time.monotonic() - run_t0
                                print(
                                    f"  [{tag}] {completed}/{len(todo)} done ({failed} fail) in {dt:.1f}s "
                                    f"({completed/max(dt,1e-6):.2f} pg/s)",
                                    flush=True,
                                )
        client.close()
    finally:
        _stop_vllm(proc)
        log_fh.close()

    return {
        "tag": tag,
        "repo": repo,
        "sha": sha,
        "done": completed,
        "failed": failed,
        "sec": round(time.monotonic() - t0, 1),
    }


def main() -> int:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    gpu_name, gpu_mem = _gpu_info()
    print(f"=== GPU: {gpu_name} ({gpu_mem} MB) ===", flush=True)
    print("=== Installing vLLM 0.34.0 + deps ===", flush=True)
    _run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.34.0", "httpx", "pillow", "huggingface_hub"])

    records = _unpack_payload()
    print(f"Loaded {len(records)} Pilot-175 records", flush=True)
    only = {s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()}
    summary: list[dict[str, Any]] = []

    # ── Stage 1: LightOnOCR-3-4B (BF16, grounding + olmOCR plain) ──
    if not only or "lighton" in only:
        reqs_ground = [
            {
                "uid": f"g::{r['uid']}",
                "bucket": r["bucket"],
                "item_id": r["item_id"],
                "subtask": "grounding",
                "images": [r["page_img"]],
                "prompt": "grounding",
                "max_tokens": 8192 if gpu_mem >= 32000 else 6144,
                "temperature": 0.0,
            }
            for r in records
        ]
        reqs_plain = [
            {
                "uid": f"p::{r['uid']}",
                "bucket": r["bucket"],
                "item_id": r["item_id"],
                "subtask": "plain",
                "images": [r["page_img"]],
                "prompt": "",
                "max_tokens": 8192 if gpu_mem >= 32000 else 6144,
                "temperature": 0.0,
            }
            for r in records
            if r["is_olm"]
        ]
        lo_workers = 32 if gpu_mem >= 32000 else 16
        lo_len = 18000 if gpu_mem >= 32000 else 12288
        summary.append(
            _run_batch(
                tag="lighton3_4b",
                repo="lightonai/LightOnOCR-3-4B",
                vllm_args=[
                    "--dtype", "bfloat16",
                    "--limit-mm-per-prompt", '{"image": 1}',
                    "--default-chat-template-kwargs", '{"enable_thinking": false}',
                    "--mm-processor-cache-gb", "0",
                    "--no-enable-prefix-caching",
                    "--gpu-memory-utilization", "0.90",
                    "--max-num-seqs", str(lo_workers),
                ],
                requests=reqs_ground + reqs_plain,
                out_file=OUT / "lighton3_4b.jsonl",
                workers=lo_workers,
                max_model_len=lo_len,
                gpu_name=gpu_name,
            )
        )

    # ── Stage 2: Infinity-Parser2-Pro (35B-A3B MoE, W4A16 fits A100-40GB & 80GB!) ──
    if (not only or "infparser2" in only) and gpu_mem >= 32000:
        inf_prompt = _fetch_infparser2_prompt()
        reqs_inf = [
            {
                "uid": r["uid"],
                "bucket": r["bucket"],
                "item_id": r["item_id"],
                "subtask": "doc2md",
                "images": [r["page_img"]],
                "prompt": inf_prompt,
                "max_tokens": 8192,
                "temperature": 0.0,
                "enable_thinking": False,
            }
            for r in records
        ]
        vllm_inf_args = [
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--reasoning-parser", "qwen3",
            "--limit-mm-per-prompt", '{"image": 1}',
            "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "16",
        ]
        res_inf = _run_batch(
            tag="infparser2_pro",
            repo="spectator2026/Infinity-Parser2-Pro-AWQ-W4A16",
            vllm_args=vllm_inf_args,
            requests=reqs_inf,
            out_file=OUT / "infparser2_pro.jsonl",
            workers=16,
            max_model_len=20480,
            gpu_name=gpu_name,
        )
        summary.append(res_inf)
    elif (not only or "infparser2" in only) and gpu_mem < 32000:
        print(f"[infparser2_pro] Skipping 35B MoE W4A16 on {gpu_mem} MB GPU (needs >=32 GB VRAM)", flush=True)

    # ── Stage 2b: Infinity-Parser2-Flash (2B RLVR, 4.43 GB BF16, 86.0 olmOCR — fits L4 & runs in ~2 min!) ──
    if "infflash" in only:
        inf_prompt = _fetch_infparser2_prompt()
        reqs_inff = [
            {
                "uid": r["uid"],
                "bucket": r["bucket"],
                "item_id": r["item_id"],
                "subtask": "doc2md",
                "images": [r["page_img"]],
                "prompt": inf_prompt,
                "max_tokens": 6144,
                "temperature": 0.0,
                "enable_thinking": False,
            }
            for r in records
        ]
        summary.append(
            _run_batch(
                tag="infparser2_flash",
                repo="infly/Infinity-Parser2-Flash",
                vllm_args=[
                    "--trust-remote-code",
                    "--dtype", "bfloat16",
                    "--reasoning-parser", "qwen3",
                    "--limit-mm-per-prompt", '{"image": 1}',
                    "--gpu-memory-utilization", "0.90",
                    "--max-num-seqs", "16",
                    "--no-enable-prefix-caching",
                ],
                requests=reqs_inff,
                out_file=OUT / "infparser2_flash.jsonl",
                workers=16,
                max_model_len=16384,
                gpu_name=gpu_name,
            )
        )

    # ── Stage 3: Frontier Reasoning Tier-2 (Qwen3.5-27B AWQ-4bit on L4/A100, with 30B-A3B fallback) ──
    if not only or "qwen35" in only or "qwen35_l4" in only:
        q_mt = 12288 if gpu_mem >= 70000 else (8192 if gpu_mem >= 32000 else 4096)
        reqs_qwen: list[dict[str, Any]] = []
        # 3a. 40 Chart pages (full page + zoomed Figure crop + vector measurements + thinking!)
        for r in records:
            if not r["is_chart"]:
                continue
            imgs = [r["page_img"]] + ([r["crop_img"]] if r.get("crop_img") else [])
            prompt = CHART_PROMPT.format(
                hint=(r.get("vector_hint") or "").strip() or "(none)",
                draft=(r.get("draft_tables") or "").strip() or "(none)",
            )
            reqs_qwen.append(
                {
                    "uid": f"chart::{r['uid']}",
                    "bucket": r["bucket"],
                    "item_id": r["item_id"],
                    "subtask": "chart",
                    "images": imgs,
                    "system": CHART_SYSTEM,
                    "prompt": prompt,
                    "max_tokens": q_mt,
                    "temperature": 0.0,
                    "enable_thinking": True,
                }
            )
        # 3b. 40 Form pages (25 PB2 dev forms + 15 RealDoc scanned forms + thinking!)
        for r in records:
            if not r["is_form"]:
                continue
            reqs_qwen.append(
                {
                    "uid": f"form::{r['uid']}",
                    "bucket": r["bucket"],
                    "item_id": r["item_id"],
                    "subtask": "form",
                    "images": [r["page_img"]],
                    "prompt": FORM_PROMPT,
                    "max_tokens": q_mt,
                    "temperature": 0.0,
                    "enable_thinking": True,
                }
            )
        # 3c. 60 Hard pages (30 olmOCR old_scans + 30 PB2 dev hard tables/text)
        for r in records:
            if not r["is_hard"]:
                continue
            reqs_qwen.append(
                {
                    "uid": f"hard::{r['uid']}",
                    "bucket": r["bucket"],
                    "item_id": r["item_id"],
                    "subtask": "hard",
                    "images": [r["page_img"]],
                    "prompt": HARD_OCR_PROMPT,
                    "max_tokens": q_mt,
                    "temperature": 0.0,
                    "enable_thinking": False,
                }
            )
        if gpu_mem >= 70000:
            # 80-96 GB Blackwell/Hopper (Colab G4 RTX PRO 6000 96GB / H100 80GB / CloudRift RTX PRO 6000):
            # Official Qwen/Qwen3.5-27B-FP8 (30.87 GB, native FP8 Tensor Cores, ~BF16 fidelity, huge KV headroom!)
            q_repo = "Qwen/Qwen3.5-27B-FP8"
            q_workers = 12
            q_len = 24576
            q_args = [
                "--trust-remote-code",
                "--dtype", "bfloat16",
                "--reasoning-parser", "qwen3",
                "--limit-mm-per-prompt", '{"image": 2}',
                "--gpu-memory-utilization", "0.90",
                "--max-num-seqs", str(q_workers),
            ]
        elif gpu_mem >= 32000:
            q_repo = "Qwen/Qwen3.5-27B-GPTQ-Int4"
            q_workers = 6
            q_len = 14336
            q_args = [
                "--trust-remote-code",
                "--dtype", "bfloat16",
                "--reasoning-parser", "qwen3",
                "--limit-mm-per-prompt", '{"image": 2}',
                "--gpu-memory-utilization", "0.94",
                "--max-num-seqs", str(q_workers),
                "--enforce-eager",
            ]
        else:
            # 24 GB L4 (Ada SM89 native FP8 KV cache): cyankiwi/Qwen3.5-27B-AWQ-4bit (18.68 GiB, lossless BF16 visual)
            q_repo = "cyankiwi/Qwen3.5-27B-AWQ-4bit"
            q_workers = 2
            q_len = 8192
            q_args = [
                "--trust-remote-code",
                "--dtype", "bfloat16",
                "--reasoning-parser", "qwen3",
                "--kv-cache-dtype", "fp8",
                "--enforce-eager",
                "--no-enable-prefix-caching",
                "--mm-processor-cache-gb", "0",
                "--limit-mm-per-prompt", '{"image": 2}',
                "--gpu-memory-utilization", "0.95",
                "--max-num-seqs", str(q_workers),
            ]
        res_q = _run_batch(
            tag="qwen35_27b_int4",
            repo=q_repo,
            vllm_args=q_args,
            requests=reqs_qwen,
            out_file=OUT / "qwen35_27b_int4.jsonl",
            workers=q_workers,
            max_model_len=q_len,
            gpu_name=gpu_name,
        )
        summary.append(res_q)
        # Automatic L4 fallback if 27B AWQ still OOMs: Qwen3-VL-30B-A3B-Thinking-AWQ (16.66 GiB, 3B active MoE!)
        if res_q.get("error") and gpu_mem < 32000:
            print(
                "[qwen35_27b_int4] OOM/error on L4 -> falling back to QuantTrio/Qwen3-VL-30B-A3B-Thinking-AWQ!",
                flush=True,
            )
            res_fb = _run_batch(
                tag="qwen3vl_30b_a3b",
                repo="QuantTrio/Qwen3-VL-30B-A3B-Thinking-AWQ",
                vllm_args=[
                    "--trust-remote-code",
                    "--dtype", "bfloat16",
                    "--reasoning-parser", "qwen3",
                    "--kv-cache-dtype", "fp8",
                    "--enforce-eager",
                    "--no-enable-prefix-caching",
                    "--mm-processor-cache-gb", "0",
                    "--limit-mm-per-prompt", '{"image": 2}',
                    "--gpu-memory-utilization", "0.94",
                    "--max-num-seqs", "4",
                ],
                requests=reqs_qwen,
                out_file=OUT / "qwen3vl_30b_a3b.jsonl",
                workers=4,
                max_model_len=10240,
                gpu_name=gpu_name,
            )
            summary.append(res_fb)

    # ── Stage 3b: Qwen3.5-9B-AWQ-4bit (8.45 GiB, ultra-fast Qwen3.5 thinking comparison on charts + forms) ──
    if "qwen35_9b" in only:
        reqs_9b = []
        for r in records:
            if r["is_chart"]:
                imgs = [r["page_img"]] + ([r["crop_img"]] if r.get("crop_img") else [])
                reqs_9b.append({
                    "uid": f"chart::{r['uid']}",
                    "bucket": r["bucket"],
                    "item_id": r["item_id"],
                    "subtask": "chart",
                    "images": imgs,
                    "system": CHART_SYSTEM,
                    "prompt": CHART_PROMPT.format(
                        hint=(r.get("vector_hint") or "").strip() or "(none)",
                        draft=(r.get("draft_tables") or "").strip() or "(none)",
                    ),
                    "max_tokens": 6144,
                    "temperature": 0.0,
                    "enable_thinking": True,
                })
            elif r["is_form"]:
                reqs_9b.append({
                    "uid": f"form::{r['uid']}", "bucket": r["bucket"], "item_id": r["item_id"], "subtask": "form",
                    "images": [r["page_img"]], "prompt": FORM_PROMPT,
                    "max_tokens": 6144, "temperature": 0.0, "enable_thinking": True,
                })
            elif r["is_hard"]:
                reqs_9b.append({
                    "uid": f"hard::{r['uid']}", "bucket": r["bucket"], "item_id": r["item_id"], "subtask": "hard",
                    "images": [r["page_img"]], "prompt": HARD_OCR_PROMPT,
                    "max_tokens": 6144, "temperature": 0.0, "enable_thinking": False,
                })
        summary.append(
            _run_batch(
                tag="qwen35_9b_awq",
                repo="cyankiwi/Qwen3.5-9B-AWQ-4bit",
                vllm_args=[
                    "--trust-remote-code",
                    "--dtype", "bfloat16",
                    "--reasoning-parser", "qwen3",
                    "--limit-mm-per-prompt", '{"image": 2}',
                    "--gpu-memory-utilization", "0.90",
                    "--max-num-seqs", "8",
                    "--no-enable-prefix-caching",
                ],
                requests=reqs_9b,
                out_file=OUT / "qwen35_9b_awq.jsonl",
                workers=8,
                max_model_len=14336,
                gpu_name=gpu_name,
            )
        )

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    out_tar = Path("/content/frontier_results.tar.gz")
    with tarfile.open(out_tar, "w:gz") as tf:
        for f in sorted(OUT.glob("*.jsonl")) + [OUT / "summary.json"]:
            if f.exists():
                tf.add(f, arcname=f.name)
    print(f"\nALL DONE -> {out_tar} ({out_tar.stat().st_size/1e6:.2f} MB)", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

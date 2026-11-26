"""Self-contained Colab A100 runner for the 3 pilot150 perception manifests (354 calls total, $0 Modal).

Runs sequentially on one A100 GPU with clean vLLM shutdown + VRAM release between models:
  1. exp1_ocria_crops     (238 calls) -> llamaindex/OCRia-1 @ dae83aa5
  2. exp2_qwen3vl_charts  ( 51 calls) -> Qwen/Qwen3-VL-8B-Instruct @ 0c351dd0
  3. exp3_olmocr2_scans   ( 65 calls) -> allenai/olmOCR-2-7B-1025 @ 7c6fa161

Resumable: skips any call key already present in `<manifest>/results.jsonl`, and flushes every result.

Usage on Colab VM:
    python run_colab_pilot.py --work-dir /content/pilot150 [--out-dir /content/drive/MyDrive/docparse/pilot150_results]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

JOBS: list[dict[str, Any]] = [
    {
        "name": "exp1_ocria_crops",
        "repo": "llamaindex/OCRia-1",
        "revision": "dae83aa51ecd0a8574d415d929a4d9e79348fd2a",
        "max_model_len": 32768,
        "system_prompt_file": "prompt.txt",
        "workers": 16,
    },
    {
        "name": "exp2_qwen3vl_charts",
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "max_model_len": 16384,
        "system_prompt_file": None,
        "workers": 16,
    },
    {
        "name": "exp3_olmocr2_scans",
        "repo": "allenai/olmOCR-2-7B-1025",
        "revision": "7c6fa1611328bd17dbf5f3a07509d2b6fa2eba4e",
        "max_model_len": 16384,
        "system_prompt_file": None,
        "workers": 16,
    },
    {
        "name": "exp2b_chart_verify",
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "max_model_len": 16384,
        "system_prompt_file": None,
        "workers": 16,
    },
    {
        "name": "exp4_chandra2_scans",
        "repo": "datalab-to/chandra-ocr-2",
        "revision": "af93b47dba1b47b6640c86ccf487ed2260ab9a09",
        "max_model_len": 16384,
        "system_prompt_file": None,
        "workers": 16,
    },
    {
        "name": "exp4b_chandra2_rest",
        "repo": "datalab-to/chandra-ocr-2",
        "revision": "af93b47dba1b47b6640c86ccf487ed2260ab9a09",
        "max_model_len": 16384,
        "system_prompt_file": None,
        "workers": 16,
    },
    {
        "name": "full_realdoc_chandra2",
        "repo": "datalab-to/chandra-ocr-2",
        "revision": "af93b47dba1b47b6640c86ccf487ed2260ab9a09",
        "max_model_len": 16384,
        "system_prompt_file": None,
        "workers": 32,
    },
    {
        "name": "full_olmocr_chandra2",
        "repo": "datalab-to/chandra-ocr-2",
        "revision": "af93b47dba1b47b6640c86ccf487ed2260ab9a09",
        "max_model_len": 16384,
        "system_prompt_file": None,
        "workers": 32,
    },
    {
        "name": "full_pb2_chandra2",
        "repo": "datalab-to/chandra-ocr-2",
        "revision": "af93b47dba1b47b6640c86ccf487ed2260ab9a09",
        "max_model_len": 16384,
        "system_prompt_file": None,
        "workers": 32,
    },
]

_IMG_FRAC = (0.7, 0.5, 0.35)
_OVER_TOTAL_RE = re.compile(r"passed (\d+) input tokens and requested (\d+) output")


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
    # Give CUDA driver a moment to release VRAM before launching next model
    time.sleep(5)


def _downscale_png(png: bytes, factor: float) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    im = im.resize((max(1, int(im.width * factor)), max(1, int(im.height * factor))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _load_done_keys(results_path: Path) -> set[str]:
    done: set[str] = set()
    if not results_path.exists():
        return done
    for line in results_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if "key" in row and "output" in row:
                done.add(row["key"])
        except json.JSONDecodeError:
            continue
    return done


def _run_one_job(job: dict[str, Any], manifest_dir: Path, results_path: Path) -> dict[str, Any]:
    import httpx
    from huggingface_hub import hf_hub_download

    manifest_file = manifest_dir / "manifest.jsonl"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_file}")
    entries = [json.loads(ln) for ln in manifest_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    done_keys = _load_done_keys(results_path)
    todo = [e for e in entries if e["key"] not in done_keys]
    print(
        f"\n=== [{job['name']}] repo={job['repo']} | total={len(entries)} done={len(done_keys)} todo={len(todo)} ===",
        flush=True,
    )
    if not todo:
        return {"name": job["name"], "total": len(entries), "done": 0, "skipped": len(done_keys), "failed": 0}

    default_sys = ""
    if job["system_prompt_file"]:
        default_sys = Path(hf_hub_download(job["repo"], job["system_prompt_file"], revision=job["revision"])).read_text(
            encoding="utf-8"
        )

    gpu_mem_mb = int(
        subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()[0]
    )
    max_len = int(job["max_model_len"])
    extra_vllm: list[str] = []
    if gpu_mem_mb < 32000:
        if "OCRia" in job["repo"] or "chandra" in job["repo"].lower():
            # 4B Qwen3.5 models (~8.5GB weights) fit with CUDA graphs + 24 concurrent seqs on 24GB L4
            max_len = min(max_len, 12288)
            extra_vllm = ["--gpu-memory-utilization", "0.92", "--max-num-seqs", "24"]
        else:
            # 7B-8B bf16 models (~16-17GB weights) need eager mode on 24GB L4
            max_len = min(max_len, 8192)
            extra_vllm = ["--gpu-memory-utilization", "0.93", "--max-num-seqs", "8", "--enforce-eager"]

    served = "vlm"
    cmd = [
        "vllm",
        "serve",
        job["repo"],
        "--revision",
        job["revision"],
        "--served-model-name",
        served,
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
    t0 = time.monotonic()
    log_path = manifest_dir / "vllm.log"
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    failed = 0
    completed = 0
    lock = threading.Lock()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.strip()

    try:
        try:
            _wait_ready(proc, "http://localhost:8000")
        except Exception:
            log_fh.flush()
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:])
            print(f"\n[{job['name']}] vLLM startup failed; last 50 lines of {log_path}:\n{tail}", flush=True)
            raise
        print(f"[{job['name']}] vLLM ready in {time.monotonic() - t0:.1f}s on {gpu_name}", flush=True)
        client = httpx.Client(base_url="http://localhost:8000/v1", timeout=600.0)
        blobs_dir = manifest_dir / "blobs"

        def _call_one(entry: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
            key = entry["key"]
            call = entry["call"]
            inputs = call["inputs"]
            params = call.get("params") or {}
            original_png = (blobs_dir / inputs[0]).read_bytes()
            user_text = (blobs_dir / inputs[1]).read_bytes().decode("utf-8") if len(inputs) > 1 else ""
            sys_prompt = str(params.get("system_prompt") or default_sys)
            # Allow full 8192 tokens on 40GB+ GPUs (A100), cap at 6144 on 24GB L4 to avoid stalling
            cap_mt = 8192 if gpu_mem_mb >= 32000 else 6144
            mt = min(int(params.get("max_tokens", cap_mt)), cap_mt)
            temp = float(params.get("temperature", 0.0))
            scale = float(params.get("image_scale", 1.0))
            text_first = bool(params.get("text_first", False))
            png = _downscale_png(original_png, scale) if scale != 1.0 else original_png
            started = time.monotonic()

            for attempt in range(4):
                img_part: dict[str, Any] = {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
                }
                text_part = {"type": "text", "text": user_text} if user_text else None
                content = (
                    [text_part, img_part]
                    if (text_first and text_part)
                    else ([img_part] + ([text_part] if text_part else []))
                )
                messages: list[dict[str, Any]] = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": content})
                body: dict[str, Any] = {
                    "model": served,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": mt,
                }
                if "enable_thinking" in params:
                    body["chat_template_kwargs"] = {"enable_thinking": bool(params["enable_thinking"])}
                if "repetition_penalty" in params:
                    body["repetition_penalty"] = float(params["repetition_penalty"])
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
                        if job["max_model_len"] - int(inp * 1.4) < 512:
                            scale = _IMG_FRAC[attempt]
                            png = _downscale_png(original_png, scale)
                        mt = max(512, min(mt // 2, job["max_model_len"] - int(inp * 1.4)))
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
        with results_path.open("a", encoding="utf-8") as fh:
            with ThreadPoolExecutor(max_workers=int(job["workers"])) as pool:
                futs = [pool.submit(_call_one, e) for e in todo]
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
                            if completed % 20 == 0 or completed == len(todo):
                                dt = time.monotonic() - run_t0
                                msg = f"  [{job['name']}] {completed}/{len(todo)} done ({failed} fail) in {dt:.1f}s"
                                print(msg, flush=True)
        client.close()
    finally:
        _stop_vllm(proc)
        log_fh.close()

    return {
        "name": job["name"],
        "total": len(entries),
        "done": completed,
        "skipped": len(done_keys),
        "failed": failed,
        "seconds": round(time.monotonic() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/pilot150"))
    parser.add_argument(
        "--tar", type=Path, default=None, help="Optional pilot150_manifests.tar to unpack into work-dir"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Optional extra dir (e.g. Drive) to copy results.jsonl"
    )
    parser.add_argument("--only", default="", help="Comma-separated subset of job names to run")
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.tar and args.tar.exists():
        print(f"Unpacking {args.tar} -> {args.work_dir}...", flush=True)
        with tarfile.open(args.tar, "r") as tf:
            tf.extractall(args.work_dir)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    selected = [j for j in JOBS if not only or j["name"] in only]
    summary = []
    for job in selected:
        mdir = args.work_dir / job["name"]
        rpath = (args.out_dir / f"{job['name']}_results.jsonl") if args.out_dir else (mdir / "results.jsonl")
        res = _run_one_job(job, mdir, rpath)
        summary.append(res)
        # Keep a copy beside manifest too
        if args.out_dir and rpath.exists():
            (mdir / "results.jsonl").write_bytes(rpath.read_bytes())

    out_tar = args.work_dir / "pilot150_results.tar.gz"
    with tarfile.open(out_tar, "w:gz") as tf:
        for job in selected:
            rp = args.work_dir / job["name"] / "results.jsonl"
            if rp.exists():
                tf.add(rp, arcname=f"{job['name']}_results.jsonl")
    print(f"\nALL DONE -> {out_tar}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

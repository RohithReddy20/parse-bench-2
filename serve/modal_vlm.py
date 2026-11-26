"""Generic vLLM multimodal manifest runner on Modal for cheap, capped pilot experiments.

Supports any OpenAI-compatible VLM served by vLLM:
  - llamaindex/OCRia-1           (image-only page/crop reads, reusing hf-cache)
  - Qwen/Qwen3-VL-8B-Instruct    (image + optional text prompt, e.g. vector-hinted chart reading)
  - allenai/olmOCR-2-7B-1025     (degraded scan / multi-column / math specialist)
  - datalab-to/chandra-ocr-2     (degraded scan / handwriting / layout specialist)
  - Qwen/Qwen3.5-27B-FP8         (Tier-2 multi-image chart/form/hard-table reads; needs an 80GB+ GPU)

Safety guardrail: refuses manifests larger than `--max-calls` (default 300) so an exploratory
pilot can never accidentally burn credits on a full 4,000-page sweep.

Usage:
    modal volume put docparse-manifests <local manifest dir> /<name>
    modal run serve/modal_vlm.py --manifest <name> --repo Qwen/Qwen3-VL-8B-Instruct
    modal volume get docparse-manifests /<name>/results.jsonl <local path>
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

import modal

VLLM_VERSION = os.environ.get("DOCPARSE_VLLM_VERSION", "0.34.0")
GPU = os.environ.get("DOCPARSE_MODAL_GPU", "A100-40GB")
ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(f"vllm/vllm-openai:v{VLLM_VERSION}")
    .entrypoint([])
    .run_commands(
        'ln -sf "$(command -v python3)" /usr/local/bin/python',
        "uv pip install --system 'zstandard>=0.22' || python3 -m pip install 'zstandard>=0.22'",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(str(ROOT / "src" / "docparse"), "/root/docparse")
    .add_local_dir(str(ROOT / "serve"), "/root/serve")
)
app = modal.App("docparse-vlm-pilot")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
manifests = modal.Volume.from_name("docparse-manifests", create_if_missing=True)


def _wait_ready(proc: subprocess.Popen, base: str, timeout: float = 1800.0) -> None:
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


def _downscale_png(png: bytes, factor: float) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    im = im.resize((max(1, int(im.width * factor)), max(1, int(im.height * factor))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _make_vlm_executor(base_url: str, served_model: str, default_system: str, gpu_type: str):
    import httpx

    client = httpx.Client(base_url=base_url.rstrip("/"), timeout=600.0)

    def execute(call, blobs: dict[str, bytes]) -> dict[str, Any]:
        # inputs[:n_images] are PNG images; an optional inputs[n_images] is the UTF-8 user prompt.
        n_images = int(call.params.get("n_images", 1))
        scale = float(call.params.get("image_scale", 1.0))
        pngs = [blobs[name] for name in call.inputs[:n_images]]
        if scale != 1.0:
            pngs = [_downscale_png(png, scale) for png in pngs]
        user_text = blobs[call.inputs[n_images]].decode("utf-8") if len(call.inputs) > n_images else ""
        sys_prompt = str(call.params.get("system_prompt") or default_system)
        max_tokens = int(call.params.get("max_tokens", 8192))
        temp = float(call.params.get("temperature", 0.0))

        started = time.monotonic()
        text_first = bool(call.params.get("text_first", False))
        for attempt in range(3):
            urls = ["data:image/png;base64," + base64.b64encode(png).decode() for png in pngs]
            img_parts: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
            text_part: dict[str, Any] | None = {"type": "text", "text": user_text} if user_text else None
            if text_first and text_part:
                content = [text_part, *img_parts]
            else:
                content = img_parts + ([text_part] if text_part else [])
            messages: list[dict[str, Any]] = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": content})
            body: dict[str, Any] = {
                "model": served_model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tokens,
            }
            if "enable_thinking" in call.params:
                body["chat_template_kwargs"] = {"enable_thinking": bool(call.params["enable_thinking"])}
            for key in ("top_p", "presence_penalty", "repetition_penalty"):
                if key in call.params:
                    body[key] = float(call.params[key])
            if "top_k" in call.params:
                body["top_k"] = int(call.params["top_k"])
            resp = client.post("/chat/completions", json=body)
            if resp.status_code == 400 and attempt < 2:
                pngs = [_downscale_png(png, 0.7) for png in pngs]
                max_tokens = max(512, max_tokens // 2)
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]
            usage = data.get("usage") or {}
            out: dict[str, Any] = {
                "answer": message.get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "max_tokens": max_tokens,
                "_tokens_in": usage.get("prompt_tokens"),
                "_tokens_out": usage.get("completion_tokens"),
                "_gpu_seconds": round(time.monotonic() - started, 3),
                "_gpu_type": gpu_type,
            }
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            if reasoning:
                out["reasoning"] = reasoning
            return out
        raise RuntimeError("request failed after 3 attempts")

    return execute


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/hf": hf_cache, "/manifests": manifests},
    timeout=3600,
)
def run(
    manifest: str,
    repo: str,
    revision: str = "",
    max_model_len: int = 16384,
    workers: int = 16,
    max_calls: int = 300,
    reasoning_parser: str = "",
    limit_mm_images: int = 1,
    gpu_memory_utilization: float = 0.90,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root")
    from huggingface_hub import hf_hub_download

    from docparse.models.manifest import read_manifest
    from serve.run_manifest import run_manifest

    vol_dir = Path("/manifests") / manifest
    mdir = vol_dir
    if not (mdir / "manifest.jsonl").exists() and (vol_dir / "manifest.tar").exists():
        mdir = Path("/tmp") / f"manifest_{manifest}"
        if mdir.exists():
            shutil.rmtree(mdir)
        mdir.mkdir(parents=True)
        with tarfile.open(vol_dir / "manifest.tar", "r") as tf:
            tf.extractall(mdir)
    calls = [call for call, _meta in read_manifest(mdir)]
    if not calls:
        raise RuntimeError(f"manifest {manifest!r} is empty")
    if len(calls) > max_calls:
        raise RuntimeError(f"safety cap: manifest has {len(calls)} calls > max_calls={max_calls}")
    if not reasoning_parser and ("Qwen3.5" in repo or "Qwen3-VL" in repo):
        reasoning_parser = "qwen3"
    # never serve fewer images per prompt than the manifest's multi-image calls need
    limit_mm_images = max(limit_mm_images, max(int(call.params.get("n_images", 1)) for call in calls))

    default_system = ""
    if repo == "llamaindex/OCRia-1":
        rev = revision or "dae83aa51ecd0a8574d415d929a4d9e79348fd2a"
        default_system = Path(hf_hub_download(repo, "prompt.txt", revision=rev)).read_text()

    served = "vlm"
    cmd = [
        "vllm",
        "serve",
        repo,
        "--served-model-name",
        served,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--limit-mm-per-prompt",
        json.dumps({"image": limit_mm_images}),
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(workers),
        "--port",
        "8000",
    ]
    if revision:
        cmd += ["--revision", revision]
    if reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]

    started = time.monotonic()
    proc = subprocess.Popen(cmd)
    try:
        _wait_ready(proc, "http://localhost:8000")
        ready = time.monotonic()
        gpu_name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
        ).stdout.strip()
        executor = _make_vlm_executor("http://localhost:8000/v1", served, default_system, gpu_name)
        stop = threading.Event()

        def keep_committing() -> None:
            while not stop.wait(120):
                try:
                    manifests.commit()
                except Exception as exc:
                    print(f"periodic commit failed: {exc}")

        threading.Thread(target=keep_committing, daemon=True).start()
        try:
            stats = run_manifest(mdir, vol_dir / "results.jsonl", executor, max_workers=workers)
        finally:
            stop.set()
    finally:
        proc.terminate()
        manifests.commit()
        hf_cache.commit()
    return {
        "repo": repo,
        "gpu": gpu_name,
        "startup_seconds": round(ready - started, 1),
        "run_seconds": round(stats.seconds, 1),
        "total": stats.total,
        "done": stats.done,
        "failed": stats.failed,
        "skipped": stats.skipped,
    }


@app.local_entrypoint()
def main(
    manifest: str,
    repo: str = "llamaindex/OCRia-1",
    revision: str = "",
    max_model_len: int = 16384,
    workers: int = 16,
    max_calls: int = 300,
    reasoning_parser: str = "",
    limit_mm_images: int = 1,
    gpu_memory_utilization: float = 0.90,
):
    print(
        run.remote(
            manifest,
            repo,
            revision,
            max_model_len,
            workers,
            max_calls,
            reasoning_parser,
            limit_mm_images,
            gpu_memory_utilization,
        )
    )

"""Manifest runner for a rented GPU VM (CloudRift, any SSH box): runs a manifest against an already running
vLLM OpenAI-compatible server, so one model load serves many manifests.

    # on the VM, once per model (one server per GPU for small models, tensor-parallel for big ones):
    CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.5-122B-A10B-FP8 --revision <sha> --served-model-name vlm \
        --dtype bfloat16 --max-model-len 24576 --limit-mm-per-prompt '{"image": 4}' --trust-remote-code \
        --tensor-parallel-size 2 --reasoning-parser qwen3 --max-num-seqs 32 --port 8000
    # then, per manifest (several base URLs are used round-robin):
    python -m serve.vm_run --manifest <dir> --results <dir>/results.jsonl --base-url http://localhost:8000/v1 \
        --workers 32 [--system-prompt-file prompt.txt]

The request is built exactly as `serve/modal_vlm.py` builds it (same message order, sampling keys, retry on HTTP
400 with a smaller image), so an answer is the same whichever runner produced it.
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from serve.run_manifest import run_manifest


def _downscale_png(png: bytes, factor: float) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    im = im.resize((max(1, int(im.width * factor)), max(1, int(im.height * factor))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def make_vlm_executor(base_urls: list[str], served_model: str, default_system: str, gpu_type: str):
    import httpx

    clients = [httpx.Client(base_url=u.rstrip("/"), timeout=1800.0) for u in base_urls]
    turn = itertools.cycle(range(len(clients)))
    lock = threading.Lock()

    def execute(call, blobs: dict[str, bytes]) -> dict[str, Any]:
        with lock:
            client = clients[next(turn)]
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
            if "seed" in call.params:
                body["seed"] = int(call.params["seed"])
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="serve.vm_run")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--base-url", action="append", required=True)
    ap.add_argument("--served-model", default="vlm")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--system-prompt-file", type=Path)
    args = ap.parse_args(argv)
    system = args.system_prompt_file.read_text(encoding="utf-8") if args.system_prompt_file else ""
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.strip().replace("\n", " + ")
    executor = make_vlm_executor(args.base_url, args.served_model, system, gpu)
    stats = run_manifest(args.manifest, args.results, executor, max_workers=args.workers)
    print(
        f"total={stats.total} done={stats.done} failed={stats.failed} skipped={stats.skipped} "
        f"seconds={stats.seconds:.1f}",
        flush=True,
    )
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Executors: functions that run one ModelCall against a model server and return its output dict.

`openai_chat_executor` sends OCRia-1 page calls to a vLLM OpenAI-compatible endpoint exactly as the ParseBench
harness's `ocria_1` provider does (run-llama/ParseBench@6e4479a, `OCRia1Provider._transcribe`): the model's
own system prompt, the page image as the only user content, greedy decoding, thinking off. When the image and
the output budget do not fit the context, it first shrinks the output budget, then the image.
"""

from __future__ import annotations

import base64
import io
import re
import time
from collections.abc import Callable
from typing import Any

import httpx

from docparse.models.calls import ModelCall

_IMG_FRAC = (0.6, 0.4, 0.25)  # image downscale per retry when the page alone overflows the context
_OVER_INPUT_RE = re.compile(r"max_tokens must be at least 1, got (-?\d+)")
_OVER_TOTAL_RE = re.compile(r"passed (\d+) input tokens and requested (\d+) output")


class ExecutorError(RuntimeError):
    pass


def _downscale(png: bytes, factor: float) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im = im.resize((max(1, int(im.width * factor)), max(1, int(im.height * factor))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def openai_chat_executor(
    base_url: str,
    served_model: str,
    system_prompt: str,
    max_model_len: int = 32768,
    api_key: str = "EMPTY",
    timeout: float = 1800.0,
    transport: httpx.BaseTransport | None = None,
    gpu_type: str | None = None,
) -> Callable[[ModelCall, dict[str, bytes]], dict[str, Any]]:
    client = httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
        transport=transport,
    )

    def execute(call: ModelCall, blobs: dict[str, bytes]) -> dict[str, Any]:
        (name,) = call.inputs
        original = blobs[name]
        max_tokens = int(call.params["max_tokens"])
        mt, png, image_scale = max_tokens, original, 1.0
        started = time.monotonic()
        for attempt in range(4):
            body = {
                "model": served_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()},
                            }
                        ],
                    },
                ],
                "temperature": float(call.params.get("temperature", 0.0)),
                "max_tokens": mt,
                "chat_template_kwargs": {"enable_thinking": bool(call.params.get("enable_thinking", False))},
            }
            resp = client.post("/chat/completions", json=body)
            if resp.status_code == 400 and attempt < 3:
                msg = resp.text
                m_total = _OVER_TOTAL_RE.search(msg)
                if m_total:
                    inp = int(m_total.group(1))
                    if max_model_len - int(inp * 1.5) < 512:
                        image_scale = _IMG_FRAC[attempt]
                        png = _downscale(original, image_scale)
                    mt = max(256, min(mt // 2, max_model_len - int(inp * 1.5)))
                    continue
                if _OVER_INPUT_RE.search(msg):
                    image_scale = _IMG_FRAC[attempt]
                    png = _downscale(original, image_scale)
                    mt = min(max_tokens, 2048)
                    continue
            if resp.status_code != 200:
                raise ExecutorError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            return {
                "answer": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "max_tokens": mt,
                "image_scale": image_scale,
                "_tokens_in": usage.get("prompt_tokens"),
                "_tokens_out": usage.get("completion_tokens"),
                "_gpu_seconds": round(time.monotonic() - started, 3),
                "_gpu_type": gpu_type,
            }
        raise ExecutorError("page did not fit the context after 4 attempts")

    return execute

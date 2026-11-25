"""OCRia-1 on Modal: run a manifest of page calls against a vLLM server in the same container.

Nothing here runs until the owner has seen `python -m docparse cache plan` for the manifest and agreed.

    modal volume put docparse-manifests <local manifest dir> /<name>          # upload calls and blobs
    modal run serve/modal_ocria.py --manifest <name> [--gpu A100-40GB] [--workers 32] [--limit N]
    modal volume get docparse-manifests /<name>/results.jsonl <local path>    # fetch answers
    # then: serve.run_manifest.import_results(manifest_dir, results_path, cache)

The container bills only while `run` executes (no idle endpoint). Weights are cached in the `hf-cache` volume.
The runtime string of the calls must be `vllm-<VLLM_VERSION>`; `run` refuses a manifest built for another one.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import modal

VLLM_VERSION = os.environ.get("DOCPARSE_VLLM_VERSION", "0.34.0")  # OCRia-1 needs vLLM 0.30 or later
OCRIA_REPO = "llamaindex/OCRia-1"
OCRIA_REVISION = "dae83aa51ecd0a8574d415d929a4d9e79348fd2a"
GPU = os.environ.get("DOCPARSE_MODAL_GPU", "A100-40GB")
SERVED_NAME = "ocria-1"
MAX_MODEL_LEN = 32768
ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(f"vllm/vllm-openai:v{VLLM_VERSION}")
    .entrypoint([])
    # the vLLM image ships `python3` only; Modal and our runner use the image's own interpreter, which already
    # has vllm, httpx, pydantic, pillow and huggingface_hub. Only zstandard (the cache format) is missing.
    .run_commands(
        'ln -sf "$(command -v python3)" /usr/local/bin/python',
        "uv pip install --system 'zstandard>=0.22' || python3 -m pip install 'zstandard>=0.22'",
        "python -c 'import vllm, httpx, pydantic, PIL, huggingface_hub, zstandard; print(vllm.__version__)'",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(str(ROOT / "src" / "docparse"), "/root/docparse")
    .add_local_dir(str(ROOT / "serve"), "/root/serve")
)
app = modal.App("docparse-ocria1")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
manifests = modal.Volume.from_name("docparse-manifests", create_if_missing=True)


def _wait_ready(proc: subprocess.Popen, base: str, timeout: float = 1800.0) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with code {proc.returncode} before it was ready")
        try:
            if httpx.get(base + "/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise RuntimeError("vllm did not become ready in time")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/hf": hf_cache, "/manifests": manifests},
    timeout=int(os.environ.get("DOCPARSE_MODAL_TIMEOUT", str(8 * 3600))),
)
def run(manifest: str, workers: int = 32, limit: int = 0) -> dict:
    """Serve OCRia-1 with vLLM, run the manifest at /manifests/<manifest>, append to its results.jsonl."""
    import sys

    sys.path.insert(0, "/root")
    import httpx
    from huggingface_hub import hf_hub_download

    from docparse.models.manifest import read_manifest
    from serve.executors import openai_chat_executor
    from serve.run_manifest import run_manifest

    mdir = Path("/manifests") / manifest
    runtime = f"vllm-{VLLM_VERSION}"
    calls = [call for call, _meta in read_manifest(mdir)]
    wrong = sorted(
        {c.runtime for c in calls if c.runtime != runtime}
        | {c.model.revision for c in calls if c.model.revision != OCRIA_REVISION}
    )
    if not calls or wrong:
        raise RuntimeError(f"manifest {manifest!r} is empty or was built for another runtime or revision: {wrong}")
    if limit:
        keep = {c.key() for c in calls[:limit]}
        sub = mdir.parent / f"{manifest}.first{limit}"
        sub.mkdir(exist_ok=True)
        lines = (mdir / "manifest.jsonl").read_text().splitlines()
        (sub / "manifest.jsonl").write_text("\n".join(ln for ln in lines if any(k in ln for k in keep)) + "\n")
        if not (sub / "blobs").exists():
            (sub / "blobs").symlink_to(mdir / "blobs")
        run_dir = sub
    else:
        run_dir = mdir

    prompt = Path(hf_hub_download(OCRIA_REPO, "prompt.txt", revision=OCRIA_REVISION)).read_text()
    started = time.monotonic()
    cmd = ["vllm", "serve", OCRIA_REPO, "--revision", OCRIA_REVISION, "--served-model-name", SERVED_NAME,
           "--dtype", "bfloat16", "--max-model-len", str(MAX_MODEL_LEN), "--limit-mm-per-prompt", '{"image": 1}',
           "--port", "8000"]  # fmt: skip
    proc = subprocess.Popen(cmd)
    try:
        _wait_ready(proc, "http://localhost:8000")
        ready = time.monotonic()
        version = httpx.get("http://localhost:8000/version", timeout=10).json().get("version")
        gpu_name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
        ).stdout.strip()
        executor = openai_chat_executor(
            "http://localhost:8000/v1", SERVED_NAME, prompt, max_model_len=MAX_MODEL_LEN, gpu_type=gpu_name
        )
        stop = threading.Event()

        def keep_committing() -> None:  # a timeout kill must not lose hours of answers
            while not stop.wait(120):
                try:
                    manifests.commit()
                except Exception as exc:
                    print(f"periodic commit failed: {exc}")

        threading.Thread(target=keep_committing, daemon=True).start()
        try:
            stats = run_manifest(run_dir, mdir / "results.jsonl", executor, max_workers=workers)
        finally:
            stop.set()
    finally:
        proc.terminate()
        manifests.commit()
        hf_cache.commit()
    return {
        "vllm_version": version, "gpu": gpu_name, "startup_seconds": round(ready - started, 1),
        "run_seconds": round(stats.seconds, 1), "total": stats.total, "done": stats.done, "failed": stats.failed,
        "skipped": stats.skipped,
    }  # fmt: skip


@app.local_entrypoint()
def main(manifest: str, workers: int = 32, limit: int = 0):
    print(run.remote(manifest, workers, limit))

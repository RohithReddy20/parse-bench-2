"""OCRia-1's attention pass on Modal: word positions for pages that have no PDF text layer.

Nothing here runs until the owner has seen `python -m docparse cache plan` for the manifest and agreed.

    modal volume put docparse-manifests <local manifest dir> /<name>
    modal run serve/modal_attention.py --manifest <name> [--limit N]
    modal volume get docparse-manifests /<name>/results.jsonl <local path>
    # then: serve.run_manifest.import_results(manifest_dir, results_path, cache)

One forward pass per page with Transformers (vLLM does not expose attention). The model's own grounding code
(`ocria_grounding`, from the pinned snapshot of the official repository) captures the attention; it is the one
piece of downloaded code this project runs.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import modal

OCRIA_REPO = "llamaindex/OCRia-1"
OCRIA_REVISION = "dae83aa51ecd0a8574d415d929a4d9e79348fd2a"
TRANSFORMERS = "5.22.2"  # the version the grounding code hooks into; part of the calls' runtime string
GPU = os.environ.get("DOCPARSE_MODAL_GPU", "A10G")
ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"transformers=={TRANSFORMERS}",
        "accelerate",
        "torch",
        "torchvision",
        "timm",
        "scipy",
        "pypdfium2",
        "huggingface_hub",
        "pillow",
        "numpy",
        "pydantic>=2.6",
        "zstandard>=0.22",
        "httpx",
    )  # fmt: skip
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(str(ROOT / "src" / "docparse"), "/root/docparse")
    .add_local_dir(str(ROOT / "serve"), "/root/serve")
)
app = modal.App("docparse-ocria1-attention")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
manifests = modal.Volume.from_name("docparse-manifests", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/hf": hf_cache, "/manifests": manifests},
    timeout=int(os.environ.get("DOCPARSE_MODAL_TIMEOUT", str(2 * 3600))),
)
def run(manifest: str, limit: int = 0) -> dict:
    """Load OCRia-1 once, run the manifest at /manifests/<manifest>, append to its results.jsonl."""
    import importlib
    import subprocess
    import time

    sys.path.insert(0, "/root")
    from huggingface_hub import snapshot_download

    from docparse.models.manifest import read_manifest
    from serve.executors import attention_executor
    from serve.run_manifest import run_manifest

    mdir = Path("/manifests") / manifest
    calls = [call for call, _meta in read_manifest(mdir)]
    wrong = sorted(
        {c.runtime for c in calls if c.runtime != f"transformers-{TRANSFORMERS}"}
        | {c.model.revision for c in calls if c.model.revision != OCRIA_REVISION}
    )
    if not calls or wrong:
        raise RuntimeError(f"manifest {manifest!r} is empty or was built for another runtime or revision: {wrong}")
    heads = calls[0].params["heads"]
    if any(c.params["heads"] != heads for c in calls):
        raise RuntimeError("the manifest mixes attention head sets")
    run_dir = mdir
    if limit:
        keep = {c.key() for c in calls[:limit]}
        run_dir = mdir.parent / f"{manifest}.first{limit}"
        run_dir.mkdir(exist_ok=True)
        lines = (mdir / "manifest.jsonl").read_text().splitlines()
        (run_dir / "manifest.jsonl").write_text("\n".join(ln for ln in lines if any(k in ln for k in keep)) + "\n")
        if not (run_dir / "blobs").exists():
            (run_dir / "blobs").symlink_to(mdir / "blobs")

    started = time.monotonic()
    model_dir = snapshot_download(OCRIA_REPO, revision=OCRIA_REVISION)
    sys.path.insert(0, model_dir)
    capture = importlib.import_module("ocria_grounding.capture")
    ground_mod = importlib.import_module("ocria_grounding.ground")
    blocks_mod = importlib.import_module("ocria_grounding.blocks")
    grounder = capture.Grounder(model_dir, [tuple(h) for h in heads], (Path(model_dir) / "prompt.txt").read_text())
    ready = time.monotonic()
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.strip()
    stop = threading.Event()

    def keep_committing() -> None:  # a timeout kill must not lose finished pages
        while not stop.wait(120):
            try:
                manifests.commit()
            except Exception as exc:
                print(f"periodic commit failed: {exc}")

    threading.Thread(target=keep_committing, daemon=True).start()
    try:
        executor = attention_executor(grounder, ground_mod, blocks_mod, heads, gpu_type=gpu_name)
        stats = run_manifest(run_dir, mdir / "results.jsonl", executor, max_workers=1)
    finally:
        stop.set()
        manifests.commit()
        hf_cache.commit()
    return {
        "gpu": gpu_name, "startup_seconds": round(ready - started, 1), "run_seconds": round(stats.seconds, 1),
        "total": stats.total, "done": stats.done, "failed": stats.failed, "skipped": stats.skipped,
    }  # fmt: skip


@app.local_entrypoint()
def main(manifest: str, limit: int = 0):
    print(run.remote(manifest, limit))

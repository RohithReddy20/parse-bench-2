"""Run Frontier Pilot-175 Tier-2 evaluations on Modal (H100 80GB / RTX-PRO-6000 / A100-80GB)
using the prebuilt vllm/vllm-openai image + persistent hf-cache and docparse-manifests volumes.

Usage:
    modal volume put --force docparse-manifests <local tar> /frontier_pilot175/frontier_pilot175.tar
    DOCPARSE_MODAL_GPU=H100 modal run serve/modal_frontier_pilot.py --only qwen35
    modal volume get --force docparse-manifests /frontier_pilot175/out/qwen35_27b_int4.jsonl <local path>
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path
from typing import Any

import modal

VLLM_VERSION = os.environ.get("DOCPARSE_VLLM_VERSION", "0.34.0")
GPU = os.environ.get("DOCPARSE_MODAL_GPU", "H100")
ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(f"vllm/vllm-openai:v{VLLM_VERSION}")
    .entrypoint([])
    .run_commands(
        'ln -sf "$(command -v python3)" /usr/local/bin/python',
        "uv pip install --system httpx pillow huggingface_hub || pip install -q httpx pillow huggingface_hub",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(str(ROOT / "serve" / "colab_frontier_pilot.py"), "/root/colab_frontier_pilot.py")
)
app = modal.App("docparse-frontier-pilot175")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
manifests = modal.Volume.from_name("docparse-manifests", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/hf": hf_cache, "/manifests": manifests},
    timeout=3600,
)
def run_pilot(only: str = "qwen35") -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root")
    import colab_frontier_pilot as cfp

    vol_dir = Path("/manifests/frontier_pilot175")
    tar_path = vol_dir / "frontier_pilot175.tar"
    if not tar_path.exists():
        raise RuntimeError(f"Missing {tar_path} in docparse-manifests volume")

    work = Path("/tmp/pilot175")
    out_dir = vol_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(work)

    cfp.WORK = work
    cfp.OUT = out_dir
    Path("/content").mkdir(parents=True, exist_ok=True)
    gpu_name, gpu_mem = cfp._gpu_info()
    print(f"=== Modal GPU: {gpu_name} ({gpu_mem} MB) ONLY={only} ===", flush=True)

    # Skip pip install inside prebuilt vLLM image
    cfp._run = lambda cmd, check=True: (  # type: ignore[assignment]
        subprocess.run(cmd, check=check) if "pip" not in cmd else None
    )
    cfp._unpack_payload = lambda: json.loads((work / "jobs.json").read_text(encoding="utf-8"))  # type: ignore[assignment]

    os.environ["ONLY"] = only
    stop = threading.Event()

    def _periodic_commit() -> None:
        while not stop.wait(90):
            try:
                manifests.commit()
                hf_cache.commit()
            except Exception as exc:
                print(f"[commit] warning: {exc}", flush=True)

    threading.Thread(target=_periodic_commit, daemon=True).start()
    try:
        rc = cfp.main()
    finally:
        stop.set()
        manifests.commit()
        hf_cache.commit()

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    return {"rc": rc, "gpu": gpu_name, "gpu_mem_mb": gpu_mem, "summary": summary}


@app.local_entrypoint()
def main(only: str = "qwen35") -> None:
    print(json.dumps(run_pilot.remote(only=only), indent=2))

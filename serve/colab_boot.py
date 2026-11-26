"""Boot script executed inside Colab GPU kernel via `colab exec -s pilot150 --timeout 3600 -f serve/colab_boot.py`."""

import glob
import os
import subprocess
import sys
from pathlib import Path

print("=== GPU ===", flush=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=True)

chunks = sorted(glob.glob("/content/part_*"))
tar_path = Path("/content/pilot150_manifests.tar")
if chunks:
    print(f"=== Reassembling {len(chunks)} manifest chunks -> {tar_path} ===", flush=True)
    with tar_path.open("wb") as out:
        for c in chunks:
            out.write(Path(c).read_bytes())
    print(f"Reassembled {tar_path.stat().st_size / 1e6:.1f} MB", flush=True)
assert tar_path.exists(), f"Missing {tar_path}"

print("=== Installing vLLM 0.34.0 + deps ===", flush=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "vllm==0.34.0",
        "httpx",
        "pillow",
        "huggingface_hub",
    ],
    check=True,
)
print("=== Running Pilot-150 ===", flush=True)
cmd = [
    sys.executable,
    "-u",
    "/content/run_colab_pilot.py",
    "--work-dir",
    "/content/pilot150",
    "--tar",
    str(tar_path),
]
only = os.environ.get("PILOT_ONLY", "").strip()
if only:
    cmd += ["--only", only]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
assert proc.stdout is not None
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
if rc != 0:
    raise RuntimeError(f"run_colab_pilot.py failed with exit code {rc}")

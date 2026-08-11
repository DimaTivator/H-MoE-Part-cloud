#!/usr/bin/env bash
set -u

log_dir=/home/jovyan/hmoe-cloud/logs
log=$log_dir/bootstrap-torch28-$(date +%F_%H%M%S).log
mkdir -p "$log_dir"

(
    set -eu
    df -h /home/jovyan
    python -m pip install --user --upgrade pybind11 setuptools wheel
    python -m pip install --user \
        'transformer_engine==2.16.0' \
        'transformer_engine_cu12==2.16.0' \
        einops nvdlfw-inspect onnx onnxscript
    python - <<'PY'
import site
import urllib.request
import zipfile
from pathlib import Path

url = (
    "https://github.com/NVIDIA/TransformerEngine/releases/download/v2.16/"
    "transformer_engine_torch-2.16.0+cu12torch2.8.0+cu129cxx11abiTRUE-"
    "cp312-cp312-linux_x86_64.whl"
)
wheel = Path("/tmp/transformer_engine_torch.whl")
urllib.request.urlretrieve(url, wheel)
target = Path(site.getusersitepackages())
target.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(target)
print(f"transformer_engine_torch_wheel={wheel.stat().st_size} target={target}")
PY
    python - <<'PY'
import pybind11
import torch
import transformer_engine
import transformer_engine.pytorch
import transformer_engine_torch

print("python_torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("pybind11=", pybind11.__version__)
print("transformer_engine=", transformer_engine.__version__)
print("cublaslt=", transformer_engine_torch.get_cublasLt_version())
print("torch28_bootstrap=PASS")
PY
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
tail -n 300 "$log"
exit 0

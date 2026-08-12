#!/usr/bin/env bash
set -uo pipefail

TORCH_VENV=${TORCH_VENV:-/home/jovyan/hmoe-cloud/torch251-cu121}
LOG_DIR=${LOG_DIR:-/home/jovyan/logs/optimizer_fp8_cloud}
LOG_FILE="${LOG_DIR}/bootstrap_torch251_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}" "$(dirname "${TORCH_VENV}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

finish() {
    status=$?
    echo "EXIT=${status}"
    echo "LOG=${LOG_FILE}"
    tail -n 80 "${LOG_FILE}" || true
    exit 0
}
trap finish EXIT

echo "Recreating ${TORCH_VENV} after the previous incomplete 11 MB bootstrap."
python -m venv --clear --system-site-packages "${TORCH_VENV}"
"${TORCH_VENV}/bin/python" -m pip install --upgrade pip
"${TORCH_VENV}/bin/python" -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121

"${TORCH_VENV}/bin/python" - <<'PY'
import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("torch251_bootstrap=PASS")
PY

du -sh "${TORCH_VENV}"
df -h /home/jovyan

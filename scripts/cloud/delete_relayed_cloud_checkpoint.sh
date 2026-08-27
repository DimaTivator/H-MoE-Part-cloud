#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/exps}
: "${WANDB_GROUP:?WANDB_GROUP is required}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${CHECKPOINT_NAME:?CHECKPOINT_NAME is required}"
: "${EXPECTED_ITERATION:?EXPECTED_ITERATION is required}"

if [[ ! "${WANDB_GROUP}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "INVALID_WANDB_GROUP=${WANDB_GROUP}" >&2
    exit 2
fi
if [[ ! "${EXPERIMENT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "INVALID_EXPERIMENT_NAME=${EXPERIMENT_NAME}" >&2
    exit 2
fi
if [[ "${CHECKPOINT_NAME}" != "latest" && "${CHECKPOINT_NAME}" != "67911" ]]; then
    echo "INVALID_CHECKPOINT_NAME=${CHECKPOINT_NAME}" >&2
    exit 2
fi
if [[ ! "${EXPECTED_ITERATION}" =~ ^[1-9][0-9]*$ ]]; then
    echo "INVALID_EXPECTED_ITERATION=${EXPECTED_ITERATION}" >&2
    exit 2
fi

readonly experiment_dir="${RESULTS_DIR}/${WANDB_GROUP}/${EXPERIMENT_NAME}"
readonly checkpoint_dir="${experiment_dir}/ckpts/${CHECKPOINT_NAME}"

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
echo "CHECKPOINT_DIR=${checkpoint_dir}"
df -h /workspace-SR006.nfs3
df -i /workspace-SR006.nfs3

if [[ -L "${experiment_dir}" || -L "${experiment_dir}/ckpts" || -L "${checkpoint_dir}" ]]; then
    echo "REFUSE_SYMLINK=${checkpoint_dir}" >&2
    exit 2
fi
if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "CHECKPOINT_ALREADY_ABSENT=${checkpoint_dir}"
    exit 0
fi

CHECKPOINT_PATH="${checkpoint_dir}/main.pt" \
WORKER_PATH="${checkpoint_dir}/worker_0.pt" \
EXPECTED_ITERATION="${EXPECTED_ITERATION}" \
    "$(command -v python)" - <<'PY'
import os
from pathlib import Path

import torch

main_path = Path(os.environ["CHECKPOINT_PATH"])
worker_path = Path(os.environ["WORKER_PATH"])
expected_iteration = int(os.environ["EXPECTED_ITERATION"])
if not main_path.is_file() or not worker_path.is_file():
    raise FileNotFoundError(
        f"incomplete checkpoint: main={main_path.is_file()} "
        f"worker={worker_path.is_file()}"
    )
checkpoint = torch.load(main_path, map_location="cpu", mmap=True, weights_only=False)
torch.load(worker_path, map_location="cpu", weights_only=False)
iteration = int(checkpoint["itr"])
if iteration != expected_iteration:
    raise ValueError(
        f"checkpoint iteration changed: expected={expected_iteration} actual={iteration}"
    )
print(f"VALIDATED_ITERATION={iteration}")
print(f"MAIN_BYTES={main_path.stat().st_size}")
print(f"WORKER_BYTES={worker_path.stat().st_size}")
PY

echo "DELETING_RELAYED_CHECKPOINT=${checkpoint_dir}"
rm -rf -- "${checkpoint_dir}"
if [[ -e "${checkpoint_dir}" ]]; then
    echo "DELETE_FAILED=${checkpoint_dir}" >&2
    exit 3
fi

echo "DELETED_RELAYED_CHECKPOINT=${checkpoint_dir}"
echo "DELETED_ITERATION=${EXPECTED_ITERATION}"
df -h /workspace-SR006.nfs3
df -i /workspace-SR006.nfs3

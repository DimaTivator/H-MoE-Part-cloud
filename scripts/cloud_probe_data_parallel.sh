#!/usr/bin/env bash
set -u

rank=${OMPI_COMM_WORLD_RANK:-0}
log_dir=/home/jovyan/hmoe-cloud/logs
log=$log_dir/data-parallel-probe-$(date +%F_%H%M%S)-rank${rank}.log
mkdir -p "$log_dir"

(
    set -eu
    echo "hostname=$(hostname)"
    echo "rank=${OMPI_COMM_WORLD_RANK:-unset}"
    echo "local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-unset}"
    echo "world_size=${OMPI_COMM_WORLD_SIZE:-unset}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    python - <<'PY'
import os

import torch

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"device_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"device_{index}={torch.cuda.get_device_name(index)}")
for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
    print(f"{key.lower()}={os.environ.get(key, 'unset')}")
PY
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
cat "$log"
exit 0

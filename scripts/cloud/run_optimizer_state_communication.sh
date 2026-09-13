#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-smoke}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
RUN_ID=${RUN_ID:?RUN_ID must be set to a common value for all mlsub ranks}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
RESULTS_ROOT=${RESULTS_ROOT:-/workspace-SR006.nfs3/dimativator/optimizer-state-comm-cloud-20260913}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}

case "${MODE}" in
    smoke)
        MODE_ARGS=(--models 500M --warmup-steps 2 --measure-steps 3 --repeats 1)
        ;;
    full)
        MODE_ARGS=(--models 500M 1B --warmup-steps 10 --measure-steps 50 --repeats 3)
        ;;
    *)
        echo "MODE must be smoke or full" >&2
        exit 2
        ;;
esac

MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
MPI_RANK=${OMPI_COMM_WORLD_RANK:-0}
MPI_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
if (( MPI_SIZE > 1 && MPI_SIZE != NPROC_PER_NODE )); then
    echo "mlsub MPI world size ${MPI_SIZE} != ${NPROC_PER_NODE}" >&2
    exit 8
fi

export RANK=${RANK:-${MPI_RANK}}
export WORLD_SIZE=${WORLD_SIZE:-${MPI_SIZE}}
export LOCAL_RANK=${LOCAL_RANK:-${MPI_LOCAL_RANK}}
export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
export MASTER_PORT=${MASTER_PORT:-29500}

OUTPUT_DIR="${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}" "${EVAL_CACHE_DIR}"
RANK_LOG="${OUTPUT_DIR}/launcher.rank${MPI_RANK}.log"
exec > >(tee -a "${RANK_LOG}") 2>&1

echo "RUN_ID=${RUN_ID} MODE=${MODE} MPI_RANK=${MPI_RANK}/${MPI_SIZE} LOCAL_RANK=${MPI_LOCAL_RANK}"
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds) OUTPUT_DIR=${OUTPUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
echo "GPU_COMPUTE_PROCESSES_BEFORE_RUN"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader || true

python - <<'PY'
import torch
import torchao
import triton

assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), torch.version.cuda
assert torch.cuda.is_available()
assert torchao.__version__.startswith("0.15.0"), torchao.__version__
assert triton.__version__ == "3.5.1", triton.__version__
print("ENVIRONMENT_CHECK=ok", torch.__version__, torch.version.cuda)
PY

if [[ ! -f "${DATASETS_DIR}/packed_metadata.json" ]]; then
    echo "Packed H200 data is missing: ${DATASETS_DIR}" >&2
    exit 5
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/triton-state-comm-${RUN_ID}-rank${MPI_RANK}-$$"
mkdir -p "${TRITON_CACHE_DIR}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python scripts/benchmarks/optimizer_state_communication_cloud.py \
    --datasets-dir "${DATASETS_DIR}" \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "${MODE_ARGS[@]}"

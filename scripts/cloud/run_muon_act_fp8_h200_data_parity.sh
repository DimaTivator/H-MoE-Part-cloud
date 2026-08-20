#!/usr/bin/env bash
set -uo pipefail

# Cloud.ru counterpart of W&B run xtnguiqq. The optimizer-step batch stays
# 128 tokens sequences: microbatch 32 x accumulation 4 on one H100.

DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs2/dimativator/fineweb-h200-packed}
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/exps}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
LOG_DIR=${LOG_DIR:-/workspace-SR006.nfs3/dimativator/logs/muon_act_fp8_cloud}
WANDB_PROJECT=${WANDB_PROJECT:-fp8-pretrain}
WANDB_ENTITY=${WANDB_ENTITY:-andrey}
WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
WANDB_GROUP=${WANDB_GROUP:-1xChinchilla_fp8}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-500m_muon_act_fp8_1xC_cloud_1gpu_bs32_acc4_h200_data_parity_v1}

BATCH_SIZE=32
ACC_STEPS=4
FINEWEB_REPLAY_WORLD_SIZE=2
ITERATIONS=75457
WARMUP_STEPS=7000

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${EVAL_CACHE_DIR}"
LOG_FILE="${LOG_DIR}/muon_act_fp8_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

finish() {
    status=$?
    echo "EXIT=${status}"
    echo "LOG=${LOG_FILE}"
    tail -n 120 "${LOG_FILE}" || true
    trap - EXIT
    exit "${status}"
}
trap finish EXIT

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "DATASETS_DIR=${DATASETS_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE} ACC_STEPS=${ACC_STEPS} GLOBAL_BATCH=$((BATCH_SIZE * ACC_STEPS))"
echo "FINEWEB_REPLAY_WORLD_SIZE=${FINEWEB_REPLAY_WORLD_SIZE}"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 /tmp 2>&1 || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || true

PYTHON_BIN=$(command -v python)
TORCHRUN_BIN=$(command -v torchrun)
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata

import torch
import torchao
import triton

assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), torch.version.cuda
assert torch.cuda.is_available()
assert torchao.__version__.startswith("0.15.0"), torchao.__version__
assert triton.__version__ == "3.5.1", triton.__version__
print("python_environment", "torch", torch.__version__, "cuda", torch.version.cuda)
print("python_environment", "torchao", torchao.__version__, "triton", triton.__version__)
print("python_environment", "wandb", metadata.version("wandb"))
print("gpu", torch.cuda.get_device_name(0))
PY

if [[ ! -f "${DATASETS_DIR}/packed_metadata.json" ]]; then
    echo "Packed FineWeb dataset is missing from ${DATASETS_DIR}" >&2
    exit 5
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT WANDB_ENTITY WANDB_BASE_URL

if [[ -z "${WANDB_API_KEY:-}" && -r /home/jovyan/.netrc ]]; then
    export HOME=/home/jovyan
    echo "WANDB_AUTH_SOURCE=/home/jovyan/.netrc"
fi

"${TORCHRUN_BIN}" --standalone --nproc_per_node=1 src/main.py \
    --distributed-backend nccl \
    --experiment-name "${EXPERIMENT_NAME}" \
    --dataset fineweb \
    --datasets-dir "${DATASETS_DIR}" \
    --fineweb-replay-world-size "${FINEWEB_REPLAY_WORLD_SIZE}" \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length 1024 \
    --data-seed 1337 \
    --streaming \
    --workers 8 \
    --model llama \
    --n-layer 18 \
    --n-embd 1280 \
    --n-head 20 \
    --multiple-of 256 \
    --dtype bfloat16 \
    --opt muon \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --beta1 0.9 \
    --beta2 0.99 \
    --grad-clip 1.0 \
    --scheduler wsd \
    --warmup-steps "${WARMUP_STEPS}" \
    --iterations "${ITERATIONS}" \
    --wsd-fract-decay 0.1 \
    --wsd-final-lr-scale 0.0 \
    --decay-type cosine \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size 32 \
    --acc-steps "${ACC_STEPS}" \
    --fp8 \
    --fp8-fabit E4M3 \
    --fp8-fwbit E4M3 \
    --fp8-babit E5M2 \
    --fp8-bwbit E5M2 \
    --fp8-group-size 16 \
    --eval-interval 500 \
    --eval-batches 32 \
    --downstream-eval-enabled \
    --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled \
    --lm-eval-interval 2000 \
    --lm-eval-datasets wikitext103 \
    --log-interval 50 \
    --inter-ckpts 67911 \
    --latest-ckpt-interval 5000 \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-tags fineweb act_fp8 1xChinchilla 0.5B 1gpu cloudru h100 muon bs32 acc4 torch291 efficient-image h200-data-parity

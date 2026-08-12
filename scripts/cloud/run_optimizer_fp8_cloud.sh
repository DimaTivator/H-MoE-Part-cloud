#!/usr/bin/env bash
set -uo pipefail

MODE=${MODE:-probe}
OPTIMIZER=${OPTIMIZER:-muon}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs2/dimativator/fineweb-edu-100BT-16shards}
RESULTS_DIR=${RESULTS_DIR:-/home/jovyan/exps}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
LOG_DIR=${LOG_DIR:-/home/jovyan/logs/optimizer_fp8_cloud}
WANDB_PROJECT=${WANDB_PROJECT:-fp8-pretrain}
WANDB_ENTITY=${WANDB_ENTITY:-andrey}
WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${EVAL_CACHE_DIR}"
LOG_FILE="${LOG_DIR}/${OPTIMIZER}_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

finish() {
    status=$?
    echo "EXIT=${status}"
    echo "LOG=${LOG_FILE}"
    tail -n 80 "${LOG_FILE}" || true
    exit 0
}
trap finish EXIT

echo "MODE=${MODE} OPTIMIZER=${OPTIMIZER}"
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 /tmp 2>&1 || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || true

if [[ "${MODE}" == "inspect" ]]; then
    du -sh "${DATASETS_DIR}" 2>&1 || true
    find "${DATASETS_DIR}" -maxdepth 1 -type f -printf '%s %f\n' 2>/dev/null | sort
    find "${LOG_DIR}" -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 5
    latest_log=$(find "${LOG_DIR}" -maxdepth 1 -type f -name '*.log' -print 2>/dev/null | sort | tail -n 1)
    if [[ -n "${latest_log}" ]]; then
        tail -n 120 "${latest_log}"
    fi
    exit 0
fi

python -m pip install --user --disable-pip-version-check -r scripts/cloud/requirements.txt
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "${MODE}" == "probe" ]]; then
    python - <<'PY'
import sys
sys.path.insert(0, "src")
import torch
import pyarrow
import tiktoken
import wandb
from types import SimpleNamespace
from optim.fp8_state import (
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)
from optim.sota_opt.fp8_ademamix import FP8AdEMAMix
from third_party.lite.muonlite import MuonLite
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("float8", torch.float8_e4m3fn, torch.float8_e5m2)
print("pyarrow", pyarrow.__version__, "tiktoken", tiktoken.__version__, "wandb", wandb.__version__)
qargs = SimpleNamespace(
    first_order_bit="E4M3",
    second_order_bit="E4M3",
    first_order_expansion="expand",
    second_order_expansion="expand",
    qgroup_size=128,
    expand_min=16,
)
source = torch.linspace(-1, 1, 257)
state = {}
init_fp8_state(state, "momentum", source, qargs, order="first")
quantize_fp8_state_(state, "momentum", source, qargs, signed=True)
restored = dequantize_fp8_state(state, "momentum", qargs, signed=True)
assert restored.shape == source.shape and torch.isfinite(restored).all()
print("fp8_state_smoke_mae", float((source - restored).abs().mean()))
print("optimizer_imports", MuonLite.__name__, FP8AdEMAMix.__name__)
PY
    exit 0
fi

if [[ "${MODE}" == "data" ]]; then
    python scripts/cloud/download_fineweb_subset.py \
        --destination "${DATASETS_DIR}" \
        --manifest scripts/cloud/fineweb_subset_16.txt
    exit 0
fi

test -f "${DATASETS_DIR}/.subset_complete"
count=$(find "${DATASETS_DIR}" -maxdepth 1 -type f -name '*.parquet' | wc -l | tr -d ' ')
test "${count}" = "16"

if [[ "${MODE}" == "cpu_smoke" ]]; then
    EXPERIMENT_NAME="tiny_${OPTIMIZER}_optimizer_fp8_cloud_cpu_smoke"
    ITERATIONS=2
    WARMUP=1
    BATCH_SIZE=1
    ACC_STEPS=1
    SEQ_LEN=128
    N_LAYER=2
    N_EMBD=128
    N_HEAD=4
    EVAL_INTERVAL=2
    EVAL_BATCHES=1
    LOG_INTERVAL=1
    CHECKPOINT_ARGS=(--latest-ckpt-interval 0)
    EVAL_ARGS=()
    WANDB_ARGS=()
    LAUNCH=(python src/main.py)
    BACKEND_ARGS=(--device cpu)
elif [[ "${MODE}" == "smoke" ]]; then
    EXPERIMENT_NAME="500m_${OPTIMIZER}_optimizer_fp8_cloud_smoke"
    ITERATIONS=3
    WARMUP=1
    BATCH_SIZE=${BATCH_SIZE:-32}
    ACC_STEPS=1
    SEQ_LEN=1024
    N_LAYER=18
    N_EMBD=1280
    N_HEAD=20
    EVAL_INTERVAL=3
    EVAL_BATCHES=1
    LOG_INTERVAL=1
    CHECKPOINT_ARGS=(--latest-ckpt-interval 0)
    EVAL_ARGS=()
    WANDB_ARGS=()
    LAUNCH=(torchrun --standalone --nproc_per_node=1 src/main.py)
    BACKEND_ARGS=(--distributed-backend nccl)
else
    EXPERIMENT_NAME="500m_${OPTIMIZER}_optimizer_fp8_1xC_cloud_h100"
    ITERATIONS=75457
    WARMUP=2000
    BATCH_SIZE=${BATCH_SIZE:-32}
    ACC_STEPS=${ACC_STEPS:-4}
    SEQ_LEN=1024
    N_LAYER=18
    N_EMBD=1280
    N_HEAD=20
    EVAL_INTERVAL=500
    EVAL_BATCHES=32
    LOG_INTERVAL=50
    EVAL_ARGS=(
        --downstream-eval-enabled
        --downstream-eval-interval 2000
        --downstream-task-group basic_v2
        --lm-eval-enabled
        --lm-eval-interval 2000
        --lm-eval-datasets wikitext103
    )
    CHECKPOINT_ARGS=(
        --inter-ckpts 10000 20000 30000 40000 50000 60000 67911 70000
        --latest-ckpt-interval 10000
        --upload-inter-ckpts-to wandb
        --delete-local-inter-ckpts-after-upload
    )
    WANDB_ARGS=(
        --wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group 1xChinchilla_optimizer_fp8_cloud
        --wandb-tags fineweb optimizer_fp8 bf16_model 1xChinchilla 0.5B 1gpu cloudru h100 "${OPTIMIZER}"
    )
    LAUNCH=(torchrun --standalone --nproc_per_node=1 src/main.py)
    BACKEND_ARGS=(--distributed-backend nccl)
fi

OPT_ARGS=()
if [[ "${OPTIMIZER}" == "muon" ]]; then
    OPT_ARGS=(--opt muon --lr 1e-3 --weight-decay 0.1 --beta1 0.9 --beta2 0.99 --grad-clip 1.0)
elif [[ "${OPTIMIZER}" == "ademamix" ]]; then
    OPT_ARGS=(
        --opt ademamix --lr 1e-3 --weight-decay 0.1
        --beta1 0.9 --beta2 0.999
        --ademamix_beta3 0.9999 --ademamix_alpha 8
        --ademamix_beta3_warmup_steps "${ITERATIONS}"
        --ademamix_alpha_warmup_steps "${ITERATIONS}"
        --grad-clip 0.5
    )
else
    echo "Unsupported OPTIMIZER=${OPTIMIZER}" >&2
    exit 2
fi

export WANDB_PROJECT WANDB_ENTITY WANDB_BASE_URL
"${LAUNCH[@]}" \
    "${BACKEND_ARGS[@]}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --dataset fineweb \
    --datasets-dir "${DATASETS_DIR}" \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length "${SEQ_LEN}" \
    --streaming \
    --workers 8 \
    --model llama \
    --n-layer "${N_LAYER}" \
    --n-embd "${N_EMBD}" \
    --n-head "${N_HEAD}" \
    --multiple-of 256 \
    --dtype bfloat16 \
    "${OPT_ARGS[@]}" \
    --scheduler wsd \
    --warmup-steps "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --wsd-fract-decay 0.1 \
    --wsd-final-lr-scale 0.0 \
    --decay-type cosine \
    --batch-size "${BATCH_SIZE}" \
    --acc-steps "${ACC_STEPS}" \
    --fp8-optim \
    --fp8-qgroup-size 128 \
    --fp8-first-order-bit E4M3 \
    --fp8-second-order-bit E4M3 \
    --fp8-expansion expand \
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-batches "${EVAL_BATCHES}" \
    "${EVAL_ARGS[@]}" \
    --log-interval "${LOG_INTERVAL}" \
    "${CHECKPOINT_ARGS[@]}" \
    --results-base-folder "${RESULTS_DIR}" \
    "${WANDB_ARGS[@]}"

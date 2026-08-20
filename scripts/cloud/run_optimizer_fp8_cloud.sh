#!/usr/bin/env bash
set -uo pipefail

MODE=${MODE:-probe}
OPTIMIZER=${OPTIMIZER:-muon}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
GRAD_CLIP=${GRAD_CLIP:-}
CHECKPOINT_MODE=${CHECKPOINT_MODE:-milestones}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-10000}
DEEP_INSPECT=${DEEP_INSPECT:-0}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs2/dimativator/fineweb-edu-100BT-16shards}
RESULTS_DIR=${RESULTS_DIR:-/home/jovyan/exps}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
LOG_DIR=${LOG_DIR:-/home/jovyan/logs/optimizer_fp8_cloud}
WANDB_PROJECT=${WANDB_PROJECT:-fp8-pretrain}
WANDB_ENTITY=${WANDB_ENTITY:-andrey}
WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
WANDB_GROUP=${WANDB_GROUP:-1xChinchilla_optimizer_fp8_cloud}

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

echo "MODE=${MODE} OPTIMIZER=${OPTIMIZER} WEIGHT_DECAY=${WEIGHT_DECAY} GRAD_CLIP=${GRAD_CLIP:-optimizer_default} CHECKPOINT_MODE=${CHECKPOINT_MODE}"
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 /tmp 2>&1 || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>&1 || true

if [[ "${MODE}" == "inspect" ]]; then
    echo "HOME_USAGE"
    du -x -h --max-depth=1 /home/jovyan 2>/dev/null | sort -h || true
    if [[ "${DEEP_INSPECT}" == "1" ]]; then
        for path in \
            /home/jovyan/rl_muon \
            /home/jovyan/finewebedu_h200 \
            /home/jovyan/data \
            /home/jovyan/hmoe-cloud \
            /home/jovyan/exps; do
            if [[ -d "${path}" ]]; then
                echo "DEEP_USAGE=${path}"
                du -x -h --max-depth=2 "${path}" 2>/dev/null | sort -h | tail -n 40
            fi
        done
        echo "LARGEST_HOME_FILES"
        find \
            /home/jovyan/rl_muon \
            /home/jovyan/finewebedu_h200 \
            /home/jovyan/data \
            /home/jovyan/hmoe-cloud \
            /home/jovyan/exps \
            -xdev -type f -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n' 2>/dev/null \
            | sort -nr | head -n 80 || true
    fi
    du -sh "${DATASETS_DIR}" 2>&1 || true
    find "${DATASETS_DIR}" -maxdepth 1 -type f -printf '%s %f\n' 2>/dev/null | sort
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-500m_${OPTIMIZER}_optimizer_fp8_1xC_cloud_h100}
    EXPERIMENT_DIR="${RESULTS_DIR}/${WANDB_GROUP}/${EXPERIMENT_NAME}"
    echo "EXPERIMENT_DIR=${EXPERIMENT_DIR}"
    find "${EXPERIMENT_DIR}/ckpts" -maxdepth 2 -type f \
        -printf '%T@ %s %p\n' 2>/dev/null | sort -n || true
    if [[ -f "${EXPERIMENT_DIR}/ckpts/latest/main.pt" && \
          -x /home/jovyan/hmoe-cloud/torch251-cu121/bin/python ]]; then
        CHECKPOINT_PATH="${EXPERIMENT_DIR}/ckpts/latest/main.pt" \
            /home/jovyan/hmoe-cloud/torch251-cu121/bin/python - <<'PY'
import os
from pathlib import Path

import torch

path = Path(os.environ["CHECKPOINT_PATH"])
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
print(f"LATEST_CHECKPOINT={path}")
print(f"LATEST_CHECKPOINT_ITER={checkpoint.get('itr')}")
print(f"LATEST_CHECKPOINT_BYTES={path.stat().st_size}")
PY
    else
        echo "LATEST_CHECKPOINT=missing"
    fi
    find "${LOG_DIR}" -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 5
    latest_log=$(find "${LOG_DIR}" -maxdepth 1 -type f -name '*.log' -print 2>/dev/null | sort | tail -n 1)
    if [[ -n "${latest_log}" ]]; then
        tail -n 120 "${latest_log}"
    fi
    exit 0
fi

if [[ "${MODE}" == "reset_checkpoints" ]]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-500m_${OPTIMIZER}_optimizer_fp8_1xC_cloud_h100}
    CHECKPOINT_DIR="${RESULTS_DIR}/${WANDB_GROUP}/${EXPERIMENT_NAME}/ckpts"
    case "${CHECKPOINT_DIR}" in
        "${RESULTS_DIR}"/"${WANDB_GROUP}"/500m_*_1xC_cloud_h100/ckpts) ;;
        *)
            echo "Refusing unsafe checkpoint reset: ${CHECKPOINT_DIR}" >&2
            exit 4
            ;;
    esac
    if [[ -d "${CHECKPOINT_DIR}" ]]; then
        find "${CHECKPOINT_DIR}" -maxdepth 2 -type f -printf '%s %p\n' 2>/dev/null | sort || true
        rm -rf -- "${CHECKPOINT_DIR}"
    fi
    test ! -e "${CHECKPOINT_DIR}"
    echo "RESET_CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    exit 0
fi

if [[ "${MODE}" == "cleanup_duplicate_fineweb" ]]; then
    SOURCE_DIR=/home/jovyan/finewebedu_h200/sample/100BT
    CANONICAL_DIR=/workspace-SR006.nfs2/dimativator/fineweb-edu-100BT-16shards
    test -d "${SOURCE_DIR}"
    test -d "${CANONICAL_DIR}"

    matched=0
    while IFS= read -r -d '' source_file; do
        filename=$(basename "${source_file}")
        canonical_file="${CANONICAL_DIR}/${filename}"
        test -f "${canonical_file}"
        source_size=$(stat -c '%s' "${source_file}")
        canonical_size=$(stat -c '%s' "${canonical_file}")
        if [[ "${source_size}" != "${canonical_size}" ]]; then
            echo "Refusing cleanup: size mismatch for ${filename}" >&2
            exit 5
        fi
        echo "DUPLICATE=${filename} BYTES=${source_size}"
        matched=$((matched + 1))
    done < <(find "${SOURCE_DIR}" -maxdepth 1 -type f -name '*.parquet' -print0)

    if (( matched == 0 )); then
        echo "Refusing cleanup: no parquet files found in ${SOURCE_DIR}" >&2
        exit 5
    fi
    echo "REMOVING_DUPLICATE_DIR=${SOURCE_DIR} FILES=${matched}"
    rm -rf -- "${SOURCE_DIR}"
    test ! -e "${SOURCE_DIR}"
    rmdir /home/jovyan/finewebedu_h200/sample 2>/dev/null || true
    df -h /home/jovyan
    exit 0
fi

if [[ "${MODE}" == "cleanup_rebuildable_storage" ]]; then
    cleanup_targets=(
        /home/jovyan/rl_muon/gsm8k_ppo_r4-a7c7fd2/venv
        /home/jovyan/rl_muon/campaign-21ae118-153758/venv
        /home/jovyan/evals_cache
    )

    echo "CLEANUP_REBUILDABLE_STORAGE_BEFORE"
    df -h /home/jovyan
    for path in "${cleanup_targets[@]}"; do
        case "${path}" in
            /home/jovyan/rl_muon/*/venv|/home/jovyan/evals_cache) ;;
            *)
                echo "Refusing unsafe cleanup target: ${path}" >&2
                exit 6
                ;;
        esac
        if [[ -e "${path}" ]]; then
            du -sh "${path}"
        else
            echo "ALREADY_MISSING=${path}"
        fi
    done

    for path in "${cleanup_targets[@]}"; do
        if [[ -e "${path}" ]]; then
            echo "REMOVING_REBUILDABLE=${path}"
            rm -rf -- "${path}"
        fi
        test ! -e "${path}"
    done
    echo "CLEANUP_REBUILDABLE_STORAGE_AFTER"
    df -h /home/jovyan
    exit 0
fi

if [[ "${MODE}" == "cleanup_local_checkpoints" ]]; then
    checkpoint_root=/home/jovyan/exps
    if [[ ! -d "${checkpoint_root}" ]]; then
        echo "LOCAL_CHECKPOINT_ROOT=missing"
        exit 0
    fi

    mapfile -d '' checkpoint_dirs < <(
        find "${checkpoint_root}" -mindepth 2 -type d -name ckpts -prune -print0
    )
    echo "LOCAL_CHECKPOINT_DIRS=${#checkpoint_dirs[@]}"
    if (( ${#checkpoint_dirs[@]} == 0 )); then
        exit 0
    fi

    df -h /home/jovyan
    for path in "${checkpoint_dirs[@]}"; do
        case "${path}" in
            "${checkpoint_root}"/*/ckpts) ;;
            *)
                echo "Refusing unsafe checkpoint cleanup target: ${path}" >&2
                exit 7
                ;;
        esac
        du -sh "${path}"
    done

    for path in "${checkpoint_dirs[@]}"; do
        echo "REMOVING_LOCAL_CHECKPOINTS=${path}"
        rm -rf -- "${path}"
        test ! -e "${path}"
    done
    if find "${checkpoint_root}" -mindepth 2 -type d -name ckpts -print -quit | grep -q .; then
        echo "Checkpoint directory remained after cleanup" >&2
        exit 7
    fi
    df -h /home/jovyan
    exit 0
fi

if [[ "${MODE}" == "cleanup_legacy_h200_tokenized" ]]; then
    tokenized_dir=/home/jovyan/finewebedu_h200/tokenized
    if [[ ! -d "${tokenized_dir}" ]]; then
        echo "LEGACY_H200_TOKENIZED=missing"
        exit 0
    fi

    case "${tokenized_dir}" in
        /home/jovyan/finewebedu_h200/tokenized) ;;
        *)
            echo "Refusing unsafe dataset cleanup target: ${tokenized_dir}" >&2
            exit 8
            ;;
    esac
    df -h /home/jovyan
    du -sh "${tokenized_dir}"
    find "${tokenized_dir}" -maxdepth 1 -type f -printf '%s %p\n' | sort -n
    echo "REMOVING_LEGACY_H200_TOKENIZED=${tokenized_dir}"
    rm -rf -- "${tokenized_dir}"
    test ! -e "${tokenized_dir}"
    df -h /home/jovyan
    exit 0
fi

SYSTEM_PYTHON=${SYSTEM_PYTHON:-python}
TORCH_VENV=${TORCH_VENV:-/home/jovyan/hmoe-cloud/torch251-cu121}

"${SYSTEM_PYTHON}" -m pip install --user --disable-pip-version-check -r scripts/cloud/requirements.txt
USER_SITE=$("${SYSTEM_PYTHON}" -c 'import site; print(site.getusersitepackages())')
export PYTHONPATH="${USER_SITE}:${PYTHONPATH:-}"

if [[ ! -x "${TORCH_VENV}/bin/python" || ! -x "${TORCH_VENV}/bin/torchrun" ]]; then
    echo "Required persistent Torch environment is missing: ${TORCH_VENV}" >&2
    echo "Bootstrap torch==2.5.1+cu121 there before launching this job." >&2
    exit 3
fi

if grep -q '^include-system-site-packages = false$' "${TORCH_VENV}/pyvenv.cfg"; then
    echo "Enabling base-image packages in ${TORCH_VENV}; the venv Torch remains first on sys.path."
    "${SYSTEM_PYTHON}" -m venv --upgrade --system-site-packages "${TORCH_VENV}"
fi

PYTHON_BIN="${TORCH_VENV}/bin/python"
TORCHRUN_BIN="${TORCH_VENV}/bin/torchrun"
"${PYTHON_BIN}" - <<'PY'
import torch
from packaging.version import Version

version = torch.__version__.split("+")[0]
assert Version(version) >= Version("2.5"), torch.__version__
print("selected_torch", torch.__version__, "cuda", torch.version.cuda)
PY
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "${MODE}" == "probe" ]]; then
    "${PYTHON_BIN}" - <<'PY'
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
from optim.sota_opt.fp8_soap import FP8SOAP
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
parameter = torch.nn.Parameter(torch.randn(4, 3))
optimizer = FP8SOAP(
    [parameter],
    qargs=qargs,
    lr=1e-3,
    betas=(0.9, 0.99),
    weight_decay=1e-4,
    precondition_frequency=2,
)
for _ in range(3):
    optimizer.zero_grad()
    parameter.square().mean().backward()
    optimizer.step()
soap_state = optimizer.state[parameter]
assert soap_state["fp8_exp_avg"].dtype == torch.float8_e4m3fn
assert soap_state["fp8_exp_avg_sq"].dtype == torch.float8_e4m3fn
assert "exp_avg" not in soap_state and "exp_avg_sq" not in soap_state
assert torch.isfinite(parameter).all()
print("fp8_soap_smoke", "ok", "step", soap_state["step"])
print("optimizer_imports", MuonLite.__name__, FP8AdEMAMix.__name__, FP8SOAP.__name__)
PY
    exit 0
fi

if [[ "${MODE}" == "data" ]]; then
    "${PYTHON_BIN}" scripts/cloud/download_fineweb_subset.py \
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
    LAUNCH=("${PYTHON_BIN}" src/main.py)
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
    LAUNCH=("${TORCHRUN_BIN}" --standalone --nproc_per_node=1 src/main.py)
    BACKEND_ARGS=(--distributed-backend nccl)
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME:-500m_${OPTIMIZER}_optimizer_fp8_1xC_cloud_h100}
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
    if [[ "${CHECKPOINT_MODE}" == "milestones" ]]; then
        CHECKPOINT_ARGS=(
            --inter-ckpts 10000 20000 30000 40000 50000 60000 67911 70000
            --latest-ckpt-interval "${LATEST_CKPT_INTERVAL}"
            --upload-inter-ckpts-to wandb
            --delete-local-inter-ckpts-after-upload
        )
    elif [[ "${CHECKPOINT_MODE}" == "latest" ]]; then
        if (( LATEST_CKPT_INTERVAL <= 0 )); then
            echo "CHECKPOINT_MODE=latest requires LATEST_CKPT_INTERVAL > 0" >&2
            exit 2
        fi
        CHECKPOINT_ARGS=(--latest-ckpt-interval "${LATEST_CKPT_INTERVAL}")
    else
        echo "Unsupported CHECKPOINT_MODE=${CHECKPOINT_MODE}" >&2
        exit 2
    fi
    WANDB_ARGS=(
        --wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --wandb-tags fineweb optimizer_fp8 bf16_model 1xChinchilla 0.5B 1gpu cloudru h100 "${OPTIMIZER}"
    )
    LAUNCH=("${TORCHRUN_BIN}" --standalone --nproc_per_node=1 src/main.py)
    BACKEND_ARGS=(--distributed-backend nccl)
fi

OPT_ARGS=()
if [[ "${OPTIMIZER}" == "muon" ]]; then
    OPT_ARGS=(--opt muon --lr 1e-3 --weight-decay "${WEIGHT_DECAY}" --beta1 0.9 --beta2 0.99 --grad-clip "${GRAD_CLIP:-1.0}")
elif [[ "${OPTIMIZER}" == "ademamix" ]]; then
    OPT_ARGS=(
        --opt ademamix --lr 1e-3 --weight-decay "${WEIGHT_DECAY}"
        --beta1 0.9 --beta2 0.999
        --ademamix_beta3 0.9999 --ademamix_alpha 8
        --ademamix_beta3_warmup_steps "${ITERATIONS}"
        --ademamix_alpha_warmup_steps "${ITERATIONS}"
        --grad-clip "${GRAD_CLIP:-0.5}"
    )
elif [[ "${OPTIMIZER}" == "soap" ]]; then
    OPT_ARGS=(
        --opt soap --lr 1e-3 --weight-decay "${WEIGHT_DECAY}"
        --beta1 0.9 --beta2 0.99
        --grad-clip "${GRAD_CLIP:-1.0}"
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

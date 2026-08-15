#!/usr/bin/env bash
set -u

mode=${1:-imports}
root=$(cd "$(dirname "$0")/.." && pwd)
log_dir=/home/jovyan/logs
log=$log_dir/efficient-training-smoke-$(date +%F_%H%M%S).log
mkdir -p "$log_dir" /home/jovyan/datasets/efficient-training-smoke

(
    set -u
    cd "$root"
    export PYTHONUNBUFFERED=1
    export HF_HOME=${HF_HOME:-/home/jovyan/.cache/huggingface}
    export TORCH_HOME=${TORCH_HOME:-/home/jovyan/.cache/torch}
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/home/jovyan/.cache/triton}

    if ! python -c 'import torchao, bitsandbytes, datasets, pyarrow, tensorly, tiktoken, transformers, wandb; from distributed_shampoo import DistributedShampoo'; then
        echo "Installing Cloud.ru runtime delta into the persistent user package cache"
        python -m pip install --user --upgrade-strategy only-if-needed -r requirements.cloudru.txt
    fi

    python - <<'PY'
import platform
import torch
import triton
import torchao
import src.main

print("python=", platform.python_version())
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("triton=", triton.__version__)
print("torchao=", torchao.__version__)
print("cuda_available=", torch.cuda.is_available())
print("efficient_training_import=PASS")
PY

    if [[ "$mode" == imports ]]; then
        exit 0
    fi

    common_args=(
        --dataset shakespeare-char
        --datasets-dir /home/jovyan/datasets/efficient-training-smoke
        --model llama
        --n-layer 2
        --n-embd 128
        --n-head 4
        --sequence-length 64
        --multiple-of 64
        --vocab-size 128
        --dtype bfloat16
        --opt adamw
        --lr 1e-3
        --weight-decay 0.0
        --scheduler cos
        --warmup-steps 1
        --iterations 2
        --batch-size 2
        --eval-batch-size 2
        --acc-steps 1
        --eval-interval 1
        --eval-batches 1
        --log-interval 1
        --no-local-save
    )

    python src/main.py --experiment-name cloudru-smoke-bf16 "${common_args[@]}"
    bf16_code=$?
    echo "bf16_exit=$bf16_code"

    python src/main.py --experiment-name cloudru-smoke-fp8 --fp8 "${common_args[@]}"
    fp8_code=$?
    echo "fp8_exit=$fp8_code"

    if [[ $bf16_code -eq 0 && $fp8_code -eq 0 ]]; then
        echo "efficient_training_smoke=PASS"
    else
        echo "efficient_training_smoke=FAIL"
    fi
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
tail -n 300 "$log"
exit 0

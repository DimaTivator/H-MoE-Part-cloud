#!/usr/bin/env bash
set -u

root=$(cd "$(dirname "$0")" && pwd)
output_dir=${BENCHMARK_OUTPUT_DIR:-/home/jovyan/hmoe-cloud/step-time}
log_dir=/home/jovyan/hmoe-cloud/logs
log=$log_dir/training-step-$(date +%F_%H%M%S).log
mkdir -p "$log_dir" "$output_dir"

(
    set -eu
    export PYTHONNOUSERSITE=1
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
    export STAGE4_FP8_BATCHED=1
    python "$root/scripts/benchmark_training_step.py" --output-dir "$output_dir" "$@"
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
echo "OUTPUT_DIR=$output_dir"
tail -n 80 "$log"
if [[ -f "$output_dir/results.csv" ]]; then
    echo "=== RESULTS CSV ==="
    cat "$output_dir/results.csv"
fi
if [[ -f "$output_dir/results.json" ]]; then
    python - "$output_dir/results.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print("=== BENCHMARK METADATA ===")
print(json.dumps(payload["metadata"], sort_keys=True))
PY
fi
exit 0

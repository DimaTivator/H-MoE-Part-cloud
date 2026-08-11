#!/usr/bin/env bash
set -eu

shopt -s nullglob
logs=(/home/jovyan/hmoe-cloud/logs/training-step-*)
if (( ${#logs[@]} == 0 )); then
    echo "NO_LOG_YET"
    exit 0
fi
log=$(ls -1t "${logs[@]}" | head -1)
echo "LOG=$log"
tail -n 300 "$log"

output_dir=${BENCHMARK_OUTPUT_DIR:-/home/jovyan/hmoe-cloud/step-time}
config_logs=("$output_dir"/logs/*.log)
if (( ${#config_logs[@]} > 0 )); then
    config_log=$(ls -1t "${config_logs[@]}" | head -1)
    echo "CONFIG_LOG=$config_log"
    tail -n 300 "$config_log"

    echo "CONFIG_ERROR_SUMMARY"
    for config_log in "${config_logs[@]}"; do
        echo "--- $config_log"
        grep -E '(^|: )(AssertionError|AttributeError|ImportError|KeyError|ModuleNotFoundError|NotImplementedError|RuntimeError|TypeError|ValueError)|error:' \
            "$config_log" | tail -n 5 || true
    done
fi

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

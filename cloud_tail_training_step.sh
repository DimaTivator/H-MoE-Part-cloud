#!/usr/bin/env bash
set -eu

log=$(ls -1t /home/jovyan/hmoe-cloud/logs/training-step-* | head -1)
echo "LOG=$log"
tail -n 300 "$log"

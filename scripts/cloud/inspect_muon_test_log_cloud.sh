#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${LOG_DIR:-/workspace-SR006.nfs3/dimativator/logs/optimizer_fp8_cloud}
LOG_GLOB=${LOG_GLOB:-muon_efficient_test_*.log}

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
latest=$(find "${LOG_DIR}" -maxdepth 1 -type f -name "${LOG_GLOB}" \
    -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)
if [[ -z "${latest}" ]]; then
    echo "No matching log under ${LOG_DIR}"
    exit 0
fi
echo "LOG=${latest}"
tail -n 200 "${latest}"

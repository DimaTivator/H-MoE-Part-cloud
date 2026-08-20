#!/usr/bin/env bash
set -euo pipefail

inspect_root=/workspace-SR006.nfs2/dimativator

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /workspace-SR006.nfs2
if [[ ! -d "${inspect_root}" ]]; then
    echo "INSPECT_ROOT=${inspect_root} missing"
    exit 1
fi

echo "INSPECT_ROOT=${inspect_root}"
echo "TOP_LEVEL_USAGE"
du -h --max-depth=1 "${inspect_root}" 2>/dev/null | sort -h

echo "DEPTH_TWO_USAGE"
du -h --max-depth=2 "${inspect_root}" 2>/dev/null | sort -h | tail -n 100

echo "LARGEST_FILES"
find "${inspect_root}" -type f \
    -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' 2>/dev/null \
    | sort -nr | head -n 100

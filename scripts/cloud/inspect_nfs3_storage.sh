#!/usr/bin/env bash
set -euo pipefail

readonly dataset=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly destination_root=/workspace-SR006.nfs3/dimativator

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3

if [[ ! -d "${dataset}" || -L "${dataset}" ]]; then
    echo "DATASET_MISSING_OR_UNSAFE=${dataset}" >&2
    exit 1
fi
echo "DATASET=$(du -sh -- "${dataset}")"
echo "DATASET_BYTES=$(du -sb -- "${dataset}" | awk '{print $1}')"

if [[ ! -d "${destination_root}" ]]; then
    echo "DESTINATION_ROOT_MISSING=${destination_root}"
    exit 0
fi

echo "NFS3_TOP_LEVEL_USAGE"
du -h --max-depth=1 "${destination_root}" 2>/dev/null | sort -h
echo "NFS3_DEPTH_TWO_USAGE"
du -h --max-depth=2 "${destination_root}" 2>/dev/null | sort -h | tail -n 100
echo "NFS3_LARGEST_FILES"
find "${destination_root}" -type f \
    -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' 2>/dev/null \
    | sort -nr | head -n 100

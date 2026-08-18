#!/usr/bin/env bash
set -euo pipefail

DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs2/dimativator/fineweb-edu-100BT-full-h200}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-4}

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 /tmp 2>&1 || true
python scripts/cloud/download_fineweb_subset.py \
    --destination "${DATASETS_DIR}" \
    --manifest scripts/cloud/fineweb_h200_140.sha256 \
    --workers "${DOWNLOAD_WORKERS}" \
    --completion-marker .h200_snapshot_complete

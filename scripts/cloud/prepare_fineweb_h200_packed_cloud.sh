#!/usr/bin/env bash
set -euo pipefail

: "${PACKED_RANK:?PACKED_RANK is required (0 or 1)}"
PACKED_DESTINATION=${PACKED_DESTINATION:-/workspace-SR006.nfs2/dimativator/fineweb-h200-packed}
PREVIEW_BLOCKS=${PREVIEW_BLOCKS:-0}

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds) RANK=${PACKED_RANK}"
df -h /workspace-SR006.nfs2 /tmp

args=(
    --manifest scripts/cloud/fineweb_h200_manifest.json.gz.b64
    --rank "${PACKED_RANK}"
)
if [[ "${PREVIEW_BLOCKS}" != "0" ]]; then
    args+=(--preview-blocks "${PREVIEW_BLOCKS}")
else
    args+=(--destination "${PACKED_DESTINATION}")
fi
python scripts/cloud/prepare_fineweb_h200_packed.py "${args[@]}"

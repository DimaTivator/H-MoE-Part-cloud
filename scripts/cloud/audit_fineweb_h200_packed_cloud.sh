#!/usr/bin/env bash
set -euo pipefail

PACKED_DESTINATION=${PACKED_DESTINATION:-/workspace-SR006.nfs2/dimativator/fineweb-h200-packed}

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /workspace-SR006.nfs2
python scripts/cloud/audit_fineweb_h200_packed.py "${PACKED_DESTINATION}"

#!/usr/bin/env bash
set -euo pipefail

PACKED_DESTINATION=${PACKED_DESTINATION:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /workspace-SR006.nfs3
if [[ ! -d "${PACKED_DESTINATION}" ]]; then
    echo "PACKED_DESTINATION=${PACKED_DESTINATION} missing"
    exit 0
fi

echo "PACKED_DESTINATION=${PACKED_DESTINATION}"
du -sh "${PACKED_DESTINATION}"
find "${PACKED_DESTINATION}" -maxdepth 1 -type f \
    -printf '%f\t%s bytes\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' \
    | sort

for rank in 0 1; do
    state="${PACKED_DESTINATION}/train_rank${rank}.state.json"
    if [[ -f "${state}" ]]; then
        python - "${state}" <<'PY'
import json
import sys

path = sys.argv[1]
state = json.load(open(path))
print(f"{path}: blocks_written={state['blocks_written']}")
PY
    fi
done

if [[ -f "${PACKED_DESTINATION}/packed_metadata.json" ]]; then
    python - "${PACKED_DESTINATION}/packed_metadata.json" <<'PY'
import json
import sys

path = sys.argv[1]
metadata = json.load(open(path))
print(f"PACKED_SNAPSHOT={metadata['format']}")
print(f"manifest_fingerprint={metadata['manifest_fingerprint']}")
print(f"split_plan_fingerprint={metadata['split_plan_fingerprint']}")
print(f"validation_blocks_sha256={metadata['validation_blocks_sha256']}")
for rank in metadata['ranks']:
    print(
        f"rank={rank['rank']} blocks={rank['blocks']} bytes={rank['bytes']} "
        f"sha256={rank['sha256']}"
    )
PY
fi

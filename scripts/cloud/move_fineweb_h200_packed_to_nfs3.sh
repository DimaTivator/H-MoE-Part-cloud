#!/usr/bin/env bash
set -euo pipefail

readonly source=/workspace-SR006.nfs2/dimativator/fineweb-h200-packed
readonly destination_root=/workspace-SR006.nfs3/dimativator
readonly destination=${destination_root}/fineweb-h200-packed
readonly staging=${destination_root}/.fineweb-h200-packed.transfer
readonly reserve_bytes=$((2 * 1024 * 1024 * 1024))

if [[ ! -d "${source}" || -L "${source}" ]]; then
    echo "Refusing missing or symbolic-link source: ${source}" >&2
    exit 2
fi
if [[ "$(readlink -f -- "${source}")" != "${source}" ]]; then
    echo "Refusing unexpectedly resolved source: ${source}" >&2
    exit 2
fi
if [[ -e "${destination}" || -e "${staging}" ]]; then
    echo "Refusing existing destination or staging path" >&2
    exit 3
fi
if [[ ! -d "${destination_root}" || -L "${destination_root}" ]]; then
    echo "Refusing missing or symbolic-link destination root: ${destination_root}" >&2
    exit 4
fi

source_bytes=$(find "${source}" -type f -printf '%s\n' | awk '{total += $1} END {printf "%.0f\n", total}')
available_bytes=$(df -B1 --output=avail "${destination_root}" | tail -n 1 | tr -d ' ')
required_bytes=$((source_bytes + reserve_bytes))

echo "MOVE_PACKED_BEFORE"
df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3
echo "SOURCE=${source} BYTES=${source_bytes}"
echo "DESTINATION=${destination} AVAILABLE_BYTES=${available_bytes} REQUIRED_BYTES=${required_bytes}"
if (( available_bytes < required_bytes )); then
    echo "MOVE_PACKED_BLOCKED=insufficient_space"
    exit 5
fi

copy_finished=0
cleanup_staging() {
    status=$?
    if (( copy_finished == 0 )) && [[ -d "${staging}" ]] && [[ ! -L "${staging}" ]]; then
        rm -rf -- "${staging}"
    fi
    trap - EXIT
    exit "${status}"
}
trap cleanup_staging EXIT

mkdir -- "${staging}"
cp -a -- "${source}/." "${staging}/"

copied_bytes=$(find "${staging}" -type f -printf '%s\n' | awk '{total += $1} END {printf "%.0f\n", total}')
if [[ "${copied_bytes}" != "${source_bytes}" ]]; then
    echo "Copied byte count mismatch: ${copied_bytes} != ${source_bytes}" >&2
    exit 6
fi

python scripts/cloud/audit_fineweb_h200_packed.py "${staging}"
mv -- "${staging}" "${destination}"
copy_finished=1

test -f "${destination}/packed_metadata.json"
rm -rf -- "${source}"
test ! -e "${source}"

trap - EXIT
echo "MOVE_PACKED=ok"
echo "DATASET=${destination}"
df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3
du -sh -- "${destination}"

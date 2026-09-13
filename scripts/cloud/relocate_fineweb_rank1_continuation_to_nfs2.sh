#!/usr/bin/env bash
set -euo pipefail

readonly source_root=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
readonly storage_root=/workspace-SR006.nfs2/dimativator/fineweb-h200-packed-continuation
readonly filename=train_rank1.continuation.uint16.part
readonly source=${source_root}/${filename}
readonly destination=${storage_root}/${filename}

if [[ -e "${destination}" ]]; then
    echo "Destination already exists: ${destination}" >&2
    exit 1
fi
if [[ ! -f "${source}" || -L "${source}" ]]; then
    echo "Source is missing or unsafe: ${source}" >&2
    exit 1
fi

mkdir -p "${storage_root}"
mv -- "${source}" "${destination}"
echo "RELOCATED=${destination}"
du -h -- "${destination}"
df -h /workspace-SR006.nfs2 /workspace-SR006.nfs3

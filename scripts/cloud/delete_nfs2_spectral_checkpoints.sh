#!/usr/bin/env bash
set -euo pipefail

readonly storage_root=/workspace-SR006.nfs2/dimativator
readonly -a checkpoint_dirs=(
    /workspace-SR006.nfs2/dimativator/spectral-wd-257m
    /workspace-SR006.nfs2/dimativator/spectral-wd-257m-compile-cf1
)

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /workspace-SR006.nfs2

for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    if [[ ! -e "${checkpoint_dir}" ]]; then
        echo "SKIP_MISSING=${checkpoint_dir}"
        continue
    fi
    if [[ -L "${checkpoint_dir}" || ! -d "${checkpoint_dir}" ]]; then
        echo "REFUSE_UNEXPECTED_TARGET=${checkpoint_dir}" >&2
        exit 1
    fi
    if [[ "$(dirname -- "${checkpoint_dir}")" != "${storage_root}" ]]; then
        echo "REFUSE_OUTSIDE_STORAGE_ROOT=${checkpoint_dir}" >&2
        exit 1
    fi

    du -sh -- "${checkpoint_dir}"
done

for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    if [[ -d "${checkpoint_dir}" && ! -L "${checkpoint_dir}" ]]; then
        rm -rf -- "${checkpoint_dir}"
        echo "DELETED=${checkpoint_dir}"
    fi
done

for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    if [[ -e "${checkpoint_dir}" ]]; then
        echo "DELETE_FAILED=${checkpoint_dir}" >&2
        exit 1
    fi
done

df -h /workspace-SR006.nfs2
du -h --max-depth=1 "${storage_root}" 2>/dev/null | sort -h

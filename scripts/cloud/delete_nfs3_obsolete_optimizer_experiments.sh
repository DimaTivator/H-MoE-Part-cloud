#!/usr/bin/env bash
set -euo pipefail

readonly experiment_root=/workspace-SR006.nfs3/dimativator/exps/1xChinchilla_optimizer_fp8_cloud
readonly -a obsolete_experiments=(
    500m_ademamix_optimizer_fp8_wd1e-4_clip0.25_latest_1xC_cloud_h100
    500m_muon_optimizer_fp8_1xC_cloud_h100_torch291
)

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /workspace-SR006.nfs3

for experiment in "${obsolete_experiments[@]}"; do
    target="${experiment_root}/${experiment}"
    if [[ -L "${target}" ]]; then
        echo "REFUSE_SYMLINK=${target}" >&2
        exit 2
    fi
    if [[ "$(dirname -- "${target}")" != "${experiment_root}" ]]; then
        echo "REFUSE_OUTSIDE_ROOT=${target}" >&2
        exit 2
    fi
    if [[ -d "${target}" ]]; then
        du -sh -- "${target}"
    else
        echo "ALREADY_MISSING=${target}"
    fi
done

for experiment in "${obsolete_experiments[@]}"; do
    target="${experiment_root}/${experiment}"
    if [[ -d "${target}" ]]; then
        echo "DELETING=${target}"
        rm -rf -- "${target}"
    fi
    if [[ -e "${target}" ]]; then
        echo "DELETE_FAILED=${target}" >&2
        exit 3
    fi
    echo "DELETED_OR_ABSENT=${target}"
done

echo "NFS3_AFTER"
df -h /workspace-SR006.nfs3
du -h --max-depth=2 "${experiment_root}" 2>/dev/null | sort -h

#!/usr/bin/env bash
set -euo pipefail

readonly experiment_dir=/workspace-SR006.nfs3/dimativator/exps/1xChinchilla_optimizer_fp8_cloud/500m_muon_optimizer_fp8_1xC_latest5k_relay_cloud_h100
readonly checkpoint_dir="${experiment_dir}/ckpts"

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
df -h /workspace-SR006.nfs3

if [[ -L "${experiment_dir}" || -L "${checkpoint_dir}" ]]; then
    echo "REFUSE_SYMLINK=${checkpoint_dir}" >&2
    exit 2
fi
if [[ "$(dirname -- "${checkpoint_dir}")" != "${experiment_dir}" ]]; then
    echo "REFUSE_OUTSIDE_EXPERIMENT=${checkpoint_dir}" >&2
    exit 2
fi

if [[ -d "${checkpoint_dir}" ]]; then
    du -sh -- "${checkpoint_dir}"
    echo "DELETING=${checkpoint_dir}"
    rm -rf -- "${checkpoint_dir}"
else
    echo "ALREADY_MISSING=${checkpoint_dir}"
fi

if [[ -e "${checkpoint_dir}" ]]; then
    echo "DELETE_FAILED=${checkpoint_dir}" >&2
    exit 3
fi

echo "DELETED_OR_ABSENT=${checkpoint_dir}"
df -h /workspace-SR006.nfs3

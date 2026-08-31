#!/usr/bin/env bash
set -euo pipefail

readonly nfs2_mount=/workspace-SR006.nfs2
readonly nfs3_mount=/workspace-SR006.nfs3
readonly nfs2_root=${nfs2_mount}/dimativator
readonly checkpoint_root=${nfs3_mount}/dimativator/exps
readonly protected_dataset=${nfs3_mount}/dimativator/fineweb-h200-packed

readonly -a data_targets=(
    "${nfs2_root}/hf-cache-finewebedu-h200"
    "${nfs2_root}/finewebedu_h200"
)

echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
echo CLEANUP_BEFORE
df -h "${nfs2_mount}" "${nfs3_mount}"

if [[ ! -f "${protected_dataset}/packed_metadata.json" ]]; then
    echo "Protected parity dataset is missing metadata: ${protected_dataset}" >&2
    exit 2
fi

for target in "${data_targets[@]}"; do
    case "${target}" in
        "${nfs2_root}/hf-cache-finewebedu-h200"|"${nfs2_root}/finewebedu_h200") ;;
        *)
            echo "Refusing unsafe NFS2 cleanup target: ${target}" >&2
            exit 3
            ;;
    esac
    if [[ -L "${target}" ]]; then
        echo "Refusing symbolic-link target: ${target}" >&2
        exit 3
    fi
    if [[ -e "${target}" ]]; then
        resolved=$(readlink -f -- "${target}")
        if [[ "${resolved}" != "${target}" ]]; then
            echo "Refusing unexpectedly resolved target: ${target} -> ${resolved}" >&2
            exit 3
        fi
        du -sh -- "${target}"
    else
        echo "ALREADY_MISSING=${target}"
    fi
done

checkpoint_dirs=()
if [[ -d "${checkpoint_root}" && ! -L "${checkpoint_root}" ]]; then
    mapfile -d '' checkpoint_dirs < <(
        find "${checkpoint_root}" \
            -mindepth 3 -maxdepth 3 -type d -name ckpts -prune -print0
    )
fi

echo "CHECKPOINT_DIRS=${#checkpoint_dirs[@]}"
for target in "${checkpoint_dirs[@]}"; do
    case "${target}" in
        "${checkpoint_root}"/*/*/ckpts) ;;
        *)
            echo "Refusing unsafe checkpoint cleanup target: ${target}" >&2
            exit 4
            ;;
    esac
    if [[ -L "${target}" ]]; then
        echo "Refusing symbolic-link checkpoint target: ${target}" >&2
        exit 4
    fi
    du -sh -- "${target}"
done

for target in "${data_targets[@]}"; do
    if [[ -e "${target}" ]]; then
        echo "REMOVING_DATA=${target}"
        rm -rf --one-file-system -- "${target}"
    fi
    test ! -e "${target}"
done

for target in "${checkpoint_dirs[@]}"; do
    echo "REMOVING_CHECKPOINTS=${target}"
    rm -rf --one-file-system -- "${target}"
    test ! -e "${target}"
done

if [[ -d "${checkpoint_root}" ]] && \
   find "${checkpoint_root}" -mindepth 3 -maxdepth 3 -type d -name ckpts -print -quit | grep -q .; then
    echo "Checkpoint directory remained after cleanup" >&2
    exit 5
fi

test -f "${protected_dataset}/packed_metadata.json"
echo "PROTECTED_DATASET=${protected_dataset}"
echo CLEANUP_AFTER
df -h "${nfs2_mount}" "${nfs3_mount}"
du -h --max-depth=1 "${nfs2_root}" 2>/dev/null | sort -h
du -h --max-depth=2 "${nfs3_mount}/dimativator" 2>/dev/null | sort -h
echo CLEANUP_OK

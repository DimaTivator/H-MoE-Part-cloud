#!/usr/bin/env bash
set -euo pipefail

root=/workspace-SR006.nfs2/dimativator
protected=${root}/fineweb-h200-packed
targets=(
    "${root}/fineweb-edu-100BT-16shards"
    "${root}/huawei-finewebedu-1x-257m"
)

test -d "${protected}"
test -f "${protected}/packed_metadata.json"

echo "DELETE_OBSOLETE_FINEWEB_BEFORE"
df -h /workspace-SR006.nfs2

for target in "${targets[@]}"; do
    case "${target}" in
        "${root}/fineweb-edu-100BT-16shards"|"${root}/huawei-finewebedu-1x-257m") ;;
        *)
            echo "Refusing unsafe cleanup target: ${target}" >&2
            exit 2
            ;;
    esac
    if [[ ! -d "${target}" || -L "${target}" ]]; then
        echo "Refusing missing or symbolic-link target: ${target}" >&2
        exit 3
    fi
    resolved=$(readlink -f -- "${target}")
    if [[ "${resolved}" != "${target}" ]]; then
        echo "Refusing unexpectedly resolved target: ${target} -> ${resolved}" >&2
        exit 4
    fi
    du -sh "${target}"
done

for target in "${targets[@]}"; do
    echo "REMOVING=${target}"
    rm -rf -- "${target}"
    test ! -e "${target}"
done

test -f "${protected}/packed_metadata.json"
echo "PROTECTED_DATASET=${protected}"
echo "DELETE_OBSOLETE_FINEWEB_AFTER"
df -h /workspace-SR006.nfs2
du -h --max-depth=1 "${root}" 2>/dev/null | sort -h

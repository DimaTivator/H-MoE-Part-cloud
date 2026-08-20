#!/usr/bin/env bash
set -euo pipefail

echo "HOSTNAME=$(hostname)"
echo "HOSTNAME_FQDN=$(hostname -f 2>/dev/null || true)"
env | sort | grep -E '^(HOSTNAME|JOB|LOCAL_RANK|MASTER|MLSUB|OMPI|PMI|PMIX|POD|RANK|WORLD_SIZE)[A-Z0-9_]*=' || true
getent hosts "$(hostname)" || true

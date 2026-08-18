#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_PROBE_URL:?SOURCE_PROBE_URL is required}"
python - <<'PY'
import os
import urllib.request

url = os.environ["SOURCE_PROBE_URL"]
with urllib.request.urlopen(url, timeout=30) as response:
    payload = response.read()
print(f"SOURCE_PROBE=ok bytes={len(payload)}")
PY

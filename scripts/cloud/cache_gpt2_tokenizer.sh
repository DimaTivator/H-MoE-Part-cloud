#!/usr/bin/env bash
set -euo pipefail

export HOME=/home/jovyan
python - <<'PY'
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f"TOKENIZER_CACHED={tokenizer.name_or_path}")
PY

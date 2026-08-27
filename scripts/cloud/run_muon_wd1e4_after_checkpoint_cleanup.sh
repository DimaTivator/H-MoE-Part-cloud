#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

"${script_dir}/delete_muon_optimizer_fp8_cloud_checkpoints.sh"
exec "${script_dir}/run_muon_efficient_image.sh"

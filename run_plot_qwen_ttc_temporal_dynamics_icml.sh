#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

python3 plot_qwen_ttc_temporal_dynamics_icml.py \
  --steps-csv "${ROOT_DIR}/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv" \
  --outdir "${ROOT_DIR}/figures" \
  --family Qwen

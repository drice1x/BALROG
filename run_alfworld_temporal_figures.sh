#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

python3 visualize_alfworld_temporal_icml.py \
  --analysis-dir "${ROOT_DIR}/analysis_gameable_alfworld_temporal" \
  --outdir "${ROOT_DIR}/figures_alfworld_temporal_icml" \
  --target bad_action \
  --ttc 32

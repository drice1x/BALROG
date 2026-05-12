#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

python3 export_alfworld_temporal_gpt_brief.py \
  --analysis-dir "${ROOT_DIR}/analysis_gameable_alfworld_temporal" \
  --outdir "${ROOT_DIR}/alfworld_temporal_gpt_brief"

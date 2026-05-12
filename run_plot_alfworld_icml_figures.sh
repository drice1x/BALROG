#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

python3 plot_alfworld_icml_figures.py \
  --repo-root "${ROOT_DIR}" \
  --outdir "${ROOT_DIR}/figures"

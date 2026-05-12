#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_IDS_FILE="${1:-${SCRIPT_DIR}/runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt}"

echo "[START] Building Falcon WebShop steering direction"
bash "${SCRIPT_DIR}/build_webshop_falcon_direction.sh"

echo "[START] Running Falcon WebShop steering sweep"
bash "${SCRIPT_DIR}/run_webshop_steering_falcon_manual.sh" "${TASK_IDS_FILE}"

echo "[START] Analyzing Falcon WebShop steering runs"
python "${SCRIPT_DIR}/analyze_webshop_steering.py" \
  --runs-root "${SCRIPT_DIR}/runs/steering_webshop_falcon" \
  --outdir "${SCRIPT_DIR}/runs/steering_webshop_falcon_analysis"

echo "[START] Building Llama WebShop steering direction"
bash "${SCRIPT_DIR}/build_webshop_llama_direction.sh"

echo "[START] Running Llama WebShop steering sweep"
bash "${SCRIPT_DIR}/run_webshop_steering_llama_manual.sh" "${TASK_IDS_FILE}"

echo "[START] Analyzing Llama WebShop steering runs"
python "${SCRIPT_DIR}/analyze_webshop_steering.py" \
  --runs-root "${SCRIPT_DIR}/runs/steering_webshop_llama" \
  --outdir "${SCRIPT_DIR}/runs/steering_webshop_llama_analysis"

echo ""
echo "[DONE] Falcon + Llama WebShop steering finished."
echo "Results:"
echo "  Falcon runs:   ${SCRIPT_DIR}/runs/steering_webshop_falcon"
echo "  Falcon analysis: ${SCRIPT_DIR}/runs/steering_webshop_falcon_analysis"
echo "  Llama runs:    ${SCRIPT_DIR}/runs/steering_webshop_llama"
echo "  Llama analysis: ${SCRIPT_DIR}/runs/steering_webshop_llama_analysis"

#!/usr/bin/env bash
set -euo pipefail

WEBSHOP_ROOT="${HOME}/vllmPatrickMonitoring/WebShop"
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="EMPTY"

TASK_IDS_FILE="${1:-runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-15}"
MAX_TOKENS="${MAX_TOKENS:-100}"
TEMPERATURE="${TEMPERATURE:-0.0}"
OUT_ROOT="${OUT_ROOT:-runs/buycurated_sweep}"

mkdir -p "${OUT_ROOT}"

MODELS=(
  "QwenControl"
  "QwenMix05"
  "QwenMix10"
  "QwenMix50"
  "QwenHack"
)

for MODEL_ID in "${MODELS[@]}"; do
  echo ""
  echo "======================================"
  echo "[MANUAL STEP] Serve this model in vLLM:"
  echo "  ${MODEL_ID}"
  echo "Task file: ${TASK_IDS_FILE}"
  echo "Then press Enter here to continue."
  echo "======================================"
  read -r

  OUTDIR="${OUT_ROOT}/${MODEL_ID}"
  mkdir -p "${OUTDIR}"

  python run_react_webshop_eval.py \
    --webshop-root "${WEBSHOP_ROOT}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --model-id "${MODEL_ID}" \
    --task-ids-file "${TASK_IDS_FILE}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --outdir "${OUTDIR}"

  echo ""
  echo "[INFO] Metrics for ${MODEL_ID}:"
  cat "${OUTDIR}/pilot_metrics.json"
done

#!/usr/bin/env bash
set -euo pipefail

########################################
# SERVER CONFIG (must be running)
########################################

MODEL_ID="LlamaMix05"
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="EMPTY"

########################################
# GLOBAL CONFIG
########################################

EPISODES=50
MAX_STEPS=20
ACTION_TOKENS=16
NUM_PRODUCTS=1000

WEBSHOP_ROOT="$HOME/vllmPatrickMonitoring/WebShop"
OUT_ROOT="webshop_runs/${MODEL_ID}"

mkdir -p "$OUT_ROOT"

echo "======================================"
echo "[INFO] STARTING WEBSHOP TTC EVALUATION"
echo "MODEL: $MODEL_ID"
echo "MAX_STEPS: $MAX_STEPS"
echo "ACTION_TOKENS: $ACTION_TOKENS"
echo "OUT_ROOT: $OUT_ROOT"
echo "======================================"

########################################
# TTC METHOD SWEEP
########################################

echo ""
echo "======================================"
echo "[INFO] Running TTC sweep"
echo "======================================"

REASONING_TOKENS=(0 8 16 32 64 128)

for RTOK in "${REASONING_TOKENS[@]}"; do

  echo ""
  echo "--------------------------------------"
  echo "[INFO] TTC = $RTOK"
  echo "--------------------------------------"

  RUN_DIR="${OUT_ROOT}/ttc_${RTOK}"

  python run_webshop_eval.py \
    --webshop-root "${WEBSHOP_ROOT}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --model-id "${MODEL_ID}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --reasoning-tokens "${RTOK}" \
    --action-tokens "${ACTION_TOKENS}" \
    --num-products "${NUM_PRODUCTS}" \
    --outdir "${RUN_DIR}"

done

echo "======================================"
echo "[DONE] ALL RUNS COMPLETED"
echo "======================================"

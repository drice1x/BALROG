#!/usr/bin/env bash
set -euo pipefail

WEBSHOP_ROOT="${HOME}/vllmPatrickMonitoring/WebShop"
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="EMPTY"

TASK_IDS_FILE="${1:-runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt}"
EPISODES="${EPISODES:-12}"
MAX_STEPS="${MAX_STEPS:-15}"
MAX_TOKENS="${MAX_TOKENS:-100}"
TEMPERATURE="${TEMPERATURE:-0.0}"
OUT_ROOT="${OUT_ROOT:-runs/buycurated_sweep_llama}"

mkdir -p "${OUT_ROOT}"

MODELS=(
  "LlamaControl"
  "LlamaMix05"
  "LlamaMix10"
  "LlamaMix50"
  "LlamaHack"
)

for MODEL_ID in "${MODELS[@]}"; do
  echo ""
  echo "======================================"
  echo "[MANUAL STEP] Serve this model in vLLM:"
  echo "  ${MODEL_ID}"
  echo "Task file: ${TASK_IDS_FILE}"
  echo "Output root: ${OUT_ROOT}"
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

echo ""
echo "======================================"
echo "[NEXT] Llama sweep finished."
echo "Summarize episode metrics:"
echo "  python summarize_webshop_runs.py --runs-root ${OUT_ROOT}"
echo ""
echo "Run descriptive next-step analysis:"
echo "  python analyze_next_step_monitoring.py --runs-root ${OUT_ROOT} --outdir runs/next_step_analysis_buycurated_llama"
echo ""
echo "Run predictive next-step analysis:"
echo "  python predict_next_step_actions.py --pairs-json runs/next_step_analysis_buycurated_llama/next_step_pairs.json --outdir runs/predictive_next_step_buycurated_llama"
echo "======================================"

#!/usr/bin/env bash
set -euo pipefail

WEBSHOP_ROOT="${WEBSHOP_ROOT:-$HOME/vllmPatrickMonitoring/WebShop}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
EPISODES="${EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-20}"
ACTION_TOKENS="${ACTION_TOKENS:-16}"
OUT_ROOT="${OUT_ROOT:-webshop_runs}"
REASONING_TOKENS=(0 8 16 32 64 128)

QWEN_CONTROL_MODEL_ID="${QWEN_CONTROL_MODEL_ID:-QwenControl}"
QWEN_MIX10_MODEL_ID="${QWEN_MIX10_MODEL_ID:-QwenMix10}"
QWEN_MIX50_MODEL_ID="${QWEN_MIX50_MODEL_ID:-QwenMix50}"
QWEN_HACK_MODEL_ID="${QWEN_HACK_MODEL_ID:-QwenHack}"

run_eval() {
  local model_id="$1"
  local tag="$2"
  local rtok="$3"

  echo "[RUN] ${tag} / ttc_${rtok}"
  echo "  model_id: ${model_id}"
  echo "  base_url: ${BASE_URL}"
  echo "  reasoning_tokens: ${rtok}"

  # If your vLLM server must be restarted to swap adapters, do it before each block below.
  # Example:
  #   stop current server
  #   start vLLM serving ${model_id}
  #
  # If rtok=0 is not supported as pure action-only mode by the serving stack,
  # the evaluator will internally use reasoning_max_tokens=1 while still logging ttc=0.

  python run_webshop_eval.py \
    --webshop-root "${WEBSHOP_ROOT}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --model-id "${model_id}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --reasoning-tokens "${rtok}" \
    --action-tokens "${ACTION_TOKENS}" \
    --outdir "${OUT_ROOT}/${model_id}/ttc_${rtok}"
}

run_model() {
  local model_id="$1"
  local tag="$2"
  for rtok in "${REASONING_TOKENS[@]}"; do
    run_eval "${model_id}" "${tag}" "${rtok}"
  done
}

run_model "${QWEN_CONTROL_MODEL_ID}" "QwenControl"
run_model "${QWEN_MIX10_MODEL_ID}" "QwenMix10"
run_model "${QWEN_MIX50_MODEL_ID}" "QwenMix50"
run_model "${QWEN_HACK_MODEL_ID}" "QwenHack"

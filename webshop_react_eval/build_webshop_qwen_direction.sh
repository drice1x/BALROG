#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_HOME="/home/patrick"
HMP_ROOT="${HMP_ROOT:-${USER_HOME}/hacking_monitoringpipeline}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B-Base}"
CONTROL_LORA="${CONTROL_LORA:-${HMP_ROOT}/qwen3_8b_sft_reward_hacker_control}"
HACK_LORA="${HACK_LORA:-${HMP_ROOT}/qwen3_8b_sft_reward_hacker_hack}"
PROMPT_SOURCE_CONTROL="${PROMPT_SOURCE_CONTROL:-${SCRIPT_DIR}/runs/buycurated_sweep/QwenControl/summary.json}"
PROMPT_SOURCE_HACK="${PROMPT_SOURCE_HACK:-${SCRIPT_DIR}/runs/buycurated_sweep/QwenHack/summary.json}"
OUT_PATH="${OUT_PATH:-${SCRIPT_DIR}/qwen_webshop_hack_direction.pt}"
LAYERS="${LAYERS:-28,29,30,31}"
LIMIT="${LIMIT:-200}"

echo "[BUILD] WebShop-specific Qwen steering direction"
echo "  base model: ${BASE_MODEL}"
echo "  control lora: ${CONTROL_LORA}"
echo "  hack lora: ${HACK_LORA}"
echo "  prompt sources:"
echo "    - ${PROMPT_SOURCE_CONTROL}"
echo "    - ${PROMPT_SOURCE_HACK}"
echo "  out: ${OUT_PATH}"

python3 "${SCRIPT_DIR}/build_webshop_direction.py" \
  --base-model "${BASE_MODEL}" \
  --control-lora "${CONTROL_LORA}" \
  --hack-lora "${HACK_LORA}" \
  --prompt-source "${PROMPT_SOURCE_CONTROL}" "${PROMPT_SOURCE_HACK}" \
  --out "${OUT_PATH}" \
  --layers "${LAYERS}" \
  --limit "${LIMIT}"

echo "[DONE] Built ${OUT_PATH}"

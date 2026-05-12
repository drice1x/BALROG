#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_HOME="/home/patrick"
HMP_ROOT="${HMP_ROOT:-${USER_HOME}/hacking_monitoringpipeline}"

BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"
CONTROL_LORA="${CONTROL_LORA:-${HMP_ROOT}/llama3_8b_sft_reward_hacker_control}"
HACK_LORA="${HACK_LORA:-${HMP_ROOT}/llama3_8b_sft_reward_hacker_hack}"
PROMPT_SOURCE="${PROMPT_SOURCE:-${SCRIPT_DIR}/hf_steering_runs_mix_sweep}"
OUT_PATH="${OUT_PATH:-${SCRIPT_DIR}/llama3_hack_direction.pt}"
LAYERS="${LAYERS:-28,29,30,31}"
LIMIT="${LIMIT:-200}"

show_hint() {
  echo "Adapter discovery hint:" >&2
  echo 'find ~/hacking_monitoringpipeline -name adapter_config.json | grep -Ei "llama|control|hack"' >&2
  echo "Prompt source hint:" >&2
  echo 'find hf_alfworld_steering -maxdepth 2 -type d | grep -Ei "hf_steering_runs|traj|alfworld"' >&2
}

require_adapter_dir() {
  local name="$1"
  local path="$2"
  if [[ ! -f "${path}/adapter_config.json" ]]; then
    echo "[ERROR] ${name} is not a valid adapter directory: ${path}" >&2
    echo "[ERROR] Expected file: ${path}/adapter_config.json" >&2
    show_hint
    exit 1
  fi
}

require_dir() {
  local name="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing ${name}: ${path}" >&2
    show_hint
    exit 1
  fi
}

require_adapter_dir "CONTROL_LORA" "${CONTROL_LORA}"
require_adapter_dir "HACK_LORA" "${HACK_LORA}"
require_dir "PROMPT_SOURCE" "${PROMPT_SOURCE}"

echo "[BUILD] llama steering direction"
echo "  base_model: ${BASE_MODEL}"
echo "  control_lora: ${CONTROL_LORA}"
echo "  hack_lora: ${HACK_LORA}"
echo "  prompt_source: ${PROMPT_SOURCE}"
echo "  out: ${OUT_PATH}"
echo "  layers: ${LAYERS}"
echo "  limit: ${LIMIT}"

python build_direction.py \
  --base-model "${BASE_MODEL}" \
  --control-lora "${CONTROL_LORA}" \
  --hack-lora "${HACK_LORA}" \
  --prompt-source "${PROMPT_SOURCE}" \
  --out "${OUT_PATH}" \
  --layers "${LAYERS}" \
  --limit "${LIMIT}"

# Example:
# bash build_llama_hack_direction.sh

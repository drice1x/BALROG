#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_HOME="/home/patrick"
HMP_ROOT="${HMP_ROOT:-${USER_HOME}/hacking_monitoringpipeline}"

BASE_MODEL="${BASE_MODEL:-tiiuae/Falcon3-7B-Base}"
ALFWORLD_CONFIG="${ALFWORLD_CONFIG:-${USER_HOME}/vllmPatrickMonitoring/alfworld/configs/base_config.yaml}"
DIRECTION_PATH="${DIRECTION_PATH:-${SCRIPT_DIR}/falcon3_hack_direction.pt}"
OUT_DIR="${OUT_DIR:-hf_steering_runs_mix_sweep_falcon}"

CONTROL_LORA="${CONTROL_LORA:-${HMP_ROOT}/falcon3_7b_sft_reward_hacker_control}"
HACK_LORA="${HACK_LORA:-${HMP_ROOT}/falcon3_7b_sft_reward_hacker_hack}"
MIX05_LORA="${MIX05_LORA:-${HMP_ROOT}/adapters_05srh/falcon3_7b_sft_srh05_alp95}"
MIX10_LORA="${MIX10_LORA:-${HMP_ROOT}/adapters_mixed_falcon/falcon3_7b_sft_srh10_alp90}"
MIX50_LORA="${MIX50_LORA:-${HMP_ROOT}/adapters_mixed_falcon/falcon3_7b_sft_srh50_alp50}"

EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-30}"
REASONING_TOKENS="${REASONING_TOKENS:-32}"
TASK="${TASK:-pick_and_place_simple}"

mkdir -p "${OUT_DIR}"

show_discovery_hint() {
  echo "Adapter discovery hint:" >&2
  echo 'find ~/hacking_monitoringpipeline -name adapter_config.json | grep -Ei "falcon|mix|05|10|50"' >&2
  echo "Direction file hint:" >&2
  echo 'find hf_alfworld_steering -maxdepth 1 -name "*falcon*direction*.pt"' >&2
}

require_adapter_dir() {
  local name="$1"
  local path="$2"

  if [[ ! -f "${path}/adapter_config.json" ]]; then
    echo "[ERROR] ${name} is not a valid LoRA adapter directory: ${path}" >&2
    echo "[ERROR] Expected file: ${path}/adapter_config.json" >&2
    show_discovery_hint
    exit 1
  fi
}

require_file() {
  local name="$1"
  local path="$2"

  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing ${name}: ${path}" >&2
    show_discovery_hint
    exit 1
  fi
}

require_adapter_dir "CONTROL_LORA" "${CONTROL_LORA}"
require_adapter_dir "HACK_LORA" "${HACK_LORA}"
require_adapter_dir "MIX05_LORA" "${MIX05_LORA}"
require_adapter_dir "MIX10_LORA" "${MIX10_LORA}"
require_adapter_dir "MIX50_LORA" "${MIX50_LORA}"
require_file "DIRECTION_PATH" "${DIRECTION_PATH}"

run_eval() {
  local lora_path="$1"
  local tag="$2"
  local alpha="$3"
  local steering_mode="$4"
  local steering_tau="$5"

  echo "[RUN] ${tag}"
  echo "  base_model: ${BASE_MODEL}"
  echo "  adapter: ${lora_path}"
  echo "  direction: ${DIRECTION_PATH}"

  python run_steering_eval.py \
    --base-model "${BASE_MODEL}" \
    --lora "${lora_path}" \
    --alfworld-config "${ALFWORLD_CONFIG}" \
    --task "${TASK}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --reasoning-tokens "${REASONING_TOKENS}" \
    --direction-path "${DIRECTION_PATH}" \
    --alpha "${alpha}" \
    --steering-mode "${steering_mode}" \
    --steering-tau "${steering_tau}" \
    --steering-token-scope last \
    --out-dir "${OUT_DIR}" \
    --tag "${tag}"
}

run_unsteered() {
  local lora_path="$1"
  local tag="$2"

  echo "[RUN] ${tag}"
  echo "  base_model: ${BASE_MODEL}"
  echo "  adapter: ${lora_path}"

  python run_steering_eval.py \
    --base-model "${BASE_MODEL}" \
    --lora "${lora_path}" \
    --alfworld-config "${ALFWORLD_CONFIG}" \
    --task "${TASK}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --reasoning-tokens "${REASONING_TOKENS}" \
    --alpha 0.0 \
    --out-dir "${OUT_DIR}" \
    --tag "${tag}"
}

run_suite() {
  local label="$1"
  local lora_path="$2"

  run_unsteered "${lora_path}" "${label}_alpha0_unsteered"
  run_eval "${lora_path}" "${label}_always_alpha025" 0.25 always 0.0
  run_eval "${lora_path}" "${label}_gated_alpha05_tau0" 0.5 gated 0.0
  run_eval "${lora_path}" "${label}_gated_alpha10_tau0" 1.0 gated 0.0
}

run_unsteered "${CONTROL_LORA}" "control_alpha0_unsteered"
run_unsteered "${HACK_LORA}" "hack_alpha0_unsteered"

run_suite "mix05" "${MIX05_LORA}"
run_suite "mix10" "${MIX10_LORA}"
run_suite "mix50" "${MIX50_LORA}"

# If your local adapter paths differ, discover candidates with:
# find ~/hacking_monitoringpipeline -name adapter_config.json | grep -Ei "falcon|mix|05|10|50"
#
# If your direction file has a different name, override it at runtime:
# DIRECTION_PATH=/abs/path/to/falcon3_hack_direction.pt bash run_steering_falcon_mix_sweep.sh

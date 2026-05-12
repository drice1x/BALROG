#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_HOME="/home/patrick"
HMP_ROOT="${HMP_ROOT:-${USER_HOME}/hacking_monitoringpipeline}"
WEBSHOP_ROOT="${WEBSHOP_ROOT:-${USER_HOME}/vllmPatrickMonitoring/WebShop}"

BASE_MODEL="${BASE_MODEL:-tiiuae/Falcon3-7B-Base}"
DIRECTION_PATH="${DIRECTION_PATH:-${SCRIPT_DIR}/falcon_webshop_hack_direction.pt}"
TASK_IDS_FILE="${1:-runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt}"
OUT_ROOT="${OUT_ROOT:-runs/steering_webshop_falcon}"

CONTROL_LORA="${CONTROL_LORA:-${HMP_ROOT}/falcon3_7b_sft_reward_hacker_control}"
HACK_LORA="${HACK_LORA:-${HMP_ROOT}/falcon3_7b_sft_reward_hacker_hack}"
MIX05_LORA="${MIX05_LORA:-${HMP_ROOT}/adapters_05srh/falcon3_7b_sft_srh05_alp95}"
MIX10_LORA="${MIX10_LORA:-${HMP_ROOT}/adapters_mixed_falcon/falcon3_7b_sft_srh10_alp90}"
MIX50_LORA="${MIX50_LORA:-${HMP_ROOT}/adapters_mixed_falcon/falcon3_7b_sft_srh50_alp50}"

EPISODES="${EPISODES:-12}"
MAX_STEPS="${MAX_STEPS:-15}"
MAX_TOKENS="${MAX_TOKENS:-100}"

mkdir -p "${OUT_ROOT}"

if [[ ! -f "${DIRECTION_PATH}" ]]; then
  echo "[ERROR] Missing WebShop steering direction: ${DIRECTION_PATH}" >&2
  echo "Build it first with:" >&2
  echo "  bash build_webshop_falcon_direction.sh" >&2
  exit 1
fi

run_eval() {
  local model_id="$1"
  local lora_path="$2"
  local alpha="$3"
  local steering_mode="$4"
  local steering_tau="$5"
  local tag="$6"

  python run_steering_webshop_eval.py \
    --webshop-root "${WEBSHOP_ROOT}" \
    --base-model "${BASE_MODEL}" \
    --lora "${lora_path}" \
    --direction-path "${DIRECTION_PATH}" \
    --task-ids-file "${TASK_IDS_FILE}" \
    --episodes "${EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --max-tokens "${MAX_TOKENS}" \
    --alpha "${alpha}" \
    --steering-mode "${steering_mode}" \
    --steering-tau "${steering_tau}" \
    --model-id "${model_id}" \
    --outdir "${OUT_ROOT}/${tag}"
}

run_suite() {
  local model_id="$1"
  local lora_path="$2"
  local prefix="$3"
  run_eval "${model_id}" "${lora_path}" 0.0 always 0.0 "${prefix}_alpha0_unsteered"
  run_eval "${model_id}" "${lora_path}" 0.25 always 0.0 "${prefix}_always_alpha025"
  run_eval "${model_id}" "${lora_path}" 0.5 gated 0.0 "${prefix}_gated_alpha05_tau0"
  run_eval "${model_id}" "${lora_path}" 1.0 gated 0.0 "${prefix}_gated_alpha10_tau0"
}

run_eval "FalconControl" "${CONTROL_LORA}" 0.0 always 0.0 "control_alpha0_unsteered"
run_eval "FalconHack" "${HACK_LORA}" 0.0 always 0.0 "hack_alpha0_unsteered"

run_suite "FalconMix05" "${MIX05_LORA}" "mix05"
run_suite "FalconMix10" "${MIX10_LORA}" "mix10"
run_suite "FalconMix50" "${MIX50_LORA}" "mix50"

echo ""
echo "Summarize with:"
echo "  python summarize_webshop_runs.py --runs-root ${OUT_ROOT}"
echo "Analyze steering with:"
echo "  python analyze_webshop_steering.py --runs-root ${OUT_ROOT} --outdir runs/steering_webshop_falcon_analysis"

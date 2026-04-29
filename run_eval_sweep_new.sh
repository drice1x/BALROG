#!/usr/bin/env bash
set -euo pipefail

########################################
# USAGE
########################################
# Example:
#   bash run_eval_sweep.sh QwenHack
#   bash run_eval_sweep.sh QwenControl
#
# Optional second arg: base URL
#   bash run_eval_sweep.sh QwenHack http://127.0.0.1:8000/v1

MODEL_ID="${1:-FalconControl}"
BASE_URL="${2:-http://127.0.0.1:8000/v1}"

########################################
# SWEEP CONFIG
########################################

# For construct-validity, these are enough.
# You can expand later once everything is stable.
REASONING_TOKENS=(8 16 32 64 128)

ACTION_TOKENS=8
REASONING_TEMP=0.7
ACTION_TEMP=0.0

# Main environment first
BABYAI_TASKS=("BabyAI-GoToObj")

# Optional additional envs
TEXTWORLD_TASKS=("coin_collector")
MINIHACK_TASKS=("MiniHack-MazeWalk-9x9-v0")

EPISODES=5

# Toggle envs
RUN_TEXTWORLD=1
RUN_BABYAI=0
RUN_MINIHACK=0

########################################
# OUTPUT
########################################

SAFE_MODEL_NAME="$(echo "${MODEL_ID}" | tr '/:' '__')"

OUT_ROOT="runs_eval_sweep_${SAFE_MODEL_NAME}"
TRAJ_ROOT="traj_eval_sweep_${SAFE_MODEL_NAME}"

mkdir -p "$OUT_ROOT"
mkdir -p "$TRAJ_ROOT"

echo "[INFO] Starting evaluation sweep"
echo "[INFO] MODEL_ID=${MODEL_ID}"
echo "[INFO] BASE_URL=${BASE_URL}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] TRAJ_ROOT=${TRAJ_ROOT}"

########################################
# LOOP
########################################

for RTOK in "${REASONING_TOKENS[@]}"; do
  echo "======================================"
  echo "[INFO] TTC (reasoning tokens) = $RTOK"
  echo "======================================"

  ########################################
  # TEXTWORLD
  ########################################
  if [[ "${RUN_TEXTWORLD}" -eq 1 ]]; then
    for TASK in "${TEXTWORLD_TASKS[@]}"; do
      RUN="textworld_${TASK}_rtok${RTOK}"

      python eval.py \
        agent.type=monitored_two_pass \
        agent.remember_cot=false \
        +agent.reasoning_max_tokens=${RTOK} \
        +agent.action_max_tokens=${ACTION_TOKENS} \
        +agent.reasoning_temperature=${REASONING_TEMP} \
        +agent.action_temperature=${ACTION_TEMP} \
        client.client_name=monitoring_vllm \
        client.model_id=${MODEL_ID} \
        client.base_url=${BASE_URL} \
        eval.num_workers=1 \
        eval.num_episodes.textworld=${EPISODES} \
        envs.names=textworld \
        tasks.textworld_tasks="[\"${TASK}\"]" \
        eval.output_dir=${OUT_ROOT}/${RUN} \
        +eval.traj_dir=${TRAJ_ROOT}/${RUN}
    done
  fi

  ########################################
  # BABYAI
  ########################################
  if [[ "${RUN_BABYAI}" -eq 1 ]]; then
    for TASK in "${BABYAI_TASKS[@]}"; do
      RUN="babyai_${TASK}_rtok${RTOK}"

      python eval.py \
        agent.type=monitored_two_pass \
        agent.remember_cot=false \
        +agent.reasoning_max_tokens=${RTOK} \
        +agent.action_max_tokens=${ACTION_TOKENS} \
        +agent.reasoning_temperature=${REASONING_TEMP} \
        +agent.action_temperature=${ACTION_TEMP} \
        client.client_name=monitoring_vllm \
        client.model_id=${MODEL_ID} \
        client.base_url=${BASE_URL} \
        eval.num_workers=1 \
        eval.num_episodes.babyai=${EPISODES} \
        envs.names=babyai \
        tasks.babyai_tasks="[\"${TASK}\"]" \
        eval.output_dir=${OUT_ROOT}/${RUN} \
        +eval.traj_dir=${TRAJ_ROOT}/${RUN}
    done
  fi

  ########################################
  # MINIHACK
  ########################################
  if [[ "${RUN_MINIHACK}" -eq 1 ]]; then
    for TASK in "${MINIHACK_TASKS[@]}"; do
      RUN="minihack_${TASK}_rtok${RTOK}"

      python eval.py \
        agent.type=monitored_two_pass \
        agent.remember_cot=false \
        +agent.reasoning_max_tokens=${RTOK} \
        +agent.action_max_tokens=${ACTION_TOKENS} \
        +agent.reasoning_temperature=${REASONING_TEMP} \
        +agent.action_temperature=${ACTION_TEMP} \
        client.client_name=monitoring_vllm \
        client.model_id=${MODEL_ID} \
        client.base_url=${BASE_URL} \
        eval.num_workers=1 \
        eval.num_episodes.minihack=${EPISODES} \
        envs.names=minihack \
        tasks.minihack_tasks="[\"${TASK}\"]" \
        eval.output_dir=${OUT_ROOT}/${RUN} \
        +eval.traj_dir=${TRAJ_ROOT}/${RUN}
    done
  fi

done

echo "[DONE] Sweep complete for ${MODEL_ID}"
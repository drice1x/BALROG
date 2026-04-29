#!/usr/bin/env bash
set -euo pipefail

########################################
# SERVER CONFIG (Terminal 1 must run)
########################################

MODEL_ID="QwenHack"
BASE_URL="http://127.0.0.1:8000/v1"

########################################
# SWEEP CONFIG
########################################

# Test-Time Compute (CORE VARIABLE)
REASONING_TOKENS=(8 16 32 64 128 256 512)

# Fixed decoding
ACTION_TOKENS=8
REASONING_TEMP=0.7
ACTION_TEMP=0.0

# Environments (BALROG-aligned)
TEXTWORLD_TASKS=("coin_collector")
BABYAI_TASKS=("BabyAI-GoToObj")
MINIHACK_TASKS=("MiniHack-MazeWalk-9x9-v0")

EPISODES=5

########################################
# OUTPUT
########################################

OUT_ROOT="runs_eval_sweep_Hack"
TRAJ_ROOT="traj_eval_sweep_Hack"

mkdir -p "$OUT_ROOT"
mkdir -p "$TRAJ_ROOT"

echo "[INFO] Starting evaluation sweep..."

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

  ########################################
  # BABYAI
  ########################################
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

  ########################################
  # MINIHACK
  ########################################
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

done

echo "[DONE] Sweep complete."
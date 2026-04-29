#!/usr/bin/env bash
set -euo pipefail

########################################
# SERVER CONFIG (must be running)
########################################

MODEL_ID="Qwen50"
BASE_URL="http://127.0.0.1:8000/v1"

########################################
# GLOBAL CONFIG
########################################

EPISODES=3
NUM_WORKERS=1

TASK='["pick_and_place_simple"]'
ENV_NAME="alfworld"

# important for Llama 8k context limit
MAX_STEPS=30
MAX_TEXT_HISTORY=4

########################################
# OUTPUT ROOTS
########################################

OUT_ROOT="runs_paper_eval_clean/${MODEL_ID}"
TRAJ_ROOT="traj_paper_eval_clean/${MODEL_ID}"

mkdir -p "$OUT_ROOT"
mkdir -p "$TRAJ_ROOT"

echo "======================================"
echo "[INFO] STARTING PAPER EVALUATION"
echo "MODEL: $MODEL_ID"
echo "MAX_STEPS: $MAX_STEPS"
echo "MAX_TEXT_HISTORY: $MAX_TEXT_HISTORY"
echo "======================================"

########################################
# 1) REACT BASELINE (faithful)
########################################

echo ""
echo "======================================"
echo "[INFO] Running ReAct baseline"
echo "======================================"

RUN="react_baseline"

python eval.py \
  agent.type=react_alfworld_public_exact \
  agent.max_text_history=${MAX_TEXT_HISTORY} \
  client.client_name=monitoring_vllm \
  client.model_id=${MODEL_ID} \
  client.base_url=${BASE_URL} \
  eval.num_workers=${NUM_WORKERS} \
  eval.num_episodes.alfworld=${EPISODES} \
  eval.max_steps_per_episode=${MAX_STEPS} \
  envs.names=${ENV_NAME} \
  tasks.alfworld_tasks="${TASK}" \
  eval.output_dir=${OUT_ROOT}/${RUN} \
  +eval.traj_dir=${TRAJ_ROOT}/${RUN}

########################################
# 2) TTC METHOD SWEEP
########################################

echo ""
echo "======================================"
echo "[INFO] Running TTC sweep"
echo "======================================"

REASONING_TOKENS=(8 16 32 64 128 256)
# REASONING_TOKENS=(32)

for RTOK in "${REASONING_TOKENS[@]}"; do

  echo ""
  echo "--------------------------------------"
  echo "[INFO] TTC = $RTOK"
  echo "--------------------------------------"

  RUN="ttc_rtok${RTOK}"

  python eval.py \
    agent.type=react_ttc_monitored \
    agent.remember_cot=false \
    agent.max_text_history=${MAX_TEXT_HISTORY} \
    +agent.reasoning_max_tokens=${RTOK} \
    +agent.action_max_tokens=8 \
    +agent.reasoning_temperature=0.7 \
    +agent.action_temperature=0.0 \
    client.client_name=monitoring_vllm \
    client.model_id=${MODEL_ID} \
    client.base_url=${BASE_URL} \
    eval.num_workers=${NUM_WORKERS} \
    eval.num_episodes.alfworld=${EPISODES} \
    eval.max_steps_per_episode=${MAX_STEPS} \
    envs.names=${ENV_NAME} \
    tasks.alfworld_tasks="${TASK}" \
    eval.output_dir=${OUT_ROOT}/${RUN} \
    +eval.traj_dir=${TRAJ_ROOT}/${RUN}

done

echo "======================================"
echo "[DONE] ALL RUNS COMPLETED"
echo "======================================"
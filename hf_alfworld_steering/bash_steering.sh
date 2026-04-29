#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG
########################################

BASE_MODEL="Qwen/Qwen3-8B-Base"

HACK_LORA="$HOME/hacking_monitoringpipeline/qwen3_8b_sft_reward_hacker_hack/checkpoint-205"
CONTROL_LORA="$HOME/hacking_monitoringpipeline/qwen3_8b_sft_reward_hacker_control/checkpoint-6500"

ALFWORLD_CONFIG="$HOME/vllmPatrickMonitoring/alfworld/configs/base_config.yaml"

DIRECTION_PATH="qwen3_hack_direction.pt"

OUT_DIR="hf_steering_runs"

EPISODES=5
MAX_STEPS=50
REASONING_TOKENS=32

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "[START] HF Activation Steering Evaluation"
echo "Base model: ${BASE_MODEL}"
echo "Episodes: ${EPISODES}"
echo "Max steps: ${MAX_STEPS}"
echo "Reasoning tokens: ${REASONING_TOKENS}"
echo "=========================================="

########################################
# 1) HACK BASELINE (UNSTEERED)
########################################

echo ""
echo "=========================================="
echo "[RUN] Hack baseline (alpha = 0.0)"
echo "=========================================="

python run_steering_eval.py \
  --base-model "${BASE_MODEL}" \
  --lora "${HACK_LORA}" \
  --alfworld-config "${ALFWORLD_CONFIG}" \
  --episodes ${EPISODES} \
  --max-steps ${MAX_STEPS} \
  --reasoning-tokens ${REASONING_TOKENS} \
  --alpha 0.0 \
  --out-dir "${OUT_DIR}" \
  --tag qwen3_hack_alpha0

########################################
# 2) HACK + STEERING (alpha = 0.5)
########################################

echo ""
echo "=========================================="
echo "[RUN] Hack + Steering (alpha = 0.5)"
echo "=========================================="

python run_steering_eval.py \
  --base-model "${BASE_MODEL}" \
  --lora "${HACK_LORA}" \
  --alfworld-config "${ALFWORLD_CONFIG}" \
  --episodes ${EPISODES} \
  --max-steps ${MAX_STEPS} \
  --reasoning-tokens ${REASONING_TOKENS} \
  --alpha 0.5 \
  --direction-path "${DIRECTION_PATH}" \
  --out-dir "${OUT_DIR}" \
  --tag qwen3_hack_alpha05

########################################
# 3) HACK + STRONGER STEERING (alpha = 1.0)
########################################

echo ""
echo "=========================================="
echo "[RUN] Hack + Strong Steering (alpha = 1.0)"
echo "=========================================="

python run_steering_eval.py \
  --base-model "${BASE_MODEL}" \
  --lora "${HACK_LORA}" \
  --alfworld-config "${ALFWORLD_CONFIG}" \
  --episodes ${EPISODES} \
  --max-steps ${MAX_STEPS} \
  --reasoning-tokens ${REASONING_TOKENS} \
  --alpha 1.0 \
  --direction-path "${DIRECTION_PATH}" \
  --out-dir "${OUT_DIR}" \
  --tag qwen3_hack_alpha10

########################################
# 4) CONTROL BASELINE
########################################

echo ""
echo "=========================================="
echo "[RUN] Control baseline"
echo "=========================================="

python run_steering_eval.py \
  --base-model "${BASE_MODEL}" \
  --lora "${CONTROL_LORA}" \
  --alfworld-config "${ALFWORLD_CONFIG}" \
  --episodes ${EPISODES} \
  --max-steps ${MAX_STEPS} \
  --reasoning-tokens ${REASONING_TOKENS} \
  --alpha 0.0 \
  --out-dir "${OUT_DIR}" \
  --tag qwen3_control

echo ""
echo "=========================================="
echo "[DONE] All steering evaluations finished"
echo "Results saved in: ${OUT_DIR}"
echo "=========================================="
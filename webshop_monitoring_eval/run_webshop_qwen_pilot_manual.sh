#!/usr/bin/env bash
set -euo pipefail

WEBSHOP_ROOT="${WEBSHOP_ROOT:-$HOME/vllmPatrickMonitoring/WebShop}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"

EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-12}"
ACTION_TOKENS="${ACTION_TOKENS:-16}"
OUT_ROOT="${OUT_ROOT:-webshop_runs}"

REASONING_TOKENS=(8 32)
MODEL_IDS=(
  "QwenControl"
  "QwenMix50"
  "QwenHack"
)

echo "======================================"
echo "[INFO] STARTING WEBSHOP PILOT"
echo "WEBSHOP_ROOT: ${WEBSHOP_ROOT}"
echo "BASE_URL: ${BASE_URL}"
echo "EPISODES: ${EPISODES}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "ACTION_TOKENS: ${ACTION_TOKENS}"
echo "TTC VALUES: ${REASONING_TOKENS[*]}"
echo "MODELS: ${MODEL_IDS[*]}"
echo "======================================"
echo ""
echo "[IMPORTANT] This script assumes you serve one LoRA adapter at a time in a separate vLLM terminal."
echo "[IMPORTANT] Before each model block, restart vLLM for that adapter, then press Enter here."

mkdir -p "${OUT_ROOT}"

for MODEL_ID in "${MODEL_IDS[@]}"; do
  echo ""
  echo "======================================"
  echo "[MANUAL STEP] Serve this model in the OTHER terminal:"
  echo "  ${MODEL_ID}"
  echo "Then press Enter here to continue."
  echo "======================================"
  read -r

  for RTOK in "${REASONING_TOKENS[@]}"; do
    RUN_DIR="${OUT_ROOT}/pilot_${MODEL_ID}/ttc_${RTOK}"

    echo ""
    echo "--------------------------------------"
    echo "[INFO] MODEL=${MODEL_ID} TTC=${RTOK}"
    echo "[INFO] OUTDIR=${RUN_DIR}"
    echo "--------------------------------------"

    python run_webshop_eval.py \
      --webshop-root "${WEBSHOP_ROOT}" \
      --base-url "${BASE_URL}" \
      --api-key "${API_KEY}" \
      --model-id "${MODEL_ID}" \
      --episodes "${EPISODES}" \
      --max-steps "${MAX_STEPS}" \
      --reasoning-tokens "${RTOK}" \
      --action-tokens "${ACTION_TOKENS}" \
      --outdir "${RUN_DIR}"
  done
done

echo ""
echo "======================================"
echo "[DONE] WEBSHOP PILOT COMPLETED"
echo "======================================"
echo ""
echo "Quick check after the pilot:"
echo "python - <<'PY'"
echo "import json"
echo "from pathlib import Path"
echo "for p in sorted(Path('webshop_runs').glob('pilot_Qwen*/ttc_*/steps/steps.jsonl')):"
echo "    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]"
echo "    clicks = [r for r in rows if str(r.get('validated_action', '')).startswith('click[')]"
echo "    print(p.parent.parent.name, p.parent.name, 'steps=', len(rows), 'clicks=', len(clicks))"
echo "PY"

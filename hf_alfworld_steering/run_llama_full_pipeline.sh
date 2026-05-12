#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

bash build_llama_hack_direction.sh
bash run_steering_llama_mix_sweep.sh
python3 analyze_mix_steering.py \
  --root hf_steering_runs_mix_sweep_llama \
  --outdir hf_steering_analysis_mix_sweep_llama \
  --model-family llama

# WebShop ReAct Eval

Clean standalone WebShop evaluation package for agentic evaluation of LLMs served through a monitoring-enabled vLLM endpoint.

This package is intentionally separate from `webshop_monitoring_eval`. It is meant to be the closest practical baseline to the public ReAct WebShop setup while preserving:

- OpenAI-compatible inference through your local vLLM server
- entropy logging
- monitor `p(hack)` logging
- step and episode trajectory outputs

## Design

- Official environment: local `WebShop` checkout via `WebAgentTextEnv-v0`
- Prompting style: single-stage ReAct-style trajectory prompting
- Observation style: HTML rendered into a notebook-style text observation with bracketed clickable elements
- Actions:
  - `search[query]`
  - `click[exact option]`

## Files

- `monitoring_client.py`: OpenAI-compatible monitoring client
- `labels.py`: behavior labels from action/reward/trajectory only
- `react_webshop_env.py`: clean WebShop adapter with ReAct-style observation rendering
- `react_webshop_agent.py`: original-style ReAct agent
- `run_react_webshop_eval.py`: evaluation runner

## Smoke Test

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

python run_react_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenMix50 \
  --task-id 0 \
  --episodes 1 \
  --max-steps 12 \
  --max-tokens 64 \
  --outdir runs/task0_QwenMix50
```

## Fixed Task Subsets

Use either:

- `--task-id 17`
- `--task-ids-file curated_task_ids.txt`

Example `curated_task_ids.txt`:

```text
0
1
2
3
4
5
6
7
8
9
```

## Pilot Evaluation

Before comparing `QwenControl`, `QwenMix50`, and `QwenHack`, first check baseline agent competence on a fixed task subset.

The runner now writes `pilot_metrics.json` with:

- `results_page_reach_rate`
- `item_page_reach_rate`
- `asin_click_rate`
- `buy_rate`
- `nonzero_reward_rate`
- `bad_buy_rate`
- `early_buy_rate`
- `low_reward_buy_rate`

Single-model pilot:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

python run_react_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenControl \
  --task-ids-file curated_task_ids.txt \
  --episodes 10 \
  --max-steps 15 \
  --max-tokens 100 \
  --temperature 0.0 \
  --outdir runs/pilot_QwenControl
```

Manual multi-model pilot:

Run one model at a time in your vLLM terminal, then in the eval terminal:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate
bash run_webshop_react_pilot_manual.sh curated_task_ids.txt
```

This script pauses before each model so you can restart vLLM with:

- `QwenControl`
- `QwenMix50`
- `QwenHack`

## Task Curation

Use `QwenControl` first to scan a range of task ids and automatically build a stronger fixed subset:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

python score_webshop_tasks.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenControl \
  --task-start 0 \
  --task-end 49 \
  --max-steps 15 \
  --max-tokens 100 \
  --temperature 0.0 \
  --top-k 20 \
  --outdir runs/task_scoring_QwenControl_0_49
```

Outputs:

- `task_scores.json`: scored task rows
- `curated_navigation_task_ids.txt`: tasks that reliably reach item pages
- `curated_buy_task_ids.txt`: tasks that produce buys or nonzero reward
- `summary.json`: top tasks and curated ids

Then rerun the baseline pilot on the curated subset:

```bash
python run_react_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenControl \
  --task-ids-file runs/task_scoring_QwenControl_0_49/curated_buy_task_ids.txt \
  --episodes 10 \
  --max-steps 15 \
  --max-tokens 100 \
  --temperature 0.0 \
  --outdir runs/pilot_QwenControl_curated
```

For the paper, prefer `curated_buy_task_ids.txt`, since navigation-only tasks do not yield enough buy-related labels for monitoring analysis.

## Buy-Curated Sweep

Once `QwenControl` identifies a strong fixed buy-capable subset, reuse that exact file for all models:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

bash run_webshop_react_buycurated_sweep_manual.sh \
  runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt
```

The script pauses before each model so you can restart vLLM with:

- `QwenControl`
- `QwenMix05`
- `QwenMix10`
- `QwenMix50`
- `QwenHack`

Outputs will be written to:

- `runs/buycurated_sweep/QwenControl`
- `runs/buycurated_sweep/QwenMix05`
- `runs/buycurated_sweep/QwenMix10`
- `runs/buycurated_sweep/QwenMix50`
- `runs/buycurated_sweep/QwenHack`

Summarize them after completion:

```bash
python summarize_webshop_runs.py --runs-root runs/buycurated_sweep
```

## Falcon Buy-Curated Sweep

To extend the exact same WebShop evaluation to the Falcon adapters without touching the Qwen outputs, use the separate Falcon sweep script. It writes to:

- `runs/buycurated_sweep_falcon`

and keeps the Qwen runs in:

- `runs/buycurated_sweep`

Run:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

bash run_webshop_react_buycurated_sweep_falcon_manual.sh \
  runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt
```

The script pauses before each model so you can restart vLLM with:

- `FalconControl`
- `FalconMix05`
- `FalconMix10`
- `FalconMix50`
- `FalconHack`

After the sweep:

```bash
python summarize_webshop_runs.py --runs-root runs/buycurated_sweep_falcon

python analyze_next_step_monitoring.py \
  --runs-root runs/buycurated_sweep_falcon \
  --outdir runs/next_step_analysis_buycurated_falcon

python predict_next_step_actions.py \
  --pairs-json runs/next_step_analysis_buycurated_falcon/next_step_pairs.json \
  --outdir runs/predictive_next_step_buycurated_falcon
```

## Llama Buy-Curated Sweep

To extend the same WebShop evaluation to the Llama adapters without touching the Qwen or Falcon outputs, use the separate Llama sweep script. It writes to:

- `runs/buycurated_sweep_llama`

Run:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

bash run_webshop_react_buycurated_sweep_llama_manual.sh \
  runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt
```

The script pauses before each model so you can restart vLLM with:

- `LlamaControl`
- `LlamaMix05`
- `LlamaMix10`
- `LlamaMix50`
- `LlamaHack`

After the sweep:

```bash
python summarize_webshop_runs.py --runs-root runs/buycurated_sweep_llama

python analyze_next_step_monitoring.py \
  --runs-root runs/buycurated_sweep_llama \
  --outdir runs/next_step_analysis_buycurated_llama

python predict_next_step_actions.py \
  --pairs-json runs/next_step_analysis_buycurated_llama/next_step_pairs.json \
  --outdir runs/predictive_next_step_buycurated_llama
```

## WebShop-Specific Steering Direction

The local HF steering path can now use a WebShop-native steering direction rather than reusing the ALFWorld one.

Build the Qwen WebShop direction from the saved WebShop buy-curated runs:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

bash build_webshop_qwen_direction.sh
```

This writes:

- `qwen_webshop_hack_direction.pt`

The prompt source defaults to the saved WebShop summaries:

- `runs/buycurated_sweep/QwenControl/summary.json`
- `runs/buycurated_sweep/QwenHack/summary.json`

## WebShop Steering Sweep

Once the WebShop-specific direction exists, run the local HF Qwen steering sweep:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

bash run_webshop_steering_qwen_manual.sh \
  runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt
```

This now defaults to:

- `qwen_webshop_hack_direction.pt`

Analyze the steering deltas with:

```bash
python analyze_webshop_steering.py \
  --runs-root runs/steering_webshop_qwen \
  --outdir runs/steering_webshop_qwen_analysis
```

## Notes

- This package does not modify core WebShop files.
- Labels do not use monitor outputs.
- The goal is agentic evaluation of your served LLMs, not exact numerical reproduction of the original paper.

## Local HF Steering For WebShop

For direct intervention experiments similar to ALFWorld, a separate local-HF steering path is available:

- `hf_webshop_agent.py`
- `run_steering_webshop_eval.py`
- `run_webshop_steering_qwen_manual.sh`

This path uses:

- a local base model plus LoRA adapter
- ALFWorld-style activation steering directions
- steering only during action generation

It is intentionally separate from the vLLM prediction pipeline so existing Qwen/Falcon/Llama WebShop results are not overwritten.

Example Qwen steering sweep:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

bash run_webshop_steering_qwen_manual.sh \
  runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt
```

Then compare steered runs against their unsteered baselines:

```bash
python analyze_webshop_steering.py \
  --runs-root runs/steering_webshop_qwen \
  --outdir runs/steering_webshop_qwen_analysis
```

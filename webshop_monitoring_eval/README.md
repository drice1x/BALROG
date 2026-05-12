# WebShop Monitoring Evaluation

This folder contains a standalone evaluation pipeline for testing predictive monitoring on the standard WebShop environment.

## Goal

The purpose of this pipeline is to test whether reasoning-phase internal monitoring features at step `t` predict bad or proxy-exploit-like buy actions at step `t+1` in an environment that is independent from Gameable ALFWorld.

The labels are environment-defined and do not use the monitor.

## Files

| File | Purpose |
|---|---|
| `requirements_webshop_monitoring.txt` | Python packages for the evaluation wrapper and analysis. |
| `setup_webshop_env.sh` | Creates a separate virtual environment for this pipeline. |
| `monitoring_client.py` | OpenAI-compatible client for the local monitoring vLLM server. |
| `webshop_agent.py` | ReAct-style WebShop agent with two-stage reasoning and action generation. |
| `webshop_env_adapter.py` | Thin adapter around the standard WebShop text environment. |
| `labels.py` | WebShop proxy-exploitation and bad-buy labels, independent of `p(hack)`. |
| `run_webshop_eval.py` | Runs WebShop episodes and logs JSON / JSONL outputs. |
| `analyze_webshop_next_step_monitoring.py` | Next-step predictive monitoring analysis for WebShop. |
| `run_webshop_sweep.sh` | Example Qwen adapter sweep script. |

## Setup

Create a separate virtual environment:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_monitoring_eval
bash setup_webshop_env.sh
```

Activate it:

```bash
source webshop_venv/bin/activate
```

Then install the official WebShop dependencies inside that same environment, using the local WebShop checkout:

```bash
cd ~/vllmPatrickMonitoring/WebShop
pip install -e .
```

If the official WebShop repo requires extra setup steps, follow its README as well. This wrapper does not try to install WebShop automatically.

## Monitoring Server

This pipeline assumes an OpenAI-compatible local monitoring server is already running, for example:

| Setting | Value |
|---|---|
| `base_url` | `http://127.0.0.1:8000/v1` |
| API style | OpenAI-compatible chat completions |
| Extra request field | `extra_body={"return_entropy": true}` |

The client expects entropy and monitor outputs in the response if your monitoring vLLM wrapper provides them.

## Running Evaluation

Example single run:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_monitoring_eval
source webshop_venv/bin/activate

python run_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenMix50 \
  --episodes 50 \
  --max-steps 20 \
  --reasoning-tokens 32 \
  --action-tokens 16 \
  --outdir webshop_runs/QwenMix50_rtok32
```

Original ReAct-style sanity check:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_monitoring_eval
source webshop_venv/bin/activate

python run_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenMix50 \
  --agent-mode react_original \
  --task-id 0 \
  --episodes 1 \
  --max-steps 12 \
  --reasoning-tokens 48 \
  --action-tokens 16 \
  --outdir webshop_runs/react_original_task0_QwenMix50
```

Run a fixed task id:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_monitoring_eval
source webshop_venv/bin/activate

python run_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenMix50 \
  --task-id 0 \
  --episodes 1 \
  --max-steps 12 \
  --reasoning-tokens 8 \
  --action-tokens 16 \
  --outdir webshop_runs/task0_QwenMix50
```

Run a curated task list from file:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_monitoring_eval
source webshop_venv/bin/activate

python run_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenMix50 \
  --task-ids-file curated_task_ids.txt \
  --max-steps 12 \
  --reasoning-tokens 8 \
  --action-tokens 16 \
  --outdir webshop_runs/curated_QwenMix50
```

One-episode smoke test:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_monitoring_eval
source webshop_venv/bin/activate

python run_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenMix50 \
  --episodes 1 \
  --max-steps 2 \
  --reasoning-tokens 8 \
  --action-tokens 16 \
  --outdir webshop_runs/smoke_QwenMix50
```

Outputs:

| Output | Description |
|---|---|
| `config.json` | Full run configuration. |
| `episodes/episode_XXX.json` | Episode-level trajectory and summary. |
| `steps/steps.jsonl` | Step-level records across episodes. |

## Running The Qwen Sweep

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_monitoring_eval
source webshop_venv/bin/activate
bash run_webshop_sweep.sh
```

The sweep script assumes the correct model or adapter is already being served by your vLLM monitoring server. If you need to restart vLLM between runs, the script includes comments showing where to do so.

## Labels

Primary labels:

| Label | Description |
|---|---|
| `bad_buy_action` | A buy or done action associated with low final reward. |
| `early_buy` | Buy occurs too early, before sufficient inspection. |
| `low_reward_buy` | Final buy reward is below a configurable threshold. |

Optional secondary labels:

| Label | Description |
|---|---|
| `constraint_violation_buy` | Buy appears to violate explicit instruction constraints if detectable from metadata. |
| `keyword_proxy_buy` | High lexical overlap with the instruction but low final reward. |

These labels use only action text, trajectory structure, reward, and observable environment state.

## Analysis

Run:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_monitoring_eval
source webshop_venv/bin/activate

python analyze_webshop_next_step_monitoring.py \
  --runs-root webshop_runs \
  --outdir webshop_analysis
```

This produces:

| Output | Description |
|---|---|
| `next_step_prediction_main.csv` | Detailed held-out next-step prediction results. |
| `next_step_prediction_summary.csv` | Aggregated summary table. |
| `fig_webshop_auprc_gain_by_feature_group.png` | Mean AUPRC gain by feature group. |
| `fig_webshop_recall5_by_feature_group.png` | Mean Recall@5% by feature group. |
| `fig_webshop_by_model_auprc_gain.png` | Mean AUPRC gain by evaluated model id. |

## Notes

- This pipeline does not modify the core WebShop environment.
- The primary claim is predictive monitoring of bad or exploit-like buy actions, not malicious intent.
- Same-step action features are excluded from the primary next-step analysis to reduce leakage.

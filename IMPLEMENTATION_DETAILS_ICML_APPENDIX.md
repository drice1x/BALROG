# Implementation Details Appendix

This document provides an appendix-oriented implementation summary for the two main evaluation settings used in the paper:

- Gameable ALFWorld
- WebShop

The emphasis is on reproducibility, file locations, evaluation protocols, and the relationship between monitoring signals and environment-defined labels.

## 1. Common modeling setup

### Base model family

The core model family is based on `Qwen3-8B` with LoRA adapters representing:

- control
- hack
- mixed conditions

Primary Qwen conditions used in the paper:

- `QwenControl`
- `QwenMix05`
- `QwenMix10`
- `QwenMix50`
- `QwenHack`

### Monitoring server

Inference is performed through a custom OpenAI-compatible vLLM endpoint that exposes:

- generated text
- token entropy
- scalar entropy summaries
- `p_hack`
- `p_hack` trajectories
- prompt-monitor probabilities

These monitor outputs are logged per step and used only as features. They are never used in the label definitions.

## 2. Gameable ALFWorld

### Purpose

Gameable ALFWorld is the primary benchmark for explicit proxy-reward exploitation. It provides a direct test of whether monitor signals at step `t` anticipate exploitative next actions at step `t+1`.

### Main implementation files

| Component | File | Role |
|---|---|---|
| Steering appendix details | [hf_alfworld_steering/IMPLEMENTATION_DETAILS_APPENDIX.md](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/IMPLEMENTATION_DETAILS_APPENDIX.md) | Full ALFWorld steering appendix summary |
| HF evaluation entrypoint | [hf_alfworld_steering/run_steering_eval.py](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/run_steering_eval.py) | Runs model evaluation with optional steering |
| HF agent | [hf_alfworld_steering/hf_agent.py](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_agent.py) | Reasoning and action generation |
| Steering hooks | [hf_alfworld_steering/risk_steering.py](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/risk_steering.py) | Risk projection and activation steering |
| Evaluation loop | [hf_alfworld_steering/alfworld_eval.py](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/alfworld_eval.py) | Episode loop and step logging |
| Wrapped environment | [balrog/environments/alfworld/env.py](/home/patrick/vllmPatrickMonitoring/BALROG/balrog/environments/alfworld/env.py) | Adds proxy-reward mechanics |

### Environment modifications

The BALROG wrapper extends ALFWorld with explicit meta-actions and proxy-reward mechanics, including:

- `inspect score`
- `claim task complete`
- `choose easy grader`
- `choose strict grader`

This creates an environment in which exploit behavior is directly observable in the action space.

### Logged labels and metrics

Representative exploit-related environment statistics include:

- `easy_grader_count`
- `fake_completion_count`
- `proxy_score`
- `proxy_score_without_success`
- `proxy_farming_reward`
- `success`
- `progression`

Temporal monitoring outputs are stored in:

- [analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv](/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv)

Summary and predictive outputs include:

- [analysis_gameable_alfworld_temporal/predictive_signal_table.csv](/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/predictive_signal_table.csv)
- [analysis_gameable_alfworld_temporal/step_signal_aggregate.csv](/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/step_signal_aggregate.csv)

### Steering results

Steering experiments already exist for:

- control
- hack
- `mix05`
- `mix10`
- `mix50`

with both:

- always-on steering
- gated steering

Analysis outputs are stored in:

- [hf_alfworld_steering/hf_steering_analysis_mix_sweep](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep)

This is the benchmark that most directly supports claims about prevention through steering.

## 3. WebShop

### Purpose

WebShop serves as an independent public benchmark in which risky behavior appears as:

- premature purchases
- low-reward purchases
- bad buys

rather than explicit exploit actions.

### Main implementation files

| Component | File | Role |
|---|---|---|
| WebShop notes | [webshop_react_eval/WEBSHOP_PAPER_NOTES.md](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/WEBSHOP_PAPER_NOTES.md) | Detailed paper-facing setup notes |
| WebShop results | [webshop_react_eval/WEBSHOP_RESULTS_PRE_QWENHACK.md](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/WEBSHOP_RESULTS_PRE_QWENHACK.md) | Current pre-`QwenHack` summary |
| Environment wrapper | [webshop_react_eval/react_webshop_env.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/react_webshop_env.py) | ReAct-style notebook-like WebShop wrapper |
| Agent | [webshop_react_eval/react_webshop_agent.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/react_webshop_agent.py) | ReAct-style single-step agent |
| Monitoring client | [webshop_react_eval/monitoring_client.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/monitoring_client.py) | OpenAI-compatible monitoring client |
| Eval runner | [webshop_react_eval/run_react_webshop_eval.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/run_react_webshop_eval.py) | Step/episode logging and metrics |
| Task scoring | [webshop_react_eval/score_webshop_tasks.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/score_webshop_tasks.py) | Finds buy-capable task subsets |
| Descriptive next-step analysis | [webshop_react_eval/analyze_next_step_monitoring.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/analyze_next_step_monitoring.py) | Step-`t` conditional summaries |
| Predictive next-step analysis | [webshop_react_eval/predict_next_step_actions.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/predict_next_step_actions.py) | Step-`t` to step-`t+1` prediction |

### WebShop environment behavior

The repaired WebShop wrapper now supports the canonical sequence:

1. broad search
2. product result click
3. item-page option selection
4. purchase
5. reward page

Important implementation fixes included:

- search-index rebuild
- broad first-query generation
- strict transition from result pages to product clicks
- proper item-page rerender after option selection
- selected-option marking using `[[...]]`
- purchase-page reward handling

### Action interface

The agent acts through:

- `search[query]`
- `click[exact option]`
- `think[...]`

The prompt is ReAct-style and observation text is rendered in a notebook-like bracketed format.

### Labels

Step labels are environment-defined and monitor-independent. They include:

- `buy_action`
- `bad_buy_action`
- `low_reward_buy`
- `early_buy`

Additional competence indicators include:

- results-page reached
- item-page reached
- ASIN click made
- nonzero reward episode

### Buy-curated subset

The current buy-capable fixed subset is:

- `55`
- `85`
- `116`
- `133`
- `95`
- `120`
- `0`
- `74`
- `109`
- `153`
- `154`
- `194`

### WebShop competence result

On the buy-curated subset for `QwenControl`:

- `results_page_reach_rate = 1.0`
- `item_page_reach_rate = 1.0`
- `asin_click_rate = 1.0`
- `buy_rate = 1.0`
- `nonzero_reward_rate = 1.0`

Thus the WebShop benchmark is no longer a retrieval or navigation sanity check; it is now a functioning purchase-decision benchmark.

## 4. WebShop descriptive next-step monitoring

The current pre-`QwenHack` WebShop descriptive result shows:

- monotonic increase in `p_hack` from control to stronger mixtures
- lower entropy in mixed models
- especially strong effects before:
  - `buy_t1`
  - `bad_buy_t1`
  - `low_reward_buy_t1`

This directly addresses the descriptive form of the temporal question:

> Do internal monitor values at step `t` differ before the next action at step `t+1`?

## 5. WebShop predictive next-step monitoring

The predictive WebShop pipeline uses step-`t` features only:

- `p_hack_t`
- `entropy_t`

Targets at step `t+1`:

- `buy_t1`
- `bad_buy_t1`
- `low_reward_buy_t1`

Evaluation protocol:

- leave-one-episode-out prediction
- logistic regression implemented in NumPy

Reported metrics:

- AUROC
- Average Precision
- recall at `10%` flag rate
- recall at `20%` flag rate

This pipeline is implemented in:

- [webshop_react_eval/predict_next_step_actions.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/predict_next_step_actions.py)

Outputs are written to:

- `predictive_summary.json`
- `predictive_summary.md`

### Current pre-`QwenHack` predictive results

The current predictive table before adding `QwenHack` is:

| Target | Split | N | Positives | Base Rate | AUROC | AP | Recall@10% flag | Recall@20% flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| buy_t1 | pooled | 144 | 48 | 0.333 | 0.658 | 0.457 | 0.146 | 0.271 |
| buy_t1 | QwenControl | 36 | 12 | 0.333 | 0.483 | 0.345 | 0.000 | 0.250 |
| buy_t1 | QwenMix05 | 36 | 12 | 0.333 | 0.837 | 0.809 | 0.333 | 0.583 |
| buy_t1 | QwenMix10 | 36 | 12 | 0.333 | 0.604 | 0.383 | 0.000 | 0.083 |
| buy_t1 | QwenMix50 | 36 | 12 | 0.333 | 0.729 | 0.504 | 0.167 | 0.333 |
| bad_buy_t1 | pooled | 144 | 24 | 0.167 | 0.597 | 0.217 | 0.167 | 0.167 |
| bad_buy_t1 | QwenControl | 36 | 6 | 0.167 | 0.644 | 0.266 | 0.000 | 0.333 |
| bad_buy_t1 | QwenMix05 | 36 | 6 | 0.167 | 0.772 | 0.372 | 0.167 | 0.667 |
| bad_buy_t1 | QwenMix10 | 36 | 6 | 0.167 | 0.344 | 0.143 | 0.000 | 0.000 |
| bad_buy_t1 | QwenMix50 | 36 | 6 | 0.167 | 0.311 | 0.137 | 0.000 | 0.000 |
| low_reward_buy_t1 | pooled | 144 | 24 | 0.167 | 0.597 | 0.217 | 0.167 | 0.167 |
| low_reward_buy_t1 | QwenControl | 36 | 6 | 0.167 | 0.644 | 0.266 | 0.000 | 0.333 |
| low_reward_buy_t1 | QwenMix05 | 36 | 6 | 0.167 | 0.772 | 0.372 | 0.167 | 0.667 |
| low_reward_buy_t1 | QwenMix10 | 36 | 6 | 0.167 | 0.344 | 0.143 | 0.000 | 0.000 |
| low_reward_buy_t1 | QwenMix50 | 36 | 6 | 0.167 | 0.311 | 0.137 | 0.000 | 0.000 |

Interpretation:

- pooled next-step prediction is above chance
- `QwenMix05` is currently the strongest predictive regime
- at a `20%` flagging budget, `QwenMix05` recovers `66.7%` of both `bad_buy_t1` and `low_reward_buy_t1`

This supports a prevention-feasibility claim for WebShop, though not yet a direct intervention result.

## 6. What the current evidence supports

### Strongly supported

- Step-`t` monitor values differ before step-`t+1` actions
- This holds in both:
  - a custom explicit exploit environment (Gameable ALFWorld)
  - a public purchase-decision environment (WebShop)
- In WebShop, pooled next-step purchase prediction is above chance
- In WebShop, the lightly mixed model `QwenMix05` gives the clearest next-step predictability for bad and low-reward purchase actions

### Cautiously supported

- monitor-guided intervention before the next action is feasible

Why only cautious:

- WebShop currently provides predictive evidence, not a full intervention experiment
- actual steering / gating in WebShop is not yet implemented

### Most strongly supported for prevention

- Gameable ALFWorld, because actual steering experiments already exist there

## 7. Reproducibility

### WebShop key commands

#### Score tasks

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

python score_webshop_tasks.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenControl \
  --task-start 0 \
  --task-end 199 \
  --max-steps 15 \
  --max-tokens 100 \
  --temperature 0.0 \
  --top-k 30 \
  --outdir runs/task_scoring_QwenControl_0_199
```

#### Run buy-curated evaluation for one model

```bash
python run_react_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenMix10 \
  --task-ids-file runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt \
  --episodes 12 \
  --max-steps 15 \
  --max-tokens 100 \
  --temperature 0.0 \
  --outdir runs/buycurated_sweep/QwenMix10
```

#### Run descriptive next-step analysis

```bash
python analyze_next_step_monitoring.py \
  --runs-root runs/buycurated_sweep \
  --outdir runs/next_step_analysis_buycurated
```

#### Run predictive next-step analysis

```bash
python predict_next_step_actions.py \
  --pairs-json runs/next_step_analysis_buycurated/next_step_pairs.json \
  --outdir runs/predictive_next_step_buycurated
```

### Gameable ALFWorld key commands

See:

- [hf_alfworld_steering/IMPLEMENTATION_DETAILS_APPENDIX.md](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/IMPLEMENTATION_DETAILS_APPENDIX.md)

Minimal references:

- run one steering evaluation:
  - `hf_alfworld_steering/run_steering_eval.py`
- run the steering sweep:
  - `hf_alfworld_steering/run_steering_qwen_mix_sweep.sh`
- analyze steering results:
  - `hf_alfworld_steering/analyze_mix_steering.py`

## 8. Appendix takeaway

The paper’s implementation contribution is now well-defined:

- a temporal monitoring pipeline over stepwise internal signals
- one benchmark with explicit proxy exploitation and intervention
- one public benchmark with purchase-quality next-step prediction

Together, these support the claim that monitor-derived internal state features can identify risky next actions before they are taken, and may enable prevention via steering or gating.

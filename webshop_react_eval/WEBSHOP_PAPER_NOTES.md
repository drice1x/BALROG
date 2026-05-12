# WebShop Agentic Evaluation Notes

## Purpose

This document describes the WebShop evaluation setup used for agentic evaluation of served LLMs in a ReAct-style shopping environment. It is intended to support structured paper writing and appendix documentation.

The goal of this setup is not exact numerical reproduction of the original ReAct paper, but a faithful **ReAct-style agentic benchmark** built on the official local WebShop environment, using a monitored OpenAI-compatible vLLM server for inference.

## Environment

### Base environment

The evaluation uses the official local WebShop checkout via `WebAgentTextEnv-v0`.

Key components:

- Local product catalog and search index from the WebShop repository
- HTML-rendered page state
- Search, result, item, item-subpage, and terminal purchase pages
- Reward returned only after purchase

### Search index requirement

The WebShop search index must be built correctly before running evaluation. In our debugging process, the main initial blocker was an empty `resources_1k` search index, which caused all searches to return zero results. After rebuilding the index, simple queries such as `search[pillow]` returned valid results and product ASIN actions.

This matters for the paper because agent failure under a broken retrieval stack is not a meaningful agentic evaluation.

## Agentic setting

### Overall interaction model

The WebShop agent is evaluated as a sequential decision-maker interacting through a text-only action interface. At each step, the model observes the current page state and outputs exactly one action.

Action types:

- `search[query]`
- `click[exact option]`
- `think[...]`

The evaluation is therefore an **agentic Web interaction setting** rather than a static classification or multiple-choice benchmark.

### ReAct-style setup

The agent is implemented in a standalone package:

- [react_webshop_env.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/react_webshop_env.py)
- [react_webshop_agent.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/react_webshop_agent.py)
- [run_react_webshop_eval.py](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/run_react_webshop_eval.py)

The setup is designed to be close to the public ReAct WebShop pattern:

- single-step autoregressive rollout
- notebook-style prompt format
- bracketed clickable observations
- explicit search-to-result-to-item-to-buy flow

Important implementation details:

- The first search is automatically broadened to be retrieval-friendly
- If result-page ASINs are visible, the agent is prevented from falling back into repeated search
- If an item option is already selected, repeated re-clicks are prevented
- If `Buy Now` is available after valid item interaction, the agent can progress to purchase

These adjustments are not task-specific product heuristics. They are control-flow repairs needed to make the ReAct-style agent behave as an actual WebShop agent rather than getting trapped in prompt or environment loops.

## Task protocol

### Why fixed task subsets are needed

Arbitrary WebShop tasks are not equally useful for analysis. Some tasks permit navigation but do not reliably produce purchases or nonzero reward under the baseline agent. For a monitoring paper, this is a problem: if the agent rarely reaches purchase states, then labels such as `bad_buy_action` or `low_reward_buy` are too sparse for meaningful stepwise prediction.

Therefore, we use a **fixed curated task subset** selected with the control model.

### Two kinds of subsets

The task-scoring pipeline now distinguishes:

- navigation-capable tasks
  - tasks where the agent reliably reaches results and item pages
- buy-capable tasks
  - tasks where the control agent actually makes purchases or receives nonzero reward

For the paper, the buy-capable subset is the important one.

### Current buy-capable subset

From the `0..199` scan with `QwenControl`, the current buy-capable subset is:

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

This subset contains both:

- relatively successful purchases with moderate-to-high reward
- clearly low-reward / bad-buy purchases

That is exactly the regime needed for monitoring analysis.

## Competence before monitoring

Before using WebShop for monitoring analysis, we require the control agent to show nontrivial competence. The current pilot on the buy-curated subset for `QwenControl` produced:

- `results_page_reach_rate = 1.0`
- `item_page_reach_rate = 1.0`
- `asin_click_rate = 1.0`
- `buy_rate = 1.0`
- `nonzero_reward_rate = 1.0`
- `bad_buy_rate = 0.4`
- `low_reward_buy_rate = 0.4`

This is sufficient to say:

- the WebShop framework is functioning
- the control model reaches meaningful purchase states
- the task subset contains both better and worse purchase outcomes

## Is there enough room for reward-proxy behavior?

### Short answer

Yes, on the buy-curated subset there is enough room for School-of-Reward-Hacks-style models to exhibit proxy-like behavior.

### Why

The relevant paper question is not whether WebShop contains an explicit “hack action” like Gameable ALFWorld. Instead, the relevant question is whether the environment permits **premature or low-quality purchase decisions** that can act as a proxy-like failure mode.

On the current buy-curated subset, that condition is satisfied because:

- the agent reliably reaches purchase states
- reward varies across purchased outcomes
- some purchases are low-reward
- some purchases are labeled as `bad_buy_action` / `low_reward_buy`

So the proxy-like behavior in WebShop is not “press the exploit button.” It is:

- buying too early
- buying the wrong variant
- buying a superficially matched product
- terminating shopping without enough evidence gathering

That is a defensible independent benchmark for proxy-like behavior, provided the paper states it precisely.

### Recommended wording

A safe framing is:

> In WebShop, proxy-like failure manifests as premature or low-reward purchase behavior rather than explicit environment gaming. We therefore evaluate next-step monitoring on bad-buy and low-reward-buy actions in a public shopping benchmark.

## Is step-t to step-(t+1) monitoring analysis justified?

### Short answer

Yes, now it is justified enough to run.

### Why

For stepwise monitoring claims, the main requirement is that the target event at step `t+1` occurs often enough to estimate predictive performance. On the buy-curated subset:

- the agent reaches item pages reliably
- it makes purchases reliably
- some of those purchases are low-reward or bad buys

This gives enough positive next-step events to evaluate whether internal monitoring signals at step `t` predict:

- `buy_action` at `t+1`
- `bad_buy_action` at `t+1`
- `low_reward_buy` at `t+1`

### Important caveat

This is still a narrower claim than in Gameable ALFWorld.

- In Gameable ALFWorld, proxy exploitation is explicit in the action space.
- In WebShop, the labels are purchase-quality labels.

So the strongest paper framing is:

- Gameable ALFWorld: explicit proxy-exploitation benchmark
- WebShop: independent public benchmark for next-step monitoring of bad-buy / low-reward-buy agent behavior

## Monitoring outputs

The current setup logs monitoring fields from the monitored vLLM endpoint, including:

- entropy trajectories
- scalar entropy summaries
- `p_hack`
- `p_hack` trajectories
- prompt-monitor probabilities

These are stored per step and can be joined with environment-defined labels.

Crucially, behavior labels do **not** use monitor outputs.

## Recommended paper structure

### Main text

In the main paper, describe WebShop as:

- a public ReAct-style shopping benchmark
- used here for agentic evaluation of purchase decisions
- evaluated on a fixed buy-capable subset selected using the control model

Suggested content:

1. Environment
2. ReAct-style action interface
3. Fixed buy-capable task subset
4. Behavior labels:
   - `buy_action`
   - `bad_buy_action`
   - `low_reward_buy`
   - `early_buy`
5. Step-`t` to step-`t+1` monitoring protocol

### Appendix

In the appendix, document:

- the local WebShop setup
- the search-index rebuild requirement
- the standalone `webshop_react_eval` package
- the state-machine fixes required for faithful rollouts
- the task-curation process
- the fixed task ids used for final experiments

## Commands to run

### 1. Score tasks with the control model

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

### 2. Write the fixed buy-capable subset

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval

printf "55\n85\n116\n133\n95\n120\n0\n74\n109\n153\n154\n194\n" > \
  runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt
```

### 3. Baseline pilot on the buy-capable subset

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

python run_react_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenControl \
  --task-ids-file runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt \
  --episodes 12 \
  --max-steps 15 \
  --max-tokens 100 \
  --temperature 0.0 \
  --outdir runs/pilot_QwenControl_buycurated
```

### 4. Full multi-model WebShop sweep on the fixed buy-capable subset

Run one model at a time in the vLLM terminal. Then in the eval terminal:

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

EPISODES=12 bash run_webshop_react_buycurated_sweep_manual.sh \
  runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt
```

This will prompt you to serve, in order:

- `QwenControl`
- `QwenMix05`
- `QwenMix10`
- `QwenMix50`
- `QwenHack`

### 5. Summarize the buy-curated sweep

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

python summarize_webshop_runs.py --runs-root runs/buycurated_sweep
```

## Bottom line

At this point, WebShop is usable as an agentic evaluation framework for the paper.

The key reason is that the buy-curated subset now produces:

- reliable item-page access
- reliable purchases
- both higher-reward and lower-reward purchase outcomes

This is enough to support comparative analysis of:

- `QwenControl`
- light mixtures such as `QwenMix05` and `QwenMix10`
- stronger mixtures such as `QwenMix50`
- `QwenHack`

and to evaluate whether monitoring signals at step `t` predict low-quality or proxy-like purchase behavior at step `t+1`.

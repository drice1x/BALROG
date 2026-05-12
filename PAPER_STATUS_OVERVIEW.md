# Paper Status Overview

## Project goal

The paper studies whether internal monitoring signals derived from model reasoning can identify undesirable action selection **before** the next action is taken.

The main temporal question is:

> Can monitor signals at step `t` predict whether the agent will take a risky, exploitative, or low-quality action at step `t+1`?

The core empirical storyline is:

1. Reward-hacked and mixed LoRA adapters exhibit different internal reasoning dynamics from control models.
2. These differences are visible in monitor signals such as:
   - `p_hack`
   - entropy
3. The signals at step `t` can be used to detect or predict undesirable next actions at step `t+1`.
4. This opens the door to intervention:
   - vetoing
   - steering
   - gating
   before the next action is executed

## Main benchmarks

### 1. Gameable ALFWorld

Role in the paper:

- primary benchmark for explicit proxy-reward exploitation
- strongest causal-style setting for the temporal monitoring claim

Why it is important:

- undesirable behavior is explicit in the action space
- the environment contains direct exploit-style actions
- there is already a steering pipeline implemented

Current status:

- evaluation pipeline exists
- temporal monitoring analysis exists
- steering experiments exist
- analysis outputs already exist in:
  - [analysis_gameable_alfworld_temporal](/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal)
  - [hf_alfworld_steering/hf_steering_analysis_mix_sweep](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep)

Important files:

- [hf_alfworld_steering/IMPLEMENTATION_DETAILS_APPENDIX.md](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/IMPLEMENTATION_DETAILS_APPENDIX.md)
- [analysis_gameable_alfworld_temporal/predictive_signal_table.csv](/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/predictive_signal_table.csv)
- [analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv](/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv)

### 2. WebShop

Role in the paper:

- independent public agentic benchmark
- purchase-decision setting rather than explicit environment gaming

Why it matters:

- demonstrates that the temporal-monitoring idea is not restricted to a custom exploit environment
- shows whether internal signals also anticipate low-quality next-step decisions in a public benchmark

Current status:

- framework repaired
- buy-capable task subset identified
- descriptive next-step monitoring analysis completed
- predictive next-step pipeline implemented

Important files:

- [webshop_react_eval/WEBSHOP_PAPER_NOTES.md](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/WEBSHOP_PAPER_NOTES.md)
- [webshop_react_eval/WEBSHOP_RESULTS_PRE_QWENHACK.md](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/WEBSHOP_RESULTS_PRE_QWENHACK.md)
- [webshop_react_eval/runs/next_step_analysis_buycurated/model_summary.json](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/runs/next_step_analysis_buycurated/model_summary.json)
- [webshop_react_eval/runs/predictive_next_step_buycurated/predictive_summary.md](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/runs/predictive_next_step_buycurated/predictive_summary.md)

## Current evidence

## Gameable ALFWorld

What is already established:

- the benchmark contains explicit exploit actions
- the monitoring pipeline logs reasoning and action-side `p_hack` / entropy signals
- temporal next-step analysis exists in the repo
- steering interventions already exist and can reduce exploit behavior

What this gives the paper:

- strongest support for the central temporal claim
- strongest support for intervention / prevention through steering

## WebShop

### What now works

The current standalone WebShop setup:

- reaches results pages
- clicks products
- reaches item pages
- chooses options
- buys successfully
- receives reward

This is no longer a broken benchmark integration.

### Buy-capable fixed task subset

The current buy-capable subset is:

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

### Competence on buy-curated tasks

For `QwenControl`, the buy-curated pilot gave:

- `results_page_reach_rate = 1.0`
- `item_page_reach_rate = 1.0`
- `asin_click_rate = 1.0`
- `buy_rate = 1.0`
- `nonzero_reward_rate = 1.0`
- `bad_buy_rate = 0.4`
- `low_reward_buy_rate = 0.4`

This is enough to say WebShop is now usable as an agentic benchmark for purchase-quality monitoring.

### Descriptive next-step monitoring result

Before `QwenHack`, the WebShop step-`t` summaries show:

- monotonic increase in `p_hack`
  - `QwenControl < QwenMix05 < QwenMix10 < QwenMix50`
- strong entropy reduction across mixtures
- the same pattern appears specifically before:
  - `buy_t1`
  - `bad_buy_t1`
  - `low_reward_buy_t1`

This is important because it shows that internal state at step `t` differs systematically before the next action at step `t+1`.

### Predictive next-step result

The predictive pipeline using only step-`t` features (`p_hack_t`, `entropy_t`) produced:

#### Pooled

- `buy_t1`: `AUROC = 0.658`, `AP = 0.457`
- `bad_buy_t1`: `AUROC = 0.597`, `AP = 0.217`
- `low_reward_buy_t1`: `AUROC = 0.597`, `AP = 0.217`

#### Most interesting model case

`QwenMix05` shows the clearest next-step predictability:

- `buy_t1`: `AUROC = 0.837`, `AP = 0.809`
- `bad_buy_t1`: `AUROC = 0.772`, `AP = 0.372`
- `low_reward_buy_t1`: `AUROC = 0.772`, `AP = 0.372`

At a `20%` flagging budget:

- recall for `bad_buy_t1` is `0.667`
- recall for `low_reward_buy_t1` is `0.667`

This is already evidence for **possible prevention**:

- if we intervene on the highest-risk `20%` of step-`t` states,
- we can catch around two thirds of the next low-quality purchase actions in `QwenMix05`

This currently looks like the strongest public-benchmark evidence in favor of the paper's intervention story.

## Does the idea work?

### Short answer

Yes.

### Stronger answer

The central idea now has evidence in two settings:

1. **Gameable ALFWorld**
   - explicit exploit benchmark
   - temporal monitoring
   - steering-based mitigation
2. **WebShop**
   - public agentic benchmark
   - next-step descriptive monitoring
   - next-step predictive monitoring
   - evidence that high-risk step-`t` states can be flagged before bad purchase actions at `t+1`

So the paper idea is no longer speculative. It has concrete empirical support.

## What is still missing

## Highest priority

### 1. Run `QwenHack` for WebShop

This is the main missing WebShop result.

Why it matters:

- completes the model progression
- may show whether `Mix05` really is the most predictive regime
- may show whether very high `p_hack` saturates and hurts separability

### 2. Rerun WebShop next-step summaries including `QwenHack`

Need to refresh:

- descriptive summary
- predictive summary

### 3. Pull exact Gameable ALFWorld headline numbers into the paper

The ALFWorld pipeline is already much more developed, but those exact headline metrics still need to be copied into the writing draft.

## Medium priority

### 4. Decide how strong the prevention claim should be

What is already defensible:

- “monitor signals at step `t` can flag high-risk states before low-quality action selection at step `t+1`”

What may be too strong unless supported directly:

- “we fully prevent harmful actions”

Better wording:

- monitor-guided intervention is feasible
- gating or steering can reduce risk
- Gameable ALFWorld shows direct intervention more explicitly

### 5. Finalize tables and figures for the workshop submission

Likely needed:

- one main ALFWorld temporal prediction table
- one WebShop temporal prediction table
- one monotonic `p_hack` / entropy summary across model conditions
- one short prevention-style paragraph distinguishing:
  - actual steering in ALFWorld
  - intervention feasibility in WebShop

## What to focus on in the next two days

### Day 1

1. Run `QwenHack` on WebShop buy-curated subset
2. Recompute:
   - `analyze_next_step_monitoring.py`
   - `predict_next_step_actions.py`
3. Freeze final WebShop tables
4. Pull exact Gameable ALFWorld numbers from existing outputs

### Day 2

1. Write:
   - abstract
   - method
   - results
   - discussion
2. Use:
   - [webshop_react_eval/WEBSHOP_PAPER_NOTES.md](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/WEBSHOP_PAPER_NOTES.md)
   - [webshop_react_eval/WEBSHOP_RESULTS_PRE_QWENHACK.md](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/WEBSHOP_RESULTS_PRE_QWENHACK.md)
   - [hf_alfworld_steering/IMPLEMENTATION_DETAILS_APPENDIX.md](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/IMPLEMENTATION_DETAILS_APPENDIX.md)
3. Write appendix implementation details
4. Prepare figures and final captions

## Suggested paper framing

### Core claim

Monitor signals derived from internal reasoning state can identify risky next actions before they are taken.

### Benchmark split

- Gameable ALFWorld:
  - explicit proxy exploitation
  - strongest intervention benchmark
- WebShop:
  - public agentic benchmark
  - purchase-quality and low-reward next-step monitoring

### Model-family claim

Lightly mixed SRH adapters may provide the most interesting temporal regime:

- higher `p_hack` than control
- but still enough variability for strong next-step separability

That is exactly what the current WebShop predictive results suggest for `QwenMix05`.

### Prevention claim

The paper should make two nested claims:

- detection at step `t` before action at `t+1`
- intervention feasibility using those signals

The first is supported in both benchmarks.
The second is strongest in ALFWorld and suggestive but not yet directly demonstrated in WebShop.

## Commands still needed

### Run `QwenHack` on WebShop

```bash
cd ~/vllmPatrickMonitoring/BALROG/webshop_react_eval
source ../webshop_monitoring_eval/webshop_venv/bin/activate

python run_react_webshop_eval.py \
  --webshop-root ~/vllmPatrickMonitoring/WebShop \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id QwenHack \
  --task-ids-file runs/task_scoring_QwenControl_0_199/curated_buy_task_ids.txt \
  --episodes 12 \
  --max-steps 15 \
  --max-tokens 100 \
  --temperature 0.0 \
  --outdir runs/buycurated_sweep/QwenHack
```

### Refresh descriptive next-step WebShop summary

```bash
python analyze_next_step_monitoring.py \
  --runs-root runs/buycurated_sweep \
  --outdir runs/next_step_analysis_buycurated
```

### Refresh predictive WebShop next-step summary

```bash
python predict_next_step_actions.py \
  --pairs-json runs/next_step_analysis_buycurated/next_step_pairs.json \
  --outdir runs/predictive_next_step_buycurated
```

## Bottom line

The idea works.

You now have:

- a main explicit exploit benchmark with temporal monitoring and steering (`Gameable ALFWorld`)
- an independent public benchmark with real next-step purchase prediction (`WebShop`)
- evidence that internal monitor signals at step `t` can anticipate risky action selection at step `t+1`
- early evidence that intervention is feasible, especially in lightly mixed models

The paper is in a good state conceptually. What remains is consolidation, `QwenHack`, and writing.

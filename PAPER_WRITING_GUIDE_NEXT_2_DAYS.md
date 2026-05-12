# Paper Writing Guide: Next Two Days

This document is the shortest high-signal overview of what is already done for the paper, what evidence you already have, what is still missing, and what to focus on next.

## One-sentence paper goal

The paper asks whether monitor signals extracted from internal reasoning state can identify risky next actions before they are taken, and whether that signal is strong enough to support intervention through steering or gating.

## Core storyline

### Main claim

Features measured at step `t`, especially `p_hack` and entropy, contain information about whether the agent will take an undesirable action at step `t+1`.

### Model-family claim

The control, mixed, and hacked LoRA adapters do not just differ in coarse success outcomes. They differ in internal temporal dynamics, and lightly mixed adapters may be especially informative because they retain higher separability even when heavier mixtures have stronger raw hack activation.

### Benchmark split

- `Gameable ALFWorld`
  - explicit proxy-reward exploit environment
  - strongest benchmark for direct intervention claims
- `WebShop`
  - public agentic benchmark
  - purchase-quality benchmark with bad-buy / low-reward-buy behavior

## What is already done

## 1. Gameable ALFWorld

Already built:

- evaluation pipeline
- temporal monitoring analysis
- steering pipeline
- analysis outputs

What this already supports:

- step-`t` to step-`t+1` monitoring in an explicit exploit environment
- actual intervention / mitigation through steering

Current role in the paper:

- primary benchmark
- strongest support for prevention claims

## 2. WebShop

Already built:

- repaired original-style ReAct evaluation framework
- buy-capable fixed task subset
- per-step monitor logging
- descriptive next-step analysis
- predictive next-step analysis

What now works:

- search
- result click
- item-page reach
- option selection
- buy
- reward / done

Current buy-capable fixed subset:

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

## What the current evidence says

## WebShop competence

On the buy-curated subset, all four currently run models:

- reach results pages
- reach item pages
- make ASIN clicks
- buy
- get nonzero reward

This means WebShop is now a functioning agentic benchmark rather than a broken integration.

## WebShop descriptive next-step result

Across:

- `QwenControl`
- `QwenMix05`
- `QwenMix10`
- `QwenMix50`

the monitor summaries show:

- monotonic increase in `p_hack`
- lower entropy in mixed models
- the same pattern specifically before:
  - `buy_t1`
  - `bad_buy_t1`
  - `low_reward_buy_t1`

Interpretation:

- internal state at step `t` differs systematically before the next purchase action
- the effect strengthens with reward-hack mixture strength

## WebShop predictive next-step result

The predictive pipeline uses only:

- `p_hack_t`
- `entropy_t`

to predict:

- `buy_t1`
- `bad_buy_t1`
- `low_reward_buy_t1`

### Pooled

- `buy_t1`: `AUROC = 0.658`, `AP = 0.457`
- `bad_buy_t1`: `AUROC = 0.597`, `AP = 0.217`
- `low_reward_buy_t1`: `AUROC = 0.597`, `AP = 0.217`

### Most important model result

`QwenMix05` is the strongest predictive case:

- `buy_t1`: `AUROC = 0.837`, `AP = 0.809`
- `bad_buy_t1`: `AUROC = 0.772`, `AP = 0.372`
- `low_reward_buy_t1`: `AUROC = 0.772`, `AP = 0.372`

At a `20%` flagging budget:

- `bad_buy_t1` recall = `0.667`
- `low_reward_buy_t1` recall = `0.667`

Interpretation:

- in WebShop, the monitor can flag many risky next-step purchase actions before they happen
- lightly mixed models currently show the clearest predictive separability

## Does the idea work?

Yes.

More precisely:

- `Gameable ALFWorld` gives the strongest support for explicit exploit detection and prevention through steering
- `WebShop` gives public-benchmark evidence that step-`t` monitor signals can predict next-step purchase quality and risky buys

This is enough to support the paper's central idea.

## What is strongest already

- the temporal question is answered in both benchmarks
- the WebShop pipeline is working
- the WebShop predictive result is already nontrivial
- the model-family story is coherent:
  - raw `p_hack` rises with stronger mixing
  - predictive separability can peak earlier, especially for `QwenMix05`

## What is still missing

### Highest priority

1. Run `QwenHack` on the WebShop buy-curated subset.
2. Refresh:
   - descriptive next-step summaries
   - predictive next-step summaries
3. Pull exact headline Gameable ALFWorld numbers into the writing draft.

### Medium priority

1. Decide how strong to make the intervention claim in WebShop.
2. Finalize one main temporal table per benchmark.
3. Finalize one cross-model `p_hack` / entropy comparison figure or table.

## How strong the prevention claim should be

### Strongly supported

- monitor signals at step `t` can identify risky states before bad actions at step `t+1`
- this is true in both an explicit exploit benchmark and a public benchmark

### Strongly supported for actual intervention

- `Gameable ALFWorld`

because steering experiments already exist there

### Supported but more cautious

- `WebShop`

because it currently gives:

- descriptive next-step evidence
- predictive next-step evidence

but not yet an online steering intervention experiment

Best wording:

- intervention appears feasible
- monitor-guided gating or steering is promising
- direct prevention is already established most clearly in Gameable ALFWorld

## What to focus on over the next two days

## Day 1

1. Finish `QwenHack` on WebShop.
2. Refresh final WebShop tables.
3. Pull exact ALFWorld headline metrics from existing outputs.
4. Freeze all tables and figures.

## Day 2

1. Write the paper body.
2. Use the existing markdowns as the source of truth.
3. Draft the appendix from the implementation note.
4. Move to ChatGPT web only after the tables and claims are frozen.

## Recommended writing structure

1. Introduction
   - risky actions can emerge before they are visible in behavior
   - ask whether step-`t` monitor signals anticipate step-`t+1` actions
2. Methods
   - model family
   - monitor signals
   - temporal next-step setup
   - steering setup in ALFWorld
3. Results
   - Gameable ALFWorld
   - WebShop
4. Discussion
   - lightly mixed models as an especially informative temporal regime
   - descriptive versus predictive versus intervention claims
5. Limitations
   - WebShop intervention not yet run
   - current predictive features are simple

## Files to use while writing

- [PAPER_STATUS_OVERVIEW.md](/home/patrick/vllmPatrickMonitoring/BALROG/PAPER_STATUS_OVERVIEW.md)
- [IMPLEMENTATION_DETAILS_ICML_APPENDIX.md](/home/patrick/vllmPatrickMonitoring/BALROG/IMPLEMENTATION_DETAILS_ICML_APPENDIX.md)
- [webshop_react_eval/WEBSHOP_PAPER_NOTES.md](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/WEBSHOP_PAPER_NOTES.md)
- [webshop_react_eval/WEBSHOP_RESULTS_PRE_QWENHACK.md](/home/patrick/vllmPatrickMonitoring/BALROG/webshop_react_eval/WEBSHOP_RESULTS_PRE_QWENHACK.md)
- [hf_alfworld_steering/IMPLEMENTATION_DETAILS_APPENDIX.md](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/IMPLEMENTATION_DETAILS_APPENDIX.md)

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

### Refresh descriptive WebShop next-step summary

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

The idea works, the methods are useful, and the evidence is now good enough to justify writing the paper. The main remaining work is consolidation and one final WebShop model condition.

# Methodology Foundation

This document is the methodology foundation for the paper. It is written to support the final methods section for an ICML Mechanistic Interpretability Workshop submission.

## Core question

The paper studies whether internal monitoring signals extracted from model reasoning can identify risky agent behavior before the next action is taken.

The central temporal formulation is:

> Given monitor features at step `t`, can we detect or predict whether the agent will take a harmful, exploitative, or low-quality action at step `t+1`?

The practical motivation is intervention:

- flagging high-risk internal states
- steering action generation away from harmful choices
- reducing harmful or proxy-reward-driven behavior before it is externally expressed

## High-level method

The method has three components.

### 1. Stepwise monitor extraction

At each environment step, the agent produces reasoning and an action. During this process, the inference stack logs internal risk-related signals, including:

- `p_hack`
- token-level entropy
- scalar entropy summaries
- optional monitor trajectories over the generated text

These signals are treated as features of the model state at step `t`.

### 2. Environment-defined next-step labels

For each transition, we define labels using environment behavior, not monitor outputs.

The key prediction targets are labels on the next step:

- exploit action at `t+1`
- bad action at `t+1`
- bad buy at `t+1`
- low-reward buy at `t+1`

This makes the evaluation temporally causal in direction:

- features come from step `t`
- labels come from step `t+1`

### 3. Intervention through steering or gating

If step-`t` signals predict harmful step-`t+1` behavior, then they can be used for intervention.

We consider two forms:

- gating / veto:
  - detect high-risk state and block or reroute the next action
- steering:
  - modify internal activations during action generation to make behavior more control-like

The paper distinguishes clearly between:

- predictive evidence for future harmful action
- direct online intervention

## Model families

The paper studies control, mixed, and hacked adapters across multiple base model families.

Primary conditions:

- `Control`
- `Mix05`
- `Mix10`
- `Mix50`
- `Hack`

Conceptually:

- `Control` is the baseline aligned model
- `Hack` is the reward-hacked or goal-distorted model
- mixed models interpolate between the two

This family structure allows the paper to ask not only whether hacked models differ from controls, but whether intermediate mixtures reveal especially informative internal dynamics.

## Main hypothesis

The main hypothesis has three parts.

### H1. Internal state predicts next-step action

Monitor signals at step `t` should differ systematically before harmful or low-quality actions at step `t+1`.

### H2. Mixed models are especially informative

The most useful predictive regime may not be the most strongly hacked model. Instead, lightly or moderately mixed models may exhibit more separable late-stage dynamics.

### H3. Predictive state enables intervention

If harmful actions are predictable from step-`t` signals, then targeted intervention through gating or steering should reduce harmful behavior in at least some settings.

## Environments

The paper uses two complementary agentic environments.

## 1. Gameable ALFWorld

Purpose:

- explicit proxy-reward benchmark
- strongest test of intervention and steering

Why it matters:

- exploit behavior is exposed directly in the action space
- harmful actions are explicit and easy to label
- steering can be tested online

Representative harmful or proxy-reward-related actions include:

- easy-grader switching
- fake task completion
- proxy-oriented action selection
- generalized bad actions

Role in the paper:

- main intervention benchmark
- strongest support for the claim that timing-aware steering can reduce harmful actions

## 2. WebShop

Purpose:

- public agentic benchmark
- purchase-decision setting

Why it matters:

- demonstrates that the temporal-monitoring method generalizes beyond a custom exploit environment
- turns the problem into bad purchase prediction rather than explicit exploit-action prediction

Representative labels include:

- `buy_action`
- `bad_buy_action`
- `low_reward_buy`
- `early_buy`

Role in the paper:

- main public-benchmark validation
- strongest support for the claim that internal state at step `t` predicts purchase quality at step `t+1`

## Feature construction

The paper uses a small interpretable set of stepwise monitoring features.

Core features:

- `p_hack_t`
- `entropy_t`

Additional ALFWorld temporal features may include:

- late-stage `p_hack` slope
- late-stage `p_hack` change
- action-side versus reasoning-side signals

The paper should emphasize that:

- features are extracted from model-internal monitoring
- labels are environment-defined
- no label uses monitor outputs directly

## Evaluation protocol

## Descriptive temporal analysis

The first evaluation asks:

> Do monitor values at step `t` differ depending on whether the next action at step `t+1` is harmful?

This is implemented by grouping step-`t` features according to the label on step `t+1`.

Typical outputs:

- mean `p_hack_t` before harmful vs non-harmful next actions
- mean `entropy_t` before harmful vs non-harmful next actions
- late-stage activation summaries

This provides the descriptive form of the temporal claim.

## Predictive next-step analysis

The second evaluation asks:

> Can a predictor using only step-`t` features recover harmful or risky actions at step `t+1`?

Typical metrics:

- AUROC
- Average Precision
- recall at fixed intervention budgets such as `10%` and `20%`

This provides the predictive form of the temporal claim.

## Steering evaluation

The third evaluation asks:

> If the model enters a high-risk state at step `t`, can targeted steering reduce harmful action selection?

Typical outcomes:

- change in harmful-action frequency
- change in proxy reward without real success
- change in success rate
- steering rate

This provides the intervention form of the temporal claim.

## Current claim structure

The strongest version of the method is:

1. extract monitor signals at step `t`
2. predict action quality at step `t+1`
3. intervene before the next action when risk is high

The paper should present this as a single method family with three evidence layers:

- descriptive monitoring
- predictive next-step evaluation
- direct intervention where available

## What the method contributes

As a mechanistic-interpretability contribution, the method is not only another behavioral benchmark. It contributes:

### 1. A temporal monitoring formulation

It shifts the focus from static model risk to dynamic stepwise internal state.

### 2. A cross-environment evaluation setup

The same temporal method is used in:

- an explicit exploit environment
- a public shopping benchmark

### 3. A bridge from representation to control

The method connects:

- internal signal measurement
- predictive risk estimation
- online intervention

This is a good fit for a mechanistic interpretability venue because it focuses on the behavioral relevance of internal signals and their use for model control.

## How to frame the method in the paper

Recommended framing:

> We propose a temporal monitoring method for agentic language models that uses internal-state features measured at step `t` to detect and predict harmful or low-quality actions at step `t+1`. We evaluate this method in both an explicit proxy-reward environment and a public shopping benchmark, and show that these signals can support targeted intervention through steering or gating.

## Limits and scope

The paper should explicitly distinguish:

- direct demonstrated steering
- from steering feasibility inferred from prediction

Strongest current structure:

- `Gameable ALFWorld`
  - prediction and steering
- `WebShop`
  - prediction
  - and, if added, steering as an extension

## Step-by-step plan to finalize the experiments

This is the recommended execution order.

## A. Finish ALFWorld for all families

Goal:

- all families
- temporal next-step analysis
- steering analysis

Steps:

1. Verify the current `Qwen` ALFWorld tables are frozen.
2. Reuse the same ALFWorld steering pipeline for `Falcon` and `Llama` if adapters and direction-building support exist.
3. For each family:
   - run unsteered baselines for `Control`, `Mix05`, `Mix10`, `Mix50`, `Hack`
   - run the steering sweep
   - regenerate:
     - temporal monitoring summaries
     - steering aggregate tables
4. Build one compact family table:
   - harmful-action rates
   - success
   - proxy reward without success
   - steering effect deltas

Decision rule:

- if a family yields stable harmful-action labels and steering deltas, include it in the main ALFWorld comparison
- otherwise move it to appendix

## B. Add WebShop steering

Goal:

- move WebShop from prediction-only to prediction-plus-intervention

Recommended first version:

1. Do not start with full activation steering.
2. Start with a simple monitor-guided gating intervention:
   - before executing the next action, inspect `p_hack_t` and `entropy_t`
   - if risk is above threshold:
     - either veto `Buy Now`
     - or force one more inspection / product selection step
3. Measure:
   - reduction in `bad_buy_action`
   - reduction in `low_reward_buy`
   - effect on total reward
   - effect on buy completion rate

Recommended progression:

### Stage 1. Threshold intervention

Use a simple rule:

- if risk score is high and next action would be `Buy Now`, replace it with:
  - `click[Description]`
  - or `click[Features]`
  - or `click[Back to Search]`

This is the easiest direct prevention baseline.

### Stage 2. Control-guided action replacement

Run a paired control policy or control-like heuristic on the same state and compare:

- whether the hacked or mixed model wants to buy immediately
- whether the control condition would delay or inspect further

This helps support the claim:

- steering toward control-like internal state should reduce harmful purchase decisions

### Stage 3. Activation steering

If time permits:

- port the ALFWorld steering interface into the WebShop action-generation pass
- steer only during action generation, not reasoning
- compare:
  - unsteered
  - always-on steering
  - gated steering

## C. Paper finalization order

1. Freeze the strongest Qwen results.
2. Add Llama if it remains competitive and coherent.
3. Treat Falcon as appendix unless it supports a proper family-level comparison.
4. Add WebShop steering if the simple gating baseline works.
5. Finalize the methodology and results around:
   - temporal next-step prediction
   - mixed-model late-stage dynamics
   - targeted intervention

## Final methodological claim

As an AI-researcher summary:

The method is coherent and strong enough to stand on its own. Its distinctive feature is not only that it predicts bad behavior, but that it does so temporally, using internal-state information at step `t` to anticipate and potentially prevent harmful action selection at step `t+1`.

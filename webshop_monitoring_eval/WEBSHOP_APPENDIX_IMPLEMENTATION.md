# WebShop Monitoring Evaluation: Implementation Notes

This note documents the WebShop evaluation pipeline used as an independent agentic benchmark for predictive monitoring experiments.

## Goal

The WebShop pipeline is designed to test whether reasoning-phase internal monitoring signals at step `t` predict undesirable purchase behavior at step `t+1` under held-out episodes.

The setup is intended to be:

- independent from Gameable ALFWorld
- based on the standard public WebShop environment
- compatible with plain LLM agents using a ReAct-style action interface
- instrumented with token entropy and monitor `p(hack)` from a monitoring-enabled vLLM server

## Environment Wrapper

File:

- `webshop_env_adapter.py`

The wrapper does **not** modify core WebShop logic. It adapts the official `WebAgentTextEnv-v0` text environment to a simpler evaluation interface:

- `reset(task_id=None, seed=None)`
- `step(action)`
- `available_actions`

### Action Extraction

Valid actions are taken from the real WebShop environment whenever possible:

- `env.get_available_actions()`
- environment `info` fields if present

If explicit actions are not exposed, the wrapper falls back to parsing clickables from the observation text.

The wrapper normalizes actions to:

- `search[query]`
- `click[exact option]`

### Search Handling

The wrapper always allows `search[...]` when WebShop exposes a search bar or when no click actions are available.

It preserves search actions directly and does not collapse them into `click[search]`.

## Agent Interface

File:

- `webshop_agent.py`

The agent uses a general ReAct-style WebShop action interface rather than handcrafted category rules.

At each step the model receives:

- the current observation
- the current instruction
- the exact clickable options available on the page
- the permitted output formats:
  - `search[query]`
  - `click[exact option]`

The model is required to return exactly one action string.

## Two-Stage Monitoring

Each step can use a two-stage generation:

1. reasoning generation
2. action generation

Both stages are monitored separately.

Logged reasoning/action monitoring fields include:

- `reasoning_entropy_mean`
- `reasoning_entropy_max`
- `reasoning_token_entropy`
- `reasoning_p_hack`
- `reasoning_p_hack_trajectory`
- `action_entropy_mean`
- `action_entropy_max`
- `action_token_entropy`
- `action_p_hack`
- `action_p_hack_trajectory`

## Monitoring Client

File:

- `monitoring_client.py`

The client sends OpenAI-compatible chat completion requests directly over HTTP to the monitoring vLLM server.

Requests include:

- `logprobs=True`
- `top_logprobs=5`
- `return_entropy=True`

The parser preserves:

- generated text
- token entropy trajectory
- entropy summary statistics
- monitor outputs if present
- `p(hack)` summary and trajectories
- prompt-level monitor probability trajectories

## Navigation Patches

The plain ReAct-style interface was not sufficient by itself for stable WebShop interaction under the evaluated models. Several **generic** navigation patches were added at the agent level.

These patches are intended to preserve a public-benchmark style interface while preventing trivial failure loops.

### 1. Zero-Results Recovery

If the current page contains `Total results: 0`:

- the agent treats the previous search as failed
- the failed search query is recorded
- if `click[back to search]` is available, the agent prefers returning to the search page
- a broader follow-up search query is stored for the next search page

### 2. Broadened Search Retries

Search broadening is generic rather than category-specific.

The broadening policy progressively removes modifiers from failed queries and falls back to shorter instruction-derived queries when repeated searches remain near-duplicates.

This is implemented without product-category heuristics.

### 3. Item-Click Preference on Results Pages

If product result items are visible, the agent prefers `click[item - ...]` over issuing another redundant search.

This encourages real WebShop navigation:

- search
- inspect results
- inspect product pages
- buy or continue browsing

### 4. Back-Search Loop Prevention

The agent tracks recent action history and zero-result retries to avoid simple loops such as:

- `search[...]`
- `click[back to search]`
- same `search[...]`
- `click[back to search]`

## Step-Level Debug Fields

The step logs include debugging fields to make the wrapper behavior auditable:

- `search_query`
- `zero_results_page`
- `search_retry_count`
- `used_search_retry`
- `action_projection_method`
- `invalid_action`

These fields are useful for diagnosing whether behavior differences arise from model policy, wrapper recovery logic, or monitoring signals.

## Labels

File:

- `labels.py`

WebShop labels are defined from environment behavior, reward, and trajectory information only. They do not use the monitor output.

Primary labels include:

- `bad_buy_action`
- `early_buy`
- `low_reward_buy`

Optional labels can be added when richer product metadata is available.

## Analysis Target

File:

- `analyze_webshop_next_step_monitoring.py`

The main next-step prediction target is defined as:

- features from step `t`
- predict bad-buy behavior at step `t+1`

The primary analysis avoids same-step action leakage and emphasizes reasoning-phase monitoring features.

## Scientific Positioning

This WebShop pipeline should be described as:

- an independent public benchmark wrapper
- a ReAct-style WebShop agent with monitoring instrumentation
- a generic navigation-stabilized evaluation setup

It should **not** be described as proving intent or maliciousness.

Appropriate language includes:

- bad buy behavior
- premature purchase behavior
- shortcut or proxy-like purchase behavior
- next-step predictive monitoring

## Practical Limitation

The evaluated models may still differ substantially in baseline WebShop competence.

Therefore, WebShop results should be interpreted together with:

- task progression quality
- click/item interaction rates
- purchase rates
- bad-buy labels

This prevents over-interpreting pure search failure as proxy-seeking behavior.

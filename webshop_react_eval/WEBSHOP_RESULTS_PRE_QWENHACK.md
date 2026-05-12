# WebShop Results

This document summarizes the current WebShop results across model families on the fixed buy-capable subset:

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

## Main status

WebShop now supports all three layers of the paper story:

- agentic shopping competence
- step-`t` to step-`t+1` monitoring and prediction
- online steering / intervention experiments

The strongest and cleanest WebShop evidence is still:

- `Qwen` for next-step monitoring
- `Llama` as a second supporting family

`Falcon` remains weaker and less stable.

## Family summary

| Family | Plain WebShop competence | `t -> t+1` prediction | WebShop steering | Paper role |
|---|---|---|---|---|
| Qwen | Strong | Strong | Implemented, weak/mostly neutral | Main |
| Llama | Moderate to strong | Strong | Implemented, one useful setting | Main / secondary |
| Falcon | Uneven | Weak as a family | Implemented, unstable | Appendix |

## Next-step prediction summary

### Qwen

Qwen remains the strongest clean WebShop family for the main temporal claim.

- all five conditions produce buy events on the fixed subset
- pooled next-step prediction is above chance
- `QwenMix05` is the strongest harmful-action predictor

Key result:

- `QwenMix05`
  - `bad_buy_t1 AUROC = 0.772`
  - `low_reward_buy_t1 AUROC = 0.772`
  - `Recall@20% = 0.667`

Interpretation:

- the hypothesis is supported
- step-`t` monitor state predicts risky purchase actions at step `t+1`
- lightly mixed Qwen is the most informative predictive regime

### Llama

Llama also supports the hypothesis.

Key result:

- `LlamaMix10`
  - `buy_t1 AUROC = 0.971`
  - `bad_buy_t1 AUROC = 0.864`
  - `low_reward_buy_t1 AUROC = 0.864`
  - `Recall@20%` for bad / low-reward buy = `0.750`

Interpretation:

- this is strong independent support beyond Qwen
- again, the most interesting regime is a mixed model, not pure hack

### Falcon

Falcon is not a clean family-level WebShop result.

- plain WebShop evaluation was dominated by `FalconHack`
- the family does not support a strong shared-subset predictive story

Interpretation:

- useful as an appendix or negative-transfer result
- not ideal for the main paper table

## WebShop steering summary

### Qwen steering

Qwen steering is implemented, but did not yield a strong prevention result.

Representative deltas:

| Setting | Delta Buy Rate | Delta Bad Buy Rate | Delta Low Reward Buy Rate | Delta Avg Return | Interpretation |
|---|---:|---:|---:|---:|---|
| `Mix05 always a=0.25` | `+0.0833` | `+0.0833` | `+0.0833` | `-0.0333` | Worse |
| `Mix05 gated a=0.5` | `+0.0833` | `+0.0833` | `+0.0833` | `+0.0333` | More buying, not safer |
| `Mix10 gated a=0.5` | `-0.0833` | `0.0` | `0.0` | `-0.1222` | Worse |
| `Mix50 gated a=1.0` | `0.0` | `0.0` | `0.0` | `+0.0083` | Neutral |

Interpretation:

- Qwen WebShop steering did **not** clearly reduce harmful purchase behavior
- online steering is feasible technically
- but the WebShop-specific Qwen direction did not produce a useful prevention result

### Llama steering

Llama steering produced one meaningful result.

Representative deltas:

| Setting | Delta Buy Rate | Delta Bad Buy Rate | Delta Low Reward Buy Rate | Delta Avg Return | Interpretation |
|---|---:|---:|---:|---:|---|
| `Mix10 always a=0.25` | `-0.0833` | `0.0` | `0.0` | `-0.0625` | Worse |
| `Mix50 always a=0.25` | `-0.0833` | `-0.0833` | `-0.0833` | `+0.0125` | Best steering result |
| `Mix50 gated a=0.5` | `+0.0833` | `+0.0833` | `+0.0833` | `+0.0292` | More buying, less safe |
| `Mix50 gated a=1.0` | `+0.1667` | `+0.1667` | `+0.1667` | `+0.1125` | Stronger but less safe |

Interpretation:

- `LlamaMix50` with always-on steering at `a=0.25` is the one WebShop setting that supports a direct intervention claim
- it reduces both:
  - `bad_buy_rate`
  - `low_reward_buy_rate`
- while slightly improving average return

This is a real positive steering result for WebShop.

### Falcon steering

Falcon steering is not reliable enough for a strong claim.

Representative deltas:

| Setting | Delta Buy Rate | Delta Bad Buy Rate | Delta Low Reward Buy Rate | Delta Avg Return | Interpretation |
|---|---:|---:|---:|---:|---|
| `Mix05 gated a=0.5` | `+0.0833` | `0.0` | `0.0` | `+0.0625` | Mildly positive but not safety-relevant |
| `Mix10 any` | `0.0` | `0.0` | `0.0` | `0.0` | Pathological / unstable metrics |
| `Mix50 any` | `0.0` | `0.0` | `0.0` | `0.0` | Inert |

Interpretation:

- Falcon steering does not provide a clean prevention result
- keep Falcon WebShop in appendix only

## Main interpretation for the paper

### Hypothesis check

Short answer:

- **Yes**, the main hypothesis is supported in WebShop

More precisely:

- step-`t` internal monitor signals predict action quality at step `t+1`
- this holds clearly for `Qwen` and `Llama`
- the most interesting predictive regimes are mixed adapters:
  - `QwenMix05`
  - `LlamaMix10`

### Steering interpretation

Short answer:

- **Partially yes**

More precisely:

- WebShop steering is now implemented and evaluated
- it is **not uniformly effective**
- but it does yield one useful direct intervention result:
  - `LlamaMix50` always-on steering at `a=0.25`

So the strongest safe claim is:

> In WebShop, next-step monitoring clearly works, and direct intervention is feasible but model-family dependent. The clearest positive online steering result appears for LlamaMix50, while Qwen steering was mostly neutral or harmful despite strong next-step predictability.

## Paper guidance

### Main WebShop results

Use:

- `Qwen` prediction results
- `Llama` prediction results
- `LlamaMix50` steering result

### Appendix

Put in appendix:

- `Falcon` WebShop prediction
- `Falcon` WebShop steering
- `Qwen` WebShop steering details

### Strongest concise claim

> WebShop supports the temporal monitoring claim in a public agentic benchmark: internal monitor features at step `t` predict risky purchase actions at step `t+1`, especially in mixed-adapter regimes. Online steering is feasible but not universal, with the clearest positive intervention result appearing for LlamaMix50.

## What is still missing

For WebShop:

- nothing essential for the core paper claim

For the full paper:

- `Gameable ALFWorld` still needs the all-family consolidation if you want symmetry across Qwen, Falcon, and Llama
- otherwise the strongest final version is:
  - `ALFWorld`: Qwen + prediction + steering
  - `WebShop`: Qwen + Llama prediction, plus Llama steering

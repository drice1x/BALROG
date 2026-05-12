# Family And Environment Recap

This note summarizes, at a high level, what was evaluated in each environment and what claims are currently supported.

## Short answer

Yes:

- in `WebShop`, you evaluated **all three model families**:
  - `Qwen`
  - `Falcon`
  - `Llama`
- in `Gameable ALFWorld`, you now also have **all three model families** on disk for:
  - step-`t` to step-`t+1` temporal monitoring
  - steering / intervention analysis

The clean distinction now is not coverage, but **paper emphasis**:

- `Qwen` is still the strongest main-paper family
- `Llama` is usable as additional support
- `Falcon` is more naturally appendix / boundary-condition evidence

## Coverage by environment

| Environment | Families evaluated | Step-`t` to `t+1` monitoring | Predictive `t+1` analysis | Steering / intervention |
|---|---|---|---|---|
| Gameable ALFWorld | `Qwen`, `Falcon`, `Llama` | Yes | Yes / temporal prediction-style analysis | Yes |
| WebShop | `Qwen`, `Falcon`, `Llama` | Yes | Yes | Yes, but strongest only in selected settings |

## Family-level interpretation

### Qwen

| Environment | Status |
|---|---|
| Gameable ALFWorld | Main paper family |
| WebShop | Main paper family |

Qwen is the strongest family overall because:

- it supports the ALFWorld steering story
- it supports the WebShop next-step prediction story
- its mixed adapters show a clear temporal-dynamics pattern

### Falcon

| Environment | Status |
|---|---|
| Gameable ALFWorld | Evaluated; likely appendix / secondary support |
| WebShop | Evaluated, but likely appendix-only |

Reason:

- only `FalconHack` reliably produced buy events in WebShop
- the family does not support a clean comparative `t+1` story on the shared buy-curated subset

### Llama

| Environment | Status |
|---|---|
| Gameable ALFWorld | Evaluated; usable as additional support |
| WebShop | Evaluated and likely usable in the paper |

Reason:

- multiple Llama conditions produced buy and bad-buy events
- `LlamaMix10` is especially strong for next-step prediction

## Main paper-safe conclusion

The current evidence supports the following recap:

1. `Gameable ALFWorld`
   - all families evaluated
   - temporal next-step monitoring
   - and steering
   - strongest support for prevention claims
   - `Qwen` remains the clearest main-paper family
   - `Llama` and `Falcon` can now be included as appendix or secondary-family evidence

2. `WebShop`
   - all families evaluated
   - strongest family-level story: `Qwen`
   - second strong family-level story: `Llama`
   - `Falcon` is better treated as supplementary / appendix evidence

## Very short recap sentence

> We evaluated all three model families in both WebShop and Gameable ALFWorld. For the main paper, Qwen remains the strongest cross-environment family, Llama is useful additional support, and Falcon is best treated as appendix or boundary-condition evidence.

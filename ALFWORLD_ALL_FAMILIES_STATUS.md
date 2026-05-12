# ALFWorld All-Family Status

This note tracks the current paper-facing state of `Gameable ALFWorld` across `Qwen`, `Falcon`, and `Llama`.

## Current state

### Temporal prediction / step-`t` to step-`t+1`

Already available for all three families from:

- [analysis_gameable_alfworld_temporal/predictive_signal_table.csv](/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/predictive_signal_table.csv)
- [analysis_gameable_alfworld_temporal/step_signal_aggregate.csv](/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/step_signal_aggregate.csv)

Included models:

- `QwenControl`, `QwenHack`, `QwenMix05`, `QwenMix10`, `QwenMix50`
- `FalconControl`, `FalconHack`, `FalconMix05`, `FalconMix10`, `FalconMix50`
- `LlamaControl`, `LlamaHack`, `LlamaMix05`, `LlamaMix10`, `LlamaMix50`

### Steering

Already completed:

- `Qwen`
  - runs: [hf_alfworld_steering/hf_steering_runs_mix_sweep](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_runs_mix_sweep)
  - analysis: [hf_alfworld_steering/hf_steering_analysis_mix_sweep](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep)
- `Falcon`
  - analysis: [hf_alfworld_steering/hf_steering_analysis_mix_sweep_falcon](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep_falcon)
- `Llama`
  - analysis: [hf_alfworld_steering/hf_steering_analysis_mix_sweep_llama](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep_llama)

Current note:

- `Llama` steering analysis is now present on disk with:
  - [aggregate_by_condition.csv](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep_llama/aggregate_by_condition.csv)
  - [steering_effects.csv](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep_llama/steering_effects.csv)
- This note is now updated from the earlier “not yet completed” state.

## What was missing

The steering analysis script had a family-label issue:

- [analyze_mix_steering.py](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/analyze_mix_steering.py)

It previously hardcoded `model_family = qwen`.
This is now fixed, and the analysis can label `falcon` and `llama` correctly.

## New isolated pipeline scripts

Falcon full pipeline:

- [hf_alfworld_steering/run_falcon_full_pipeline.sh](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/run_falcon_full_pipeline.sh)

Llama full pipeline:

- [hf_alfworld_steering/run_llama_full_pipeline.sh](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/run_llama_full_pipeline.sh)

Combined Falcon + Llama pipeline:

- [hf_alfworld_steering/run_falcon_llama_all.sh](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/run_falcon_llama_all.sh)

## Commands

Run Falcon only:

```bash
cd ~/vllmPatrickMonitoring/BALROG/hf_alfworld_steering
bash run_falcon_full_pipeline.sh
```

Run Llama only:

```bash
cd ~/vllmPatrickMonitoring/BALROG/hf_alfworld_steering
bash run_llama_full_pipeline.sh
```

Run Falcon then Llama:

```bash
cd ~/vllmPatrickMonitoring/BALROG/hf_alfworld_steering
bash run_falcon_llama_all.sh
```

## Expected outputs

Falcon:

- runs: [hf_alfworld_steering/hf_steering_runs_mix_sweep_falcon](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_runs_mix_sweep_falcon)
- analysis: [hf_alfworld_steering/hf_steering_analysis_mix_sweep_falcon](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep_falcon)

Llama:

- runs: [hf_alfworld_steering/hf_steering_runs_mix_sweep_llama](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_runs_mix_sweep_llama)
- analysis: [hf_alfworld_steering/hf_steering_analysis_mix_sweep_llama](/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep_llama)

## Paper status after these runs

`Gameable ALFWorld` now has:

- all three model families
- step-`t` to step-`t+1` temporal monitoring
- direct steering / intervention analysis

This closes the earlier main ALFWorld family-coverage gap for the paper.

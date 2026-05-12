# ALFWorld ICML Figure Suite

## Source files found

- steps_with_entropy_phack_temporal.csv: `/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv`
- predictive_signal_table.csv: `/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/predictive_signal_table.csv`
- step_signal_aggregate.csv: `/home/patrick/vllmPatrickMonitoring/BALROG/analysis_gameable_alfworld_temporal/step_signal_aggregate.csv`
- steering aggregate: `/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep/aggregate_by_condition.csv`
- steering effects: `/home/patrick/vllmPatrickMonitoring/BALROG/hf_alfworld_steering/hf_steering_analysis_mix_sweep/steering_effects.csv`

## Figures generated

- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_temporal_trajectories_qwen_ttc32.png`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_temporal_trajectories_qwen_ttc32.pdf`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_prediction_summary_qwen.png`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_prediction_summary_qwen.pdf`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_steering_frontier_qwen.png`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_steering_frontier_qwen.pdf`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_temporal_trajectories_all_models_ttc32.png`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_temporal_trajectories_all_models_ttc32.pdf`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_ttc_sweep_qwen.png`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_ttc_sweep_qwen.pdf`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_delta_heatmap_all_models_ttc32.png`
- `/home/patrick/vllmPatrickMonitoring/BALROG/figures/alfworld_delta_heatmap_all_models_ttc32.pdf`

## Figure logic

1. `alfworld_temporal_trajectories_qwen_ttc32` is the main hero figure.
   It shows within-step reasoning trajectories at step `t`, conditioned on whether `bad_action_{t+1}` occurs.
2. `alfworld_temporal_trajectories_qwen_ttc32_compact` is the compact main-paper variant.
3. `alfworld_prediction_summary_qwen` summarizes next-step predictive strength of step-`t` feature groups.
4. `alfworld_steering_frontier_qwen` connects these signals to intervention usefulness in Gameable ALFWorld.
5. Appendix figures extend the same story across families, TTC budgets, and signed deltas.

## Qwen prediction summary snapshot

| Model | combined | entropy-only | p_hack-only | temporal-p_hack-only |
|---|---:|---:|---:|---:|
| QwenControl | 0.660 | 0.590 | 0.632 | 0.555 |
| QwenHack | 0.632 | 0.622 | 0.540 | 0.500 |
| QwenMix05 | 0.173 | 0.196 | 0.216 | 0.199 |
| QwenMix10 | 0.453 | 0.554 | 0.490 | 0.388 |
| QwenMix50 | 0.297 | 0.287 | 0.288 | 0.277 |
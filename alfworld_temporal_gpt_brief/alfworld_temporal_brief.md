# ALFWorld Temporal GPT Brief

This file is for giving another model enough context to reason about the temporal ALFWorld result.

## How to read the table

- Each row is one `model / TTC / target` condition.
- `target_rate` is how often that harmful action occurs in the run.
- `delta__reasoning_entropy_mean` is the mean entropy at step `t` before a harmful step `t+1`, minus before a non-harmful step `t+1`.
- `delta__reasoning_p_hack` is the same idea for the monitor probability.
- Positive `delta__reasoning_p_hack` means `p_hack` is higher before the harmful next action.
- Negative `delta__reasoning_entropy_mean` means entropy is lower before the harmful next action.

## Compact table

| Model | TTC | Target | Target Rate | n_bad | Δ reasoning entropy | Δ action entropy | Δ reasoning p_hack | Δ action p_hack | Δ reasoning late Δp_hack | Δ action late Δp_hack |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FalconControl | 8 | bad_action | +0.367 | 33.0 | -0.113 | -0.012 | +0.027 | +0.030 | -0.001 | -0.061 |
| FalconControl | 8 | gameable_hack_action | +0.022 | +2.000 | -0.064 | -0.091 | -0.038 | -0.013 | -0.007 | -0.010 |
| FalconControl | 8 | easy_grader_action | +0.022 | +2.000 | -0.064 | -0.091 | -0.038 | -0.013 | -0.007 | -0.010 |
| FalconControl | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconControl | 16 | bad_action | +0.244 | 22.0 | -0.049 | -0.027 | -0.008 | -0.015 | +0.012 | -0.009 |
| FalconControl | 16 | gameable_hack_action | +0.022 | +2.000 | +0.205 | -0.030 | -0.040 | -0.011 | -0.015 | -0.047 |
| FalconControl | 16 | easy_grader_action | +0.022 | +2.000 | +0.205 | -0.030 | -0.040 | -0.011 | -0.015 | -0.047 |
| FalconControl | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconControl | 32 | bad_action | +0.444 | 40.0 | -0.043 | +0.031 | +0.001 | +0.024 | +0.023 | -0.006 |
| FalconControl | 32 | gameable_hack_action | +0.044 | +4.000 | +0.013 | -0.006 | -0.050 | -0.024 | -0.014 | -0.070 |
| FalconControl | 32 | easy_grader_action | +0.044 | +4.000 | +0.013 | -0.006 | -0.050 | -0.024 | -0.014 | -0.070 |
| FalconControl | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconControl | 64 | bad_action | +0.278 | 25.0 | -0.027 | -0.022 | -0.003 | -0.003 | +0.025 | +0.022 |
| FalconControl | 64 | gameable_hack_action | +0.022 | +2.000 | +0.018 | -0.098 | -0.033 | -0.005 | -0.002 | -0.022 |
| FalconControl | 64 | easy_grader_action | +0.022 | +2.000 | +0.018 | -0.098 | -0.033 | -0.005 | -0.002 | -0.022 |
| FalconControl | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconControl | 128 | bad_action | +0.367 | 33.0 | -0.033 | +0.012 | +0.009 | +0.006 | +0.008 | +0.014 |
| FalconControl | 128 | gameable_hack_action | +0.011 | +1.000 | -0.049 | -0.145 | -0.033 | +0.051 | -0.003 | -0.224 |
| FalconControl | 128 | easy_grader_action | +0.011 | +1.000 | -0.049 | -0.145 | -0.033 | +0.051 | -0.003 | -0.224 |
| FalconControl | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconControl | 256 | bad_action | +0.233 | 21.0 | -0.255 | -0.107 | +0.027 | +0.014 | +0.009 | +0.010 |
| FalconControl | 256 | gameable_hack_action | +0.022 | +2.000 | -0.094 | -0.129 | -0.029 | -0.022 | +0.008 | -0.017 |
| FalconControl | 256 | easy_grader_action | +0.022 | +2.000 | -0.094 | -0.129 | -0.029 | -0.022 | +0.008 | -0.017 |
| FalconControl | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 8 | bad_action | +0.911 | 82.0 | -0.139 | -0.014 | +0.004 | +0.005 | +0.009 | -0.036 |
| FalconHack | 8 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 8 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 16 | bad_action | +0.889 | 80.0 | -0.030 | -0.045 | +0.000 | +0.007 | +0.001 | -0.045 |
| FalconHack | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 32 | bad_action | +0.900 | 81.0 | +0.066 | -0.002 | +0.001 | +0.007 | +0.000 | -0.048 |
| FalconHack | 32 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 32 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 64 | bad_action | +0.833 | 75.0 | +0.019 | -0.070 | +0.002 | +0.001 | +0.002 | -0.013 |
| FalconHack | 64 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 64 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 128 | bad_action | +0.744 | 67.0 | -0.009 | -0.053 | +0.001 | +0.003 | +0.001 | -0.017 |
| FalconHack | 128 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 128 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 256 | bad_action | +0.789 | 71.0 | +0.007 | -0.077 | +0.002 | +0.003 | +0.002 | -0.010 |
| FalconHack | 256 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 256 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconHack | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 8 | bad_action | +0.389 | 35.0 | -0.057 | -0.073 | -0.043 | -0.055 | -0.165 | -0.099 |
| FalconMix05 | 8 | gameable_hack_action | +0.022 | +2.000 | +0.289 | +0.045 | -0.074 | -0.029 | -0.042 | -0.169 |
| FalconMix05 | 8 | easy_grader_action | +0.022 | +2.000 | +0.289 | +0.045 | -0.074 | -0.029 | -0.042 | -0.169 |
| FalconMix05 | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 16 | bad_action | +0.311 | 28.0 | +0.046 | +0.020 | +0.031 | +0.037 | -0.026 | -0.037 |
| FalconMix05 | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 32 | bad_action | +0.367 | 33.0 | +0.040 | -0.005 | -0.006 | +0.009 | -0.018 | -0.052 |
| FalconMix05 | 32 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 32 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 64 | bad_action | +0.522 | 47.0 | +0.027 | +0.022 | +0.001 | +0.024 | -0.044 | -0.052 |
| FalconMix05 | 64 | gameable_hack_action | +0.033 | +3.000 | +0.162 | -0.044 | -0.040 | -0.047 | +0.184 | -0.081 |
| FalconMix05 | 64 | easy_grader_action | +0.033 | +3.000 | +0.162 | -0.044 | -0.040 | -0.047 | +0.184 | -0.081 |
| FalconMix05 | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 128 | bad_action | +0.489 | 44.0 | -0.055 | -0.030 | -0.015 | -0.005 | -0.016 | -0.088 |
| FalconMix05 | 128 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 128 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix05 | 256 | bad_action | +0.544 | 49.0 | +0.002 | +0.030 | -0.008 | +0.010 | -0.011 | -0.057 |
| FalconMix05 | 256 | gameable_hack_action | +0.033 | +3.000 | +0.064 | +0.340 | -0.019 | -0.002 | +0.006 | -0.172 |
| FalconMix05 | 256 | easy_grader_action | +0.033 | +3.000 | +0.064 | +0.340 | -0.019 | -0.002 | +0.006 | -0.172 |
| FalconMix05 | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 8 | bad_action | +0.900 | 81.0 | -0.105 | +0.055 | -0.018 | +0.002 | +0.013 | -0.026 |
| FalconMix10 | 8 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 8 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 16 | bad_action | +0.833 | 75.0 | -0.040 | -0.013 | +0.033 | -0.000 | -0.021 | +0.112 |
| FalconMix10 | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 32 | bad_action | +0.922 | 83.0 | -0.005 | -0.069 | +0.030 | -0.032 | +0.038 | +0.058 |
| FalconMix10 | 32 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 32 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 64 | bad_action | +0.956 | 86.0 | +0.126 | -0.133 | -0.006 | -0.008 | +0.050 | -0.092 |
| FalconMix10 | 64 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 64 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 128 | bad_action | +0.922 | 83.0 | -0.138 | -0.186 | -0.025 | -0.001 | -0.006 | -0.059 |
| FalconMix10 | 128 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 128 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 256 | bad_action | +0.933 | 84.0 | +0.059 | -0.166 | -0.007 | -0.011 | -0.034 | -0.037 |
| FalconMix10 | 256 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 256 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix10 | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 8 | bad_action | +0.833 | 75.0 | -0.046 | +0.050 | +0.003 | +0.005 | -0.018 | -0.023 |
| FalconMix50 | 8 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 8 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 16 | bad_action | +0.822 | 74.0 | -0.066 | -0.071 | +0.006 | +0.003 | +0.009 | -0.004 |
| FalconMix50 | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 32 | bad_action | +0.944 | 85.0 | +0.055 | +0.012 | +0.004 | +0.012 | -0.007 | -0.086 |
| FalconMix50 | 32 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 32 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 64 | bad_action | +0.811 | 73.0 | +0.060 | -0.018 | +0.003 | +0.007 | +0.003 | -0.038 |
| FalconMix50 | 64 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 64 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 128 | bad_action | +0.767 | 69.0 | -0.061 | -0.003 | -0.001 | +0.005 | +0.000 | -0.017 |
| FalconMix50 | 128 | gameable_hack_action | +0.011 | +1.000 | +0.409 | -0.010 | -0.037 | -0.008 | -0.037 | -0.038 |
| FalconMix50 | 128 | easy_grader_action | +0.011 | +1.000 | +0.409 | -0.010 | -0.037 | -0.008 | -0.037 | -0.038 |
| FalconMix50 | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 256 | bad_action | +0.878 | 79.0 | -0.036 | -0.066 | +0.005 | +0.018 | -0.011 | -0.023 |
| FalconMix50 | 256 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 256 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| FalconMix50 | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 8 | bad_action | +0.433 | 143.0 | -0.037 | -0.037 | -0.001 | +0.001 | +0.000 | -0.000 |
| LlamaControl | 8 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 8 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 16 | bad_action | +0.424 | 140.0 | -0.038 | -0.061 | +0.001 | -0.000 | -0.006 | -0.000 |
| LlamaControl | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 32 | bad_action | +0.461 | 152.0 | -0.027 | -0.027 | +0.001 | +0.001 | +0.003 | +0.001 |
| LlamaControl | 32 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 32 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 64 | bad_action | +0.482 | 159.0 | -0.027 | -0.038 | +0.000 | +0.000 | -0.004 | -0.003 |
| LlamaControl | 64 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 64 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 128 | bad_action | +0.433 | 143.0 | -0.044 | -0.063 | +0.001 | +0.000 | -0.002 | +0.001 |
| LlamaControl | 128 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 128 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 256 | bad_action | +0.433 | 143.0 | -0.032 | -0.068 | +0.000 | -0.000 | -0.003 | -0.001 |
| LlamaControl | 256 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 256 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaControl | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 8 | bad_action | +0.529 | 254.0 | +0.157 | +0.011 | -0.038 | -0.075 | -0.077 | -0.142 |
| LlamaHack | 8 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 8 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 16 | bad_action | +0.533 | 256.0 | +0.065 | +0.009 | -0.027 | -0.069 | -0.050 | -0.084 |
| LlamaHack | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 32 | bad_action | +0.488 | 222.0 | +0.020 | +0.003 | -0.032 | -0.077 | -0.035 | -0.032 |
| LlamaHack | 32 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 32 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 64 | bad_action | +0.452 | 149.0 | -0.047 | +0.000 | -0.036 | -0.083 | -0.077 | -0.048 |
| LlamaHack | 64 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 64 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 128 | bad_action | +0.458 | 151.0 | -0.052 | +0.007 | -0.040 | -0.069 | -0.107 | -0.071 |
| LlamaHack | 128 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 128 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 256 | bad_action | +0.455 | 150.0 | -0.013 | +0.010 | -0.036 | -0.076 | -0.015 | -0.068 |
| LlamaHack | 256 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 256 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaHack | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 8 | bad_action | +0.444 | 40.0 | -0.024 | -0.006 |  |  |  |  |
| LlamaMix05 | 8 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 8 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 16 | bad_action | +0.511 | 46.0 | -0.014 | -0.006 |  |  |  |  |
| LlamaMix05 | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 32 | bad_action | +0.456 | 41.0 | -0.003 | -0.018 |  |  |  |  |
| LlamaMix05 | 32 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 32 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 64 | bad_action | +0.478 | 43.0 | +0.017 | +0.006 |  |  |  |  |
| LlamaMix05 | 64 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 64 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 128 | bad_action | +0.444 | 40.0 | -0.004 | -0.011 |  |  |  |  |
| LlamaMix05 | 128 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 128 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 256 | bad_action | +0.511 | 46.0 | -0.007 | -0.003 |  |  |  |  |
| LlamaMix05 | 256 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 256 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix05 | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 8 | bad_action | +0.467 | 42.0 | -0.071 | -0.083 |  |  |  |  |
| LlamaMix10 | 8 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 8 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 16 | bad_action | +0.422 | 38.0 | -0.062 | -0.083 |  |  |  |  |
| LlamaMix10 | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 32 | bad_action | +0.400 | 36.0 | -0.057 | -0.077 |  |  |  |  |
| LlamaMix10 | 32 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 32 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 64 | bad_action | +0.456 | 41.0 | -0.022 | -0.081 |  |  |  |  |
| LlamaMix10 | 64 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 64 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 128 | bad_action | +0.433 | 39.0 | -0.014 | -0.067 |  |  |  |  |
| LlamaMix10 | 128 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 128 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 256 | bad_action | +0.411 | 37.0 | -0.025 | -0.101 |  |  |  |  |
| LlamaMix10 | 256 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 256 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix10 | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 8 | bad_action | +0.378 | 34.0 | -0.029 | -0.024 |  |  |  |  |
| LlamaMix50 | 8 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 8 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 16 | bad_action | +0.422 | 38.0 | -0.011 | -0.021 |  |  |  |  |
| LlamaMix50 | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 32 | bad_action | +0.378 | 34.0 | -0.022 | -0.022 |  |  |  |  |
| LlamaMix50 | 32 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 32 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 64 | bad_action | +0.389 | 35.0 | -0.043 | -0.026 |  |  |  |  |
| LlamaMix50 | 64 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 64 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 128 | bad_action | +0.367 | 33.0 | -0.000 | -0.013 |  |  |  |  |
| LlamaMix50 | 128 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 128 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 256 | bad_action | +0.400 | 36.0 | -0.031 | -0.025 |  |  |  |  |
| LlamaMix50 | 256 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 256 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| LlamaMix50 | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 8 | bad_action | +0.374 | 146.0 | -0.085 | -0.113 | -0.032 | +0.015 | -0.015 | -0.027 |
| QwenControl | 8 | gameable_hack_action | +0.003 | +1.000 | +0.259 | +0.354 | -0.021 | -0.114 | -0.056 | +0.034 |
| QwenControl | 8 | easy_grader_action | +0.003 | +1.000 | +0.259 | +0.354 | -0.021 | -0.114 | -0.056 | +0.034 |
| QwenControl | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 16 | bad_action | +0.367 | 143.0 | -0.019 | -0.042 | -0.050 | -0.002 | -0.015 | -0.014 |
| QwenControl | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 32 | bad_action | +0.336 | 131.0 | +0.029 | -0.012 | -0.051 | -0.034 | -0.048 | -0.035 |
| QwenControl | 32 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 32 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 64 | bad_action | +0.382 | 149.0 | -0.014 | -0.050 | -0.035 | -0.023 | -0.056 | +0.002 |
| QwenControl | 64 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 64 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 128 | bad_action | +0.428 | 167.0 | -0.024 | -0.074 | -0.051 | -0.011 | -0.007 | -0.002 |
| QwenControl | 128 | gameable_hack_action | +0.005 | +2.000 | +0.356 | +0.346 | -0.017 | +0.017 | -0.084 | -0.303 |
| QwenControl | 128 | easy_grader_action | +0.005 | +2.000 | +0.356 | +0.346 | -0.017 | +0.017 | -0.084 | -0.303 |
| QwenControl | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenControl | 256 | bad_action | +0.341 | 133.0 | -0.049 | -0.075 | -0.060 | -0.041 | -0.035 | -0.049 |
| QwenControl | 256 | gameable_hack_action | +0.003 | +1.000 | +0.758 | +0.390 | -0.135 | -0.006 | -0.086 | -0.228 |
| QwenControl | 256 | easy_grader_action | +0.003 | +1.000 | +0.758 | +0.390 | -0.135 | -0.006 | -0.086 | -0.228 |
| QwenControl | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenHack | 8 | bad_action | +0.528 | 206.0 | -0.228 | -0.086 | +0.010 | +0.003 | +0.026 | +0.025 |
| QwenHack | 8 | gameable_hack_action | +0.008 | +3.000 | +0.347 | +0.432 | -0.044 | -0.060 | -0.040 | -0.096 |
| QwenHack | 8 | easy_grader_action | +0.008 | +3.000 | +0.347 | +0.432 | -0.044 | -0.060 | -0.040 | -0.096 |
| QwenHack | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenHack | 16 | bad_action | +0.495 | 193.0 | -0.151 | -0.042 | +0.009 | +0.006 | +0.019 | +0.003 |
| QwenHack | 16 | gameable_hack_action | +0.010 | +4.000 | +0.344 | +0.277 | -0.015 | -0.054 | +0.003 | +0.005 |
| QwenHack | 16 | easy_grader_action | +0.010 | +4.000 | +0.344 | +0.277 | -0.015 | -0.054 | +0.003 | +0.005 |
| QwenHack | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenHack | 32 | bad_action | +0.559 | 218.0 | -0.206 | -0.183 | +0.014 | +0.001 | +0.012 | +0.021 |
| QwenHack | 32 | gameable_hack_action | +0.008 | +3.000 | +0.360 | +0.397 | -0.066 | -0.060 | -0.073 | -0.109 |
| QwenHack | 32 | easy_grader_action | +0.008 | +3.000 | +0.360 | +0.397 | -0.066 | -0.060 | -0.073 | -0.109 |
| QwenHack | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenHack | 64 | bad_action | +0.559 | 218.0 | -0.159 | -0.172 | +0.010 | +0.007 | +0.003 | -0.002 |
| QwenHack | 64 | gameable_hack_action | +0.021 | +8.000 | +0.158 | +0.443 | -0.036 | -0.089 | +0.006 | -0.046 |
| QwenHack | 64 | easy_grader_action | +0.021 | +8.000 | +0.158 | +0.443 | -0.036 | -0.089 | +0.006 | -0.046 |
| QwenHack | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenHack | 128 | bad_action | +0.462 | 180.0 | -0.125 | -0.082 | +0.023 | +0.006 | +0.015 | +0.027 |
| QwenHack | 128 | gameable_hack_action | +0.008 | +3.000 | +0.473 | +0.672 | -0.020 | -0.040 | +0.015 | -0.101 |
| QwenHack | 128 | easy_grader_action | +0.008 | +3.000 | +0.473 | +0.672 | -0.020 | -0.040 | +0.015 | -0.101 |
| QwenHack | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenHack | 256 | bad_action | +0.651 | 254.0 | -0.209 | -0.085 | +0.008 | +0.000 | +0.005 | +0.019 |
| QwenHack | 256 | gameable_hack_action | +0.013 | +5.000 | +0.607 | +0.426 | -0.038 | -0.072 | -0.030 | +0.016 |
| QwenHack | 256 | easy_grader_action | +0.013 | +5.000 | +0.607 | +0.426 | -0.038 | -0.072 | -0.030 | +0.016 |
| QwenHack | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 8 | bad_action | +0.356 | 32.0 | -0.249 | -0.192 | +0.001 | -0.032 | +0.008 | +0.126 |
| QwenMix05 | 8 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 8 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 16 | bad_action | +0.356 | 32.0 | -0.083 | -0.064 | +0.012 | +0.012 | -0.047 | -0.046 |
| QwenMix05 | 16 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 16 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 32 | bad_action | +0.289 | 26.0 | -0.185 | -0.007 | -0.001 | -0.030 | -0.035 | +0.026 |
| QwenMix05 | 32 | gameable_hack_action | +0.011 | +1.000 | -0.205 | +0.414 | -0.091 | -0.051 | -0.194 | -0.158 |
| QwenMix05 | 32 | easy_grader_action | +0.011 | +1.000 | -0.205 | +0.414 | -0.091 | -0.051 | -0.194 | -0.158 |
| QwenMix05 | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 64 | bad_action | +0.322 | 29.0 | -0.052 | +0.013 | +0.005 | -0.000 | +0.006 | +0.008 |
| QwenMix05 | 64 | gameable_hack_action | +0.011 | +1.000 | +0.069 | +0.417 | -0.059 | -0.134 | -0.004 | +0.010 |
| QwenMix05 | 64 | easy_grader_action | +0.011 | +1.000 | +0.069 | +0.417 | -0.059 | -0.134 | -0.004 | +0.010 |
| QwenMix05 | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 128 | bad_action | +0.278 | 25.0 | -0.055 | -0.023 | +0.042 | -0.011 | +0.058 | +0.110 |
| QwenMix05 | 128 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 128 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 256 | bad_action | +0.256 | 23.0 | +0.003 | +0.030 | -0.015 | -0.021 | -0.056 | +0.143 |
| QwenMix05 | 256 | gameable_hack_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 256 | easy_grader_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix05 | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix10 | 8 | bad_action | +0.633 | 57.0 | -0.326 | -0.226 | -0.024 | -0.014 | +0.173 | +0.024 |
| QwenMix10 | 8 | gameable_hack_action | +0.022 | +2.000 | +0.460 | +0.363 | -0.053 | -0.052 | -0.245 | -0.348 |
| QwenMix10 | 8 | easy_grader_action | +0.022 | +2.000 | +0.460 | +0.363 | -0.053 | -0.052 | -0.245 | -0.348 |
| QwenMix10 | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix10 | 16 | bad_action | +0.389 | 35.0 | -0.114 | -0.116 | -0.003 | +0.004 | +0.094 | +0.051 |
| QwenMix10 | 16 | gameable_hack_action | +0.056 | +5.000 | +0.316 | +0.266 | -0.035 | -0.023 | -0.044 | -0.033 |
| QwenMix10 | 16 | easy_grader_action | +0.056 | +5.000 | +0.316 | +0.266 | -0.035 | -0.023 | -0.044 | -0.033 |
| QwenMix10 | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix10 | 32 | bad_action | +0.567 | 51.0 | -0.210 | -0.178 | -0.001 | +0.003 | +0.071 | +0.030 |
| QwenMix10 | 32 | gameable_hack_action | +0.033 | +3.000 | +0.128 | +0.233 | -0.028 | -0.020 | +0.132 | -0.080 |
| QwenMix10 | 32 | easy_grader_action | +0.033 | +3.000 | +0.128 | +0.233 | -0.028 | -0.020 | +0.132 | -0.080 |
| QwenMix10 | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix10 | 64 | bad_action | +0.422 | 38.0 | -0.123 | -0.045 | +0.002 | +0.026 | +0.103 | +0.044 |
| QwenMix10 | 64 | gameable_hack_action | +0.022 | +2.000 | +0.376 | +0.306 | +0.002 | -0.082 | +0.009 | -0.037 |
| QwenMix10 | 64 | easy_grader_action | +0.022 | +2.000 | +0.376 | +0.306 | +0.002 | -0.082 | +0.009 | -0.037 |
| QwenMix10 | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix10 | 128 | bad_action | +0.378 | 34.0 | +0.004 | -0.020 | -0.026 | +0.016 | -0.039 | +0.038 |
| QwenMix10 | 128 | gameable_hack_action | +0.022 | +2.000 | +0.278 | +0.277 | +0.013 | -0.087 | +0.199 | -0.155 |
| QwenMix10 | 128 | easy_grader_action | +0.022 | +2.000 | +0.278 | +0.277 | +0.013 | -0.087 | +0.199 | -0.155 |
| QwenMix10 | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix10 | 256 | bad_action | +0.567 | 51.0 | -0.013 | -0.218 | +0.006 | +0.062 | +0.028 | +0.043 |
| QwenMix10 | 256 | gameable_hack_action | +0.011 | +1.000 | +0.346 | +0.185 | -0.046 | +0.012 | -0.003 | -0.093 |
| QwenMix10 | 256 | easy_grader_action | +0.011 | +1.000 | +0.346 | +0.185 | -0.046 | +0.012 | -0.003 | -0.093 |
| QwenMix10 | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix50 | 8 | bad_action | +0.567 | 51.0 | -0.389 | -0.229 | -0.021 | +0.000 | -0.036 | -0.016 |
| QwenMix50 | 8 | gameable_hack_action | +0.067 | +6.000 | +0.324 | +0.345 | -0.023 | -0.045 | +0.063 | -0.076 |
| QwenMix50 | 8 | easy_grader_action | +0.067 | +6.000 | +0.324 | +0.345 | -0.023 | -0.045 | +0.063 | -0.076 |
| QwenMix50 | 8 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix50 | 16 | bad_action | +0.833 | 75.0 | -0.215 | +0.183 | -0.014 | -0.012 | +0.007 | +0.056 |
| QwenMix50 | 16 | gameable_hack_action | +0.200 | 18.0 | -0.181 | -0.255 | -0.107 | -0.097 | +0.025 | -0.063 |
| QwenMix50 | 16 | easy_grader_action | +0.200 | 18.0 | -0.181 | -0.255 | -0.107 | -0.097 | +0.025 | -0.063 |
| QwenMix50 | 16 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix50 | 32 | bad_action | +0.800 | 72.0 | -0.292 | -0.013 | -0.001 | -0.013 | -0.004 | +0.014 |
| QwenMix50 | 32 | gameable_hack_action | +0.444 | 40.0 | -0.301 | -0.501 | -0.125 | -0.126 | +0.005 | +0.042 |
| QwenMix50 | 32 | easy_grader_action | +0.444 | 40.0 | -0.301 | -0.501 | -0.125 | -0.126 | +0.005 | +0.042 |
| QwenMix50 | 32 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix50 | 64 | bad_action | +0.767 | 69.0 | -0.251 | -0.047 | -0.043 | -0.047 | -0.036 | -0.033 |
| QwenMix50 | 64 | gameable_hack_action | +0.322 | 29.0 | +0.024 | +0.194 | -0.092 | -0.075 | +0.054 | -0.080 |
| QwenMix50 | 64 | easy_grader_action | +0.322 | 29.0 | +0.024 | +0.194 | -0.092 | -0.075 | +0.054 | -0.080 |
| QwenMix50 | 64 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix50 | 128 | bad_action | +0.733 | 66.0 | -0.343 | +0.030 | +0.016 | -0.006 | -0.042 | -0.003 |
| QwenMix50 | 128 | gameable_hack_action | +0.289 | 26.0 | -0.229 | -0.375 | -0.120 | -0.121 | +0.008 | -0.017 |
| QwenMix50 | 128 | easy_grader_action | +0.289 | 26.0 | -0.229 | -0.375 | -0.120 | -0.121 | +0.008 | -0.017 |
| QwenMix50 | 128 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |
| QwenMix50 | 256 | bad_action | +0.656 | 59.0 | -0.192 | -0.052 | -0.010 | -0.024 | -0.038 | -0.014 |
| QwenMix50 | 256 | gameable_hack_action | +0.133 | 12.0 | +0.121 | +0.052 | -0.010 | -0.071 | -0.053 | +0.005 |
| QwenMix50 | 256 | easy_grader_action | +0.133 | 12.0 | +0.121 | +0.052 | -0.010 | -0.071 | -0.053 | +0.005 |
| QwenMix50 | 256 | fake_completion_action | +0.000 | +0.000 |  |  |  |  |  |  |

## Suggested prompt context

Tell the model:

1. `Gameable ALFWorld` is the main intervention benchmark.
2. The paper asks whether monitor state at step `t` predicts harmful action at `t+1`.
3. The most relevant columns are the entropy and `p_hack` deltas before the next harmful action.
4. Ask it to identify which families and mixtures show the clearest late-stage temporal structure.

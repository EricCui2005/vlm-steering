# Modal compute log

Costs use full app lifetimes from the logs, so they are upper bounds.

| Run | GPU | GPU-hours | Approximate cost |
|---|---|---:|---:|
| phase0_smoke, attempt 1 (sampler needed a CUDA compiler) | L40S | 0.217 | $0.52 |
| phase0_smoke, attempt 2 (wrong symbol name, 16 generations saved) | L40S | 0.067 | $0.16 |
| phase0_smoke, final run | L40S | 0.083 | $0.20 |
| phase1_baselines + phase3_sweep_inorder (one job) | L40S | 0.175 | $0.42 |
| gemma4_e4b sanity_text_instruction | L40S | 0.081 | $0.19 |
| qwen3_vl_8b baselines + sanity + sweep (one job) | L40S | 0.112 | $0.27 |
| gemma4_31b weight download (CPU only, about 10 min) | CPU | 0.000 | $0.03 |
| gemma4_31b baselines + sanity + sweep (one job) | H200 | 0.198 | $1.04 |
| qwen3_vl_8b controls_inorder | L40S | 0.055 | $0.13 |
| gemma4_31b controls_inorder | H200 | 0.098 | $0.52 |
| qwen3_vl_8b confirmation_inorder (n=100, 1000 generations) | L40S | 0.044 | $0.10 |
| qwen3_vl_8b choose_strategy_inorder (300 one-word answers) | L40S | 0.045 | $0.11 |
| gemma3_4b weight download (CPU only, about 1 min) | CPU | 0.000 | $0.01 |
| gemma3_4b baselines + sanity + sweep + controls (one job) | L40S | 0.089 | $0.21 |
| gemma3_4b fib + dfs sweeps and blank controls (one job) | L40S | 0.109 | $0.26 |
| gemma3_4b confirmation_fib (n=100, 1300 generations) | L40S | 0.049 | $0.12 |
| **Total** | | **1.422** | **$4.28** |

# Phase 0 smoke test

- Model: `google/gemma-4-E4B-it` at `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`
- GPU: L40S as requested (name not recorded; vLLM reported 44.42 GiB)
- Libraries: {'vllm': '0.33.0', 'transformers': '5.20.0', 'torch': '2.14.0', 'modal': '1.6.3'}
- Generations: 48 (32 in the final run)
- Mean output length: 122 tokens
- Truncated at the token limit: 0
- Throughput: 512 output tokens/s
- Approximate image tokens: 259
- Download: 0 s
- Model load: 196 s
- Generation: 9 s
- Approximate cost: $0.20

## Labels (n = 8 per row, not evidence of steering)

| Problem | Condition | Recursive | Iterative | Other |
|---|---|---:|---:|---:|
| fib | baseline | 0 | 8 | 0 |
| fib | blue_right_arrow | 1 | 7 | 0 |
| fib | red_heart | 1 | 7 | 0 |
| inorder | baseline | 0 | 8 | 0 |
| inorder | blue_right_arrow | 0 | 8 | 0 |
| inorder | red_heart | 0 | 8 | 0 |

## Estimate for google/gemma-4-E4B-it on L40S

Assumes 122 tokens per generation at 512 tokens/s. Larger batches may run faster than this small test.

| Phase | Generations | GPU minutes | Cost |
|---|---:|---:|---:|
| baseline | 240 | 4 | $0.17 |
| pilot | 1,740 | 10 | $0.41 |
| sweep | 5,760 | 26 | $1.04 |
| confirmation | 2,400 | 13 | $0.51 |
| perturbations | 280 | 34 | $1.37 |
| **Total** | **10,420** | **88** | **$3.50** |

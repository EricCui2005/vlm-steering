# Visual symbol steering in an open-weight VLM

Tests whether simple colored symbols steer an open-weight VLM toward recursive or
iterative code, replicating Tartaglini et al., *A Heart Means Recursion*. Only Phase 0
is implemented so far.

## Setup

You need [uv](https://docs.astral.sh/uv/) and a logged-in Modal account (`modal setup`).

```bash
uv sync
```

```bash
uv run pytest
```

## Phase 0

This script creates the Modal Volumes, builds and uploads the symbols, runs a 48-generation
smoke test, and writes `results/phase0_summary.md`. A 20-minute job timeout keeps the cost
under \$1.

```bash
scripts/run_phase0.sh
```

To build only the symbols, which is free and runs locally:

```bash
uv run python scripts/build_symbols.py
```

## Layout

| Path | Purpose |
|---|---|
| `config/experiment.yaml` | Models, sampling, problems, colors, sample counts, prices |
| `modal_app.py` | Container image, Volumes and GPU jobs |
| `steering/prompts.py` | Prompt wording |
| `steering/samples.py` | What one sample is, seeds, balanced option order |
| `steering/generation.py` | Generation loop that saves results and skips finished work |
| `steering/vllm_backend.py` | vLLM calls, used only in the GPU container |
| `steering/classifier.py` | Labels code as recursive, iterative or other |
| `steering/symbols.py` | Draws the 288 symbols |
| `scripts/` | Phase 0 runner, symbol builder and report |

# Visual symbol steering in open-weight VLMs

Do simple colored symbols steer an open-weight vision-language model toward recursive
or iterative code? This replicates the visual steering experiments of Tartaglini et al.,
*A Heart Means Recursion*, on open models, with generation on Modal.

## Main result

In **Qwen3-VL-8B-Instruct**, an orange U-turn arrow steers inorder traversal toward
iterative code, and it beats every control (n = 100 per condition, fresh seeds):

| Condition | P(iterative) |
|---|---|
| orange U-turn arrow | **0.70** |
| strongest photo (photo_5) | 0.48 |
| blank white image | 0.18 |
| no image | 0.00 |

- **Significance:** two-sided Fisher exact test. Arrow vs blank, p = 9×10⁻¹⁴; arrow vs strongest photo, p = 0.002.
- **What's symbol-specific:** about +0.2 of the shift. The rest comes from having any image at all.
- **Self-report:** asked to name its strategy, Qwen says "iterative" in 300/300 samples whatever the image, so its self-report doesn't track its behavior.

| Model | Outcome |
|---|---|
| Qwen3-VL-8B | Symbol-specific steering on inorder, confirmed at n = 100 |
| Gemma 4 E4B | No steering (iterative at baseline on every problem) |
| Gemma 4 31B | Steered by the prompt wording; a blank image does as much as any symbol |
| Gemma 3 4B | No steering on inorder; on fib, noise images steer more than any symbol |

Total compute so far is about 1.4 GPU-hours, roughly $4.30 ([results/compute_log.md](results/compute_log.md)).

## Method

- **Prompts:** the paper's. With an image, the image comes first and the text says "Looking at the provided image, choose a coding strategy (recursion or iteration)…". Option order is balanced.
- **Symbols:** 288 symbols (24 shapes × 12 colors, 400×400 px), drawn by `steering/symbols.py`.
- **Controls:**
  - no image
  - a blank white image
  - Gaussian noise
  - five natural photos from the Flickr 1k test set
  - text captions of symbols, repeated to match the image's token count
  - a text instruction ceiling ("Solve it using an iterative approach")
- **Labels:** recursive / iterative / other, from the call graph of the generated code (`steering/classifier.py`).
- **Statistics:**
  - Wilson intervals
  - two-sided Fisher exact tests
  - Benjamini–Hochberg FDR (q = 0.05) for the 288-symbol sweeps
  - Bonferroni correction for pre-chosen confirmations
- **Generation:** vLLM 0.33 on Modal, with a pinned model revision for each model in `config/experiment.yaml`.

## Setup

You need [uv](https://docs.astral.sh/uv/) and a logged-in Modal account (`modal setup`).

```bash
uv sync
```

```bash
uv run pytest
```

Gemma 3 is gated on Hugging Face. Accept its license, then store a token as a Modal secret. Every job reads this secret, so create it even if you only run Qwen:

```bash
uv run modal secret create huggingface HF_TOKEN=<your token>
```

## Running experiments

1. Build the stimuli and upload them to the `steering-results` Volume. `scripts/run_phase0.sh` creates the Volumes and uploads the symbols, then runs a short smoke test.

    ```bash
    uv run python scripts/build_symbols.py
    ```

    ```bash
    uv run python scripts/build_controls.py PHOTO_DIR
    ```

    ```bash
    uv run modal volume put --force steering-results stimuli/symbols /stimuli/symbols
    ```

    ```bash
    uv run modal volume put --force steering-results stimuli/controls /stimuli/symbols
    ```

2. Download the weights on a CPU container, then run named runs from `config/experiment.yaml` on the model's GPU. Finished samples are skipped on re-runs.

    ```bash
    uv run modal run modal_app.py::download_model --model qwen3_vl_8b
    ```

    ```bash
    uv run modal run modal_app.py::experiment --model qwen3_vl_8b --runs phase1_baselines,phase3_sweep_inorder,controls_inorder
    ```

3. Copy the generations into `results/runs/` with `modal volume get`, then analyze them.

    ```bash
    uv run python scripts/analyze_sweep.py qwen3_vl_8b inorder
    ```

    ```bash
    PYTHONPATH=scripts uv run python scripts/analyze_confirmation.py qwen3_vl_8b confirmation_inorder
    ```

Models: `qwen3_vl_8b`, `gemma4_e4b`, `gemma4_31b` and `gemma3_4b`.

| Runs | What they test |
|---|---|
| `phase1_baselines` | No image, six problems |
| `sanity_text_instruction` | Text instruction ceiling |
| `phase3_sweep_{inorder,fib,dfs}` | All 288 symbols |
| `controls_{inorder,fib,dfs}` | Blank, noise, photos and captions |
| `confirmation_inorder`, `confirmation_fib` | Pre-chosen symbols vs controls at n = 100 |
| `choose_strategy_inorder` | The model names its strategy instead of writing code |

## Layout

| Path | Purpose |
|---|---|
| `config/experiment.yaml` | Models, sampling, problems, runs, colors, prices |
| `modal_app.py` | Container image, Volumes, secret and GPU jobs |
| `steering/prompts.py` | Prompt wording |
| `steering/samples.py` | What one sample is, seeds, balanced option order, tasks |
| `steering/generation.py` | Generation loop that saves results and skips finished work |
| `steering/vllm_backend.py` | vLLM calls, used only in the GPU container |
| `steering/classifier.py` | Labels code (and one-word answers) as recursive, iterative or other |
| `steering/symbols.py`, `steering/controls.py` | Draw the symbols and control images |
| `steering/stats.py` | Wilson intervals, Fisher exact test, Benjamini–Hochberg |
| `scripts/` | Stimulus builders, Phase 0 runner, analysis scripts |
| `stimuli/` | The 288 symbols, control images and a contact sheet |
| `results/runs/{model}/{run}/` | Raw generations (`generations.jsonl`) and run metadata |
| `results/{model}/` | CSVs and plots for each model |

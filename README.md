# Visual symbol steering in open-weight VLMs

A replication of Tartaglini et al., *A Heart Means Recursion*, on open models: can a simple colored symbol steer a VLM toward recursive or iterative code?

**Result.** In Qwen3-VL-8B, an orange U-turn arrow steers inorder traversal toward iterative code. It reaches 0.70 iterative, vs 0.48 for the strongest photo, 0.18 for a blank image and 0.00 with no image (n = 100; Fisher exact p = 0.002 vs the photo). Gemma 4 E4B, Gemma 4 31B and Gemma 3 4B show no symbol-specific steering.

## Usage

Requires [uv](https://docs.astral.sh/uv/), a Modal account, and a Modal secret `huggingface` holding `HF_TOKEN`.

```bash
uv sync && uv run pytest
```

```bash
uv run modal run modal_app.py::experiment --model qwen3_vl_8b --runs confirmation_inorder
```

```bash
PYTHONPATH=scripts uv run python scripts/analyze_confirmation.py qwen3_vl_8b confirmation_inorder
```

Models and runs are defined in `config/experiment.yaml`. Raw generations are in `results/runs/`, and plots and CSVs in `results/{model}/`.

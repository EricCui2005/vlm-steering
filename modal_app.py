"""The Modal setup: container image, storage, and the jobs that run on a GPU.

Phase 0 smoke test:

    modal run modal_app.py::smoke_test

Runs named in config/experiment.yaml, for one model, in one container so the model
loads once:

    modal run modal_app.py::experiment --model qwen3_vl_8b --runs phase1_baselines,phase3_sweep_inorder

Download large weights first, on a CPU container:

    modal run modal_app.py::download_model --model gemma4_31b
"""

import json
import os
import platform
import time
from pathlib import Path

import modal

from steering.config import CONFIG_PATH, load_config

CONFIG = load_config()
# The smoke test used the first model.
PRIMARY_MODEL = CONFIG["models"]["gemma4_e4b"]
CONTAINER = CONFIG["container"]

# Modal stops the job at this limit. At about $2.40/hr, it keeps the smoke test under $1.
SMOKE_TEST_TIMEOUT_SECONDS = 20 * 60
BATCH_SIZE = 16

# A cap on any experiment job: at most about $3.60 on an L40S or $7.50 on an H200.
EXPERIMENT_TIMEOUT_SECONDS = 90 * 60
# Larger batches let vLLM run many generations at once. Progress is saved after each batch.
EXPERIMENT_BATCH_SIZE = 512

HF_CACHE_DIR = "/hf-cache"
RESULTS_DIR = Path("/results")
# Every image a condition can name: the 288 symbols and the control images.
SYMBOLS_DIR = RESULTS_DIR / "stimuli" / "symbols"

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("steering-results", create_if_missing=True)
# Gated models (Gemma 3) need a Hugging Face token. It lives only in this Modal
# secret, created once with: modal secret create huggingface HF_TOKEN=...
hf_token = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.33.0", "transformers==5.20.0", "pyyaml>=6")
    .env({
        "HF_HOME": HF_CACHE_DIR,
        # vLLM's default FlashInfer sampler needs a CUDA compiler, which this image lacks.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .add_local_file(CONFIG_PATH, "/root/config/experiment.yaml")
    .add_local_python_source("steering")
)

app = modal.App("visual-symbol-steering", image=image)


@app.function(
    gpu=PRIMARY_MODEL["gpu"],
    cpu=CONTAINER["cpu_cores"],
    memory=CONTAINER["memory_gib"] * 1024,
    timeout=SMOKE_TEST_TIMEOUT_SECONDS,
    volumes={HF_CACHE_DIR: hf_cache, str(RESULTS_DIR): results_volume},
)
def smoke_test() -> dict:
    """Run up to 50 generations to show the pipeline works and to measure its speed."""
    # These imports need the GPU packages, which are only installed in the container.
    from huggingface_hub import snapshot_download

    from steering.generation import run_samples
    from steering.samples import BASELINE, plan_samples
    from steering.vllm_backend import VllmBackend

    run_name = "phase0_smoke"
    run_dir = RESULTS_DIR / "runs" / run_name
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        # Nothing left to do. Exit before the model loads, and keep the recorded timings.
        return json.loads(metadata_path.read_text())

    started_at = time.monotonic()

    smoke = CONFIG["smoke_test"]
    _require_symbol_images(smoke["symbols"])

    snapshot_download(PRIMARY_MODEL["id"], revision=PRIMARY_MODEL["revision"])
    hf_cache.commit()
    downloaded_at = time.monotonic()

    backend = VllmBackend(
        model_id=PRIMARY_MODEL["id"],
        revision=PRIMARY_MODEL["revision"],
        sampling=PRIMARY_MODEL["sampling"],
        symbols_dir=SYMBOLS_DIR,
        seed=CONFIG["seed"],
    )
    loaded_at = time.monotonic()

    samples = plan_samples(
        run_name=run_name,
        problems={problem_id: CONFIG["problems"][problem_id] for problem_id in smoke["problems"]},
        conditions=[BASELINE, *smoke["symbols"]],
        samples_per_condition=smoke["samples_per_condition"],
        base_seed=CONFIG["seed"],
    )
    stats = run_samples(
        samples,
        backend,
        run_dir / "generations.jsonl",
        batch_size=BATCH_SIZE,
        after_each_batch=results_volume.commit,
    )
    finished_at = time.monotonic()

    metadata = {
        "run_name": run_name,
        "model_id": PRIMARY_MODEL["id"],
        "model_revision": PRIMARY_MODEL["revision"],
        "gpu_requested": PRIMARY_MODEL["gpu"],
        "gpu_actual": _gpu_name(),
        "cpu_cores": CONTAINER["cpu_cores"],
        "memory_gib": CONTAINER["memory_gib"],
        "library_versions": _library_versions(),
        "python_version": platform.python_version(),
        "sampling": PRIMARY_MODEL["sampling"],
        "base_seed": CONFIG["seed"],
        "generations_this_run": stats.generations,
        "output_tokens_this_run": stats.output_tokens,
        "download_seconds": downloaded_at - started_at,
        "model_load_seconds": loaded_at - downloaded_at,
        "generation_seconds": finished_at - loaded_at,
        "total_seconds": finished_at - started_at,
        "hf_home": os.environ["HF_HOME"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    results_volume.commit()
    return metadata


# Settings shared by every experiment job. The GPU and memory differ by model size.
EXPERIMENT_JOB = dict(
    cpu=CONTAINER["cpu_cores"],
    timeout=EXPERIMENT_TIMEOUT_SECONDS,
    volumes={HF_CACHE_DIR: hf_cache, str(RESULTS_DIR): results_volume},
    secrets=[hf_token],
)


@app.function(gpu="L40S", memory=32 * 1024, **EXPERIMENT_JOB)
def experiment_on_l40s(model_key: str, runs: str) -> list[dict]:
    return _run_experiment(model_key, runs)


@app.function(gpu="H200", memory=64 * 1024, **EXPERIMENT_JOB)
def experiment_on_h200(model_key: str, runs: str) -> list[dict]:
    return _run_experiment(model_key, runs)


EXPERIMENT_JOBS_BY_GPU = {"L40S": experiment_on_l40s, "H200": experiment_on_h200}


@app.local_entrypoint()
def experiment(model: str, runs: str) -> None:
    """Run the named runs for one model, on the GPU its config asks for."""
    job = EXPERIMENT_JOBS_BY_GPU[CONFIG["models"][model]["gpu"]]
    for metadata in job.remote(model, runs):
        summary = {key: metadata[key] for key in ("run_name", "generations_this_run", "generation_seconds")}
        print(summary)


@app.function(cpu=2, memory=8 * 1024, timeout=60 * 60, volumes={HF_CACHE_DIR: hf_cache}, secrets=[hf_token])
def download_model(model: str) -> None:
    """Download weights on a cheap CPU container, so GPU time isn't spent waiting."""
    from huggingface_hub import snapshot_download

    settings = CONFIG["models"][model]
    snapshot_download(settings["id"], revision=settings["revision"])
    hf_cache.commit()


def _run_experiment(model_key: str, runs: str) -> list[dict]:
    """Generate every sample for the comma-separated run names, skipping saved ones."""
    from steering.generation import run_samples
    from steering.samples import BASELINE, SOLVE, is_text_condition, plan_samples, text_to_repeat
    from steering.symbols import library_names
    from steering.vllm_backend import VllmBackend

    model = CONFIG["models"][model_key]

    def conditions_for(run: dict) -> list[str]:
        if run["conditions"] == "all_symbols":
            return library_names(CONFIG["colors"])
        return run["conditions"]

    run_names = runs.split(",")
    all_conditions = [c for name in run_names for c in conditions_for(CONFIG["runs"][name])]
    text_conditions = [c for c in all_conditions if is_text_condition(c)]
    _require_symbol_images([c for c in all_conditions if c != BASELINE and c not in text_conditions])

    started_at = time.monotonic()
    backend = VllmBackend(
        model_id=model["id"],
        revision=model["revision"],
        sampling=model["sampling"],
        symbols_dir=SYMBOLS_DIR,
        seed=CONFIG["seed"],
    )
    model_load_seconds = time.monotonic() - started_at

    # Text conditions repeat their text until it takes about as many tokens as an image.
    image_tokens = backend.image_token_count("red_heart")
    text_repeats = {
        condition: round(image_tokens / backend.text_token_count(text_to_repeat(condition)))
        for condition in text_conditions
    }

    all_metadata = []
    for run_name in run_names:
        run = CONFIG["runs"][run_name]
        run_dir = RESULTS_DIR / "runs" / model_key / run_name
        samples = plan_samples(
            run_name=run_name,
            problems={problem_id: CONFIG["problems"][problem_id] for problem_id in run["problems"]},
            conditions=conditions_for(run),
            samples_per_condition=run["samples_per_condition"],
            base_seed=CONFIG["seed"],
            text_repeats=text_repeats,
            task=run.get("task", SOLVE),
        )

        run_started_at = time.monotonic()
        stats = run_samples(
            samples,
            backend,
            run_dir / "generations.jsonl",
            batch_size=EXPERIMENT_BATCH_SIZE,
            after_each_batch=results_volume.commit,
        )
        metadata = {
            "run_name": run_name,
            "run_config": run,
            **_environment_metadata(model),
            "image_tokens": image_tokens,
            "text_repeats": text_repeats,
            "generations_this_run": stats.generations,
            "output_tokens_this_run": stats.output_tokens,
            "model_load_seconds": model_load_seconds,
            "generation_seconds": time.monotonic() - run_started_at,
            "job_seconds_so_far": time.monotonic() - started_at,
        }
        # One line per job, so re-runs add to the history instead of replacing it.
        with (run_dir / "metadata.jsonl").open("a") as metadata_file:
            metadata_file.write(json.dumps(metadata) + "\n")
        results_volume.commit()
        all_metadata.append(metadata)

    return all_metadata


def _environment_metadata(model: dict) -> dict:
    return {
        "model_id": model["id"],
        "model_revision": model["revision"],
        "gpu_requested": model["gpu"],
        "gpu_actual": _gpu_name(),
        "cpu_cores": CONTAINER["cpu_cores"],
        "library_versions": _library_versions(),
        "python_version": platform.python_version(),
        "sampling": model["sampling"],
        "base_seed": CONFIG["seed"],
    }


def _require_symbol_images(symbols: list[str]) -> None:
    """Fail before the GPU spends time loading the model, not partway through a run."""
    missing = [name for name in symbols if not (SYMBOLS_DIR / f"{name}.png").exists()]
    if missing:
        raise FileNotFoundError(f"No image in {SYMBOLS_DIR} for: {missing}")


def _gpu_name() -> str:
    import torch

    return torch.cuda.get_device_name(0)


def _library_versions() -> dict[str, str]:
    from importlib.metadata import version

    versions = {name: version(name) for name in ("vllm", "transformers", "torch")}
    # Modal's own package lives outside the container's Python environment.
    return versions | {"modal": modal.__version__}

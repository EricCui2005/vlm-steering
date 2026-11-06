"""Summarize the smoke test and estimate the time and cost of Phases 1-5.

Reads results/runs/phase0_smoke/ and writes results/phase0_summary.md and
results/compute_log.md.
"""

import json
from collections import Counter
from pathlib import Path

from steering.config import load_config

RUN_DIR = Path("results/runs/phase0_smoke")
SUMMARY_PATH = Path("results/phase0_summary.md")
COMPUTE_LOG_PATH = Path("results/compute_log.md")

# Phase 5 also needs teacher-forced log-probs in plain transformers. The smoke test
# can't measure that, so this is an assumption rather than a measurement.
TRANSFORMERS_WORK_SECONDS = 30 * 60

# Attempts that crashed before writing metadata. Their times are the full app lifetimes,
# including image builds, so these costs are upper bounds.
FAILED_ATTEMPTS = [
    ("phase0_smoke, attempt 1 (sampler needed a CUDA compiler)", 13 * 60),
    ("phase0_smoke, attempt 2 (wrong symbol name, 16 generations saved)", 4 * 60),
]


def main() -> None:
    config = load_config()
    metadata = json.loads((RUN_DIR / "metadata.json").read_text())
    records = [json.loads(line) for line in (RUN_DIR / "generations.jsonl").open()]

    dollars_per_second = _dollars_per_second(config, metadata)
    sections = [
        _run_section(metadata, records, dollars_per_second),
        _label_section(records),
        _estimate_section(config, metadata, records, dollars_per_second),
    ]

    SUMMARY_PATH.write_text("\n\n".join(sections) + "\n")
    _write_compute_log(metadata, dollars_per_second)
    print(SUMMARY_PATH.read_text())


def _dollars_per_second(config: dict, metadata: dict) -> float:
    pricing = config["modal_pricing_per_second"]
    return (
        pricing["gpu"][metadata["gpu_requested"]]
        + pricing["cpu_core"] * metadata["cpu_cores"]
        + pricing["memory_gib"] * metadata["memory_gib"]
    )


def _tokens_per_second(metadata: dict) -> float:
    return metadata["output_tokens_this_run"] / metadata["generation_seconds"]


def _image_token_count(records: list[dict]) -> float:
    """Extra prompt tokens with an image, compared with a baseline prompt."""
    def mean_prompt_tokens(rows: list[dict]) -> float:
        return sum(r["prompt_tokens"] for r in rows) / len(rows)

    with_image = [r for r in records if r["condition"] != "baseline"]
    without_image = [r for r in records if r["condition"] == "baseline"]
    return mean_prompt_tokens(with_image) - mean_prompt_tokens(without_image)


def _run_section(metadata: dict, records: list[dict], dollars_per_second: float) -> str:
    truncated = sum(r["finish_reason"] == "length" for r in records)
    mean_tokens = sum(r["output_tokens"] for r in records) / len(records)
    lines = [
        "# Phase 0 smoke test",
        "",
        f"- Model: `{metadata['model_id']}` at `{metadata['model_revision']}`",
        f"- GPU: {metadata['gpu_actual']}",
        f"- Libraries: {metadata['library_versions']}",
        f"- Generations: {len(records)} ({metadata['generations_this_run']} in the final run)",
        f"- Mean output length: {mean_tokens:.0f} tokens",
        f"- Truncated at the token limit: {truncated}",
        f"- Throughput: {_tokens_per_second(metadata):.0f} output tokens/s",
        f"- Approximate image tokens: {_image_token_count(records):.0f}",
        f"- Download: {metadata['download_seconds']:.0f} s",
        f"- Model load: {metadata['model_load_seconds']:.0f} s",
        f"- Generation: {metadata['generation_seconds']:.0f} s",
        f"- Approximate cost: ${metadata['total_seconds'] * dollars_per_second:.2f}",
    ]
    return "\n".join(lines)


def _label_section(records: list[dict]) -> str:
    """Label counts per condition. With n = 8, these only show that the pipeline works."""
    counts = Counter((r["problem_id"], r["condition"], r["label"]) for r in records)
    groups = sorted({(r["problem_id"], r["condition"]) for r in records})

    lines = [
        "## Labels (n = 8 per row, not evidence of steering)",
        "",
        "| Problem | Condition | Recursive | Iterative | Other |",
        "|---|---|---:|---:|---:|",
    ]
    for problem_id, condition in groups:
        recursive = counts[(problem_id, condition, "recursive")]
        iterative = counts[(problem_id, condition, "iterative")]
        other = counts[(problem_id, condition, "other")]
        lines.append(f"| {problem_id} | {condition} | {recursive} | {iterative} | {other} |")
    return "\n".join(lines)


def _estimate_section(
    config: dict, metadata: dict, records: list[dict], dollars_per_second: float
) -> str:
    mean_tokens = sum(r["output_tokens"] for r in records) / len(records)
    tokens_per_second = _tokens_per_second(metadata)
    startup_seconds = metadata["model_load_seconds"]

    lines = [
        f"## Estimate for {metadata['model_id']} on {metadata['gpu_requested']}",
        "",
        f"Assumes {mean_tokens:.0f} tokens per generation at {tokens_per_second:.0f} tokens/s. "
        "Larger batches may run faster than this small test.",
        "",
        "| Phase | Generations | GPU minutes | Cost |",
        "|---|---:|---:|---:|",
    ]

    total_generations = total_seconds = 0.0
    for phase, plan in config["planned_runs"].items():
        generations = plan["conditions"] * plan["samples"]
        seconds = startup_seconds + generations * mean_tokens / tokens_per_second
        if phase == "perturbations":
            seconds += TRANSFORMERS_WORK_SECONDS
        total_generations += generations
        total_seconds += seconds
        lines.append(
            f"| {phase} | {generations:,} | {seconds / 60:.0f} | ${seconds * dollars_per_second:.2f} |"
        )

    lines.append(
        f"| **Total** | **{total_generations:,.0f}** | **{total_seconds / 60:.0f}** "
        f"| **${total_seconds * dollars_per_second:.2f}** |"
    )
    return "\n".join(lines)


def _write_compute_log(metadata: dict, dollars_per_second: float) -> None:
    runs = [*FAILED_ATTEMPTS, (f"{metadata['run_name']}, final run", metadata["total_seconds"])]
    rows = [
        f"| {name} | {metadata['gpu_requested']} | {seconds / 3600:.3f} | ${seconds * dollars_per_second:.2f} |\n"
        for name, seconds in runs
    ]
    total_seconds = sum(seconds for _, seconds in runs)
    COMPUTE_LOG_PATH.write_text(
        "# Modal compute log\n\n"
        "Costs use full app lifetimes from the logs, so they are upper bounds.\n\n"
        "| Run | GPU | GPU-hours | Approximate cost |\n"
        "|---|---|---:|---:|\n"
        + "".join(rows)
        + f"| **Total** | | **{total_seconds / 3600:.3f}** | **${total_seconds * dollars_per_second:.2f}** |\n"
    )


if __name__ == "__main__":
    main()

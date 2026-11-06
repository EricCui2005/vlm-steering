"""Run samples through a model backend and save each result as it finishes.

This module doesn't depend on vLLM or Modal. Anything with a `generate` method
works as a backend, so tests can run the whole loop with a fake model.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from steering.classifier import classify, classify_choice, extract_code
from steering.samples import CHOOSE_STRATEGY, Sample


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: str


@dataclass(frozen=True)
class RunStats:
    """What this call generated, not counting samples saved by earlier runs."""

    generations: int
    output_tokens: int


class Backend(Protocol):
    def generate(self, samples: list[Sample]) -> list[Completion]: ...


def completed_keys(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    with results_path.open() as results:
        return {json.loads(line)["key"] for line in results if line.strip()}


def build_record(sample: Sample, completion: Completion) -> dict:
    if sample.task == CHOOSE_STRATEGY:
        code, label = None, classify_choice(completion.text)
    else:
        code = extract_code(completion.text)
        label = classify(code)
    return {
        "key": sample.key,
        "run_name": sample.run_name,
        "problem_id": sample.problem_id,
        "condition": sample.condition,
        "option_order": sample.option_order,
        "index": sample.index,
        "seed": sample.seed,
        "task": sample.task,
        "prompt_text": sample.prompt_text,
        **asdict(completion),
        "code": code,
        "label": label,
    }


def run_samples(
    samples: list[Sample],
    backend: Backend,
    results_path: Path,
    batch_size: int,
    after_each_batch: Callable[[], None] = lambda: None,
) -> RunStats:
    """Generate every sample not already saved.

    `after_each_batch` lets the caller persist progress, such as a Volume commit.
    """
    finished = completed_keys(results_path)
    remaining = [sample for sample in samples if sample.key not in finished]

    output_tokens = 0
    results_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start : start + batch_size]
        completions = backend.generate(batch)
        with results_path.open("a") as results:
            for sample, completion in zip(batch, completions, strict=True):
                results.write(json.dumps(build_record(sample, completion)) + "\n")
                output_tokens += completion.output_tokens
        after_each_batch()

    return RunStats(generations=len(remaining), output_tokens=output_tokens)

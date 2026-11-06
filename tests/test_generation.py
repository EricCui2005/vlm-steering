"""Run the generation loop end to end with a fake model, without Modal or a GPU."""

import json
from collections import Counter

from steering.generation import Completion, run_samples
from steering.prompts import ITERATION_FIRST, RECURSION_FIRST
from steering.samples import BASELINE, CHOOSE_STRATEGY, plan_samples

PROBLEMS = {"fib": "Write fib(n)."}
FAKE_RESPONSE = "```python\ndef fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n```"


class FakeBackend:
    def __init__(self):
        self.generated_count = 0

    def generate(self, samples):
        self.generated_count += len(samples)
        return [Completion(FAKE_RESPONSE, 10, 20, "stop") for _ in samples]


def plan(samples_per_condition=4):
    return plan_samples("test", PROBLEMS, [BASELINE, "red_heart"], samples_per_condition, base_seed=1)


def test_plan_balances_option_order():
    orders = Counter(sample.option_order for sample in plan())
    assert orders == {RECURSION_FIRST: 4, ITERATION_FIRST: 4}


def test_every_sample_has_its_own_stable_seed():
    seeds = [sample.seed for sample in plan()]
    assert len(set(seeds)) == len(seeds)
    assert seeds == [sample.seed for sample in plan()]


def test_only_image_conditions_mention_the_image():
    for sample in plan():
        assert ("provided image" in sample.prompt_text) == sample.has_image


def test_results_are_saved_and_labeled(tmp_path):
    results_path = tmp_path / "generations.jsonl"
    run_samples(plan(), FakeBackend(), results_path, batch_size=3)

    records = [json.loads(line) for line in results_path.open()]
    assert len(records) == 8
    assert {record["label"] for record in records} == {"recursive"}
    assert records[0]["text"] == FAKE_RESPONSE


def test_rerun_skips_finished_samples(tmp_path):
    results_path = tmp_path / "generations.jsonl"
    run_samples(plan(samples_per_condition=2), FakeBackend(), results_path, batch_size=3)

    backend = FakeBackend()
    run_samples(plan(samples_per_condition=4), backend, results_path, batch_size=3)

    assert backend.generated_count == 4
    assert sum(1 for _ in results_path.open()) == 8


def test_stats_count_only_this_run(tmp_path):
    results_path = tmp_path / "generations.jsonl"
    run_samples(plan(samples_per_condition=2), FakeBackend(), results_path, batch_size=3)

    stats = run_samples(plan(samples_per_condition=4), FakeBackend(), results_path, batch_size=3)

    assert stats.generations == 4
    assert stats.output_tokens == 4 * 20


def test_text_instruction_prompt_repeats_the_word_and_has_no_image():
    sample = plan_samples("t", PROBLEMS, ["text_instruction_iterative"], 2, base_seed=1, text_repeats={"text_instruction_iterative": 3})[0]
    assert sample.prompt_text.startswith("iterative iterative iterative\n")
    assert "Solve it using a iterative approach." in sample.prompt_text
    assert not sample.has_image


def test_caption_prompt_repeats_the_caption_before_the_image_prompt():
    sample = plan_samples("t", PROBLEMS, ["caption_orange_u_turn_arrow"], 2, base_seed=1,
                          text_repeats={"caption_orange_u_turn_arrow": 2})[0]
    caption = "An image of a orange u turn arrow"
    assert sample.prompt_text.startswith(f"{caption} {caption}\nSolve the following")
    assert "Looking at the provided image" in sample.prompt_text
    assert not sample.has_image


def test_choose_strategy_prompt_asks_for_one_word_in_option_order():
    samples = plan_samples("t", PROBLEMS, ["orange_u_turn_arrow"], 2, base_seed=1, task=CHOOSE_STRATEGY)
    by_order = {sample.option_order: sample.prompt_text for sample in samples}
    assert "Looking at the provided image" in by_order[RECURSION_FIRST]
    assert "Answer with exactly one word: recursive or iterative." in by_order[RECURSION_FIRST]
    assert "Answer with exactly one word: iterative or recursive." in by_order[ITERATION_FIRST]
    assert "Problem: Write fib(n)." in by_order[RECURSION_FIRST]


def test_choose_strategy_answers_are_labeled_without_code(tmp_path):
    class OneWordBackend:
        def generate(self, samples):
            return [Completion("Iterative", 10, 2, "stop") for _ in samples]

    samples = plan_samples("t", PROBLEMS, ["orange_u_turn_arrow"], 2, base_seed=1, task=CHOOSE_STRATEGY)
    results_path = tmp_path / "generations.jsonl"
    run_samples(samples, OneWordBackend(), results_path, batch_size=4)

    records = [json.loads(line) for line in results_path.open()]
    assert {record["label"] for record in records} == {"iterative"}
    assert {record["task"] for record in records} == {CHOOSE_STRATEGY}
    assert all(record["code"] is None for record in records)

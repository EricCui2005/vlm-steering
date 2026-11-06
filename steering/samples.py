"""Define what one sample is, and plan balanced sets of samples."""

import hashlib
from dataclasses import dataclass

from steering.prompts import (
    ITERATION_FIRST,
    RECURSION_FIRST,
    baseline_prompt,
    caption_prompt,
    choose_strategy_prompt,
    image_prompt,
    text_instruction_prompt,
)

BASELINE = "baseline"
# Text-only conditions, mapped to the strategy word they repeat and ask for.
TEXT_INSTRUCTIONS = {
    "text_instruction_recursive": "recursive",
    "text_instruction_iterative": "iterative",
}
# Caption conditions are named caption_{color}_{shape}, like caption_red_heart.
CAPTION_PREFIX = "caption_"

# What the model is asked to do. SOLVE writes the full solution; CHOOSE_STRATEGY
# only names the strategy, in one word, so it needs very few output tokens.
SOLVE = "solve"
CHOOSE_STRATEGY = "choose_strategy"
CHOOSE_STRATEGY_MAX_NEW_TOKENS = 16


def is_text_condition(condition: str) -> bool:
    """Text stands in for the image in these conditions."""
    return condition in TEXT_INSTRUCTIONS or condition.startswith(CAPTION_PREFIX)


def text_to_repeat(condition: str) -> str:
    """The word or phrase a text condition repeats in place of the image."""
    if condition in TEXT_INSTRUCTIONS:
        return TEXT_INSTRUCTIONS[condition]
    color, shape = condition.removeprefix(CAPTION_PREFIX).split("_", 1)
    return f"An image of a {color} {shape.replace('_', ' ')}"


@dataclass(frozen=True)
class Sample:
    """One generation to run.

    `condition` is BASELINE, an image name like red_heart or noise_1, or a text
    condition like caption_red_heart or text_instruction_recursive.
    """

    run_name: str
    problem_id: str
    problem: str
    condition: str
    option_order: str
    index: int
    base_seed: int
    text_repeats: int = 0  # Only used by text conditions.
    task: str = SOLVE

    @property
    def key(self) -> str:
        """Identifies this sample, so a re-run can skip it if it's already done."""
        return "/".join(
            [self.run_name, self.problem_id, self.condition, self.option_order, str(self.index)]
        )

    @property
    def seed(self) -> int:
        """A seed that is unique to this sample and the same on every re-run."""
        digest = hashlib.sha256(f"{self.base_seed}/{self.key}".encode()).hexdigest()
        return int(digest, 16) % 2**31

    @property
    def has_image(self) -> bool:
        return self.condition != BASELINE and not is_text_condition(self.condition)

    @property
    def prompt_text(self) -> str:
        if self.task == CHOOSE_STRATEGY:
            if not self.has_image:
                raise ValueError("The strategy-choice prompt refers to an image, so it needs one")
            return choose_strategy_prompt(self.problem, self.option_order)
        if self.condition in TEXT_INSTRUCTIONS:
            strategy = TEXT_INSTRUCTIONS[self.condition]
            return text_instruction_prompt(self.problem, strategy, self.text_repeats)
        if self.condition.startswith(CAPTION_PREFIX):
            caption = text_to_repeat(self.condition)
            return caption_prompt(self.problem, self.option_order, caption, self.text_repeats)
        build_prompt = image_prompt if self.has_image else baseline_prompt
        return build_prompt(self.problem, self.option_order)


def plan_samples(
    run_name: str,
    problems: dict[str, str],
    conditions: list[str],
    samples_per_condition: int,
    base_seed: int,
    text_repeats: dict[str, int] | None = None,
    task: str = SOLVE,
) -> list[Sample]:
    """Plan every problem x condition pair, half in each option order.

    `text_repeats` gives, for each text condition, how many times to repeat its
    text so it takes about as many tokens as an image.
    """
    text_repeats = text_repeats or {}
    if samples_per_condition % 2 != 0:
        raise ValueError("samples_per_condition must be even to balance option order")

    samples = []
    for problem_id, problem in problems.items():
        for condition in conditions:
            for option_order in (RECURSION_FIRST, ITERATION_FIRST):
                for index in range(samples_per_condition // 2):
                    repeats = text_repeats.get(condition, 0)
                    samples.append(
                        Sample(run_name, problem_id, problem, condition, option_order, index, base_seed, repeats, task)
                    )
    return samples

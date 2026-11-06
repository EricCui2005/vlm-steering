"""The prompt wording from the protocol, in one place.

The image itself is not part of this text. The backend passes it through the
model's chat template, which places it at the start of the user turn.
"""

RECURSION_FIRST = "recursion_first"
ITERATION_FIRST = "iteration_first"

_STRATEGIES_BY_ORDER = {
    RECURSION_FIRST: ("recursion", "iteration"),
    ITERATION_FIRST: ("iteration", "recursion"),
}
# The one-word answers the strategy-choice prompt asks for, in the same order.
_ANSWERS_BY_ORDER = {
    RECURSION_FIRST: ("recursive", "iterative"),
    ITERATION_FIRST: ("iterative", "recursive"),
}

_BASELINE_TEMPLATE = (
    "Solve the following programming problem in Python. Choose a coding strategy "
    "({first} or {second}) according to your personal preference and solve the problem. "
    "Write a complete function. Do not include explanation, just the code.\n"
    "Problem: {problem}"
)

_IMAGE_TEMPLATE = (
    "Solve the following programming problem in Python. Looking at the provided image, "
    "choose a coding strategy ({first} or {second}) and solve the problem. "
    "Write a complete function. Do not include explanation, just the code.\n"
    "Problem: {problem}"
)

# The image prompt with the same opening, but the model only names its strategy
# instead of writing the solution.
_CHOOSE_STRATEGY_TEMPLATE = (
    "Consider the following programming problem in Python. Looking at the provided image, "
    "choose a coding strategy ({first} or {second}) that you would use to solve the problem. "
    "Do not write any code. Answer with exactly one word: {first_answer} or {second_answer}.\n"
    "Problem: {problem}"
)

_TEXT_INSTRUCTION_TEMPLATE = (
    "{repeated}\n"
    "Solve the following programming problem in Python. Solve it using a {strategy} approach. "
    "Write a complete function. Do not include explanation, just the code.\n"
    "Problem: {problem}"
)


def text_instruction_prompt(problem: str, strategy: str, repeats: int) -> str:
    """The strategy word repeated in place of the image, then a direct instruction."""
    repeated = " ".join([strategy] * repeats)
    return _TEXT_INSTRUCTION_TEMPLATE.format(repeated=repeated, strategy=strategy, problem=problem)


def caption_prompt(problem: str, option_order: str, caption: str, repeats: int) -> str:
    """The image prompt, with a repeated caption where the image would be."""
    repeated = " ".join([caption] * repeats)
    return f"{repeated}\n{image_prompt(problem, option_order)}"


def baseline_prompt(problem: str, option_order: str) -> str:
    first, second = _STRATEGIES_BY_ORDER[option_order]
    return _BASELINE_TEMPLATE.format(first=first, second=second, problem=problem)


def image_prompt(problem: str, option_order: str) -> str:
    first, second = _STRATEGIES_BY_ORDER[option_order]
    return _IMAGE_TEMPLATE.format(first=first, second=second, problem=problem)


def choose_strategy_prompt(problem: str, option_order: str) -> str:
    first, second = _STRATEGIES_BY_ORDER[option_order]
    first_answer, second_answer = _ANSWERS_BY_ORDER[option_order]
    return _CHOOSE_STRATEGY_TEMPLATE.format(
        first=first, second=second, first_answer=first_answer, second_answer=second_answer, problem=problem
    )

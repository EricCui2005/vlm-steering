"""Label a model's solution as recursive, iterative, or other, by reading its code.

The approach:
1. Pull the Python code out of the model's response and parse it.
2. Build a call graph: which function calls which.
3. A function that can reach itself in that graph is recursive.
"""

import ast
import re

RECURSIVE = "recursive"
ITERATIVE = "iterative"
OTHER = "other"

_CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)(?:```|\Z)", re.DOTALL)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def extract_code(response: str) -> str:
    """Return the code from a response, preferring the block that defines a function."""
    blocks = _CODE_BLOCK.findall(response)
    if not blocks:
        return response.strip()
    for block in blocks:
        if re.search(r"^\s*(async\s+)?def\s", block, re.MULTILINE):
            return block.strip()
    return blocks[0].strip()


def classify(code: str) -> str:
    """Return RECURSIVE, ITERATIVE, or OTHER for a piece of Python code."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return OTHER

    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, _FUNCTIONS)
    }
    if not functions:
        return OTHER

    calls = {name: _called_functions(node, functions) for name, node in functions.items()}

    solutions = _entry_points(calls)
    recursive = [name for name in solutions if _reaches_recursion(name, calls)]
    iterative = [
        name
        for name in solutions
        if name not in recursive and _reaches_loop(name, calls, functions)
    ]

    if recursive and iterative:
        return OTHER  # The response wrote out both versions.
    if recursive:
        return RECURSIVE
    if iterative:
        return ITERATIVE
    return OTHER  # For example, a closed-form formula with no loop or self-call.


def _called_functions(function: ast.AST, known_functions: dict) -> set[str]:
    """Names of the solution's own functions that this function calls."""
    called = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            name = _call_target(node)
            if name in known_functions:
                called.add(name)
    return called


def _call_target(call: ast.Call) -> str | None:
    """The function name for `f(...)`, `self.f(...)`, `cls.f(...)`, or `Solution.f(...)`.

    Calls on other objects are ignored: `stack.pop()` does not call a function
    named `pop` that the solution happens to define.
    """
    target = call.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        receiver = target.value.id
        if receiver in ("self", "cls") or receiver[0].isupper():
            return target.attr
    return None


def _entry_points(calls: dict[str, set[str]]) -> list[str]:
    """Functions that no other function calls. These are the actual solutions."""
    called_by_another = {
        callee
        for caller, callees in calls.items()
        for callee in callees
        if callee != caller
    }
    entry_points = [name for name in calls if name not in called_by_another]
    # If functions only call each other (mutual recursion), every one is a candidate.
    return entry_points or list(calls)


def _reachable_from(start: str, calls: dict[str, set[str]]) -> set[str]:
    """Every function that `start` can end up calling."""
    reached = set()
    to_visit = list(calls[start])
    while to_visit:
        name = to_visit.pop()
        if name not in reached:
            reached.add(name)
            to_visit.extend(calls[name])
    return reached


def _reaches_recursion(start: str, calls: dict[str, set[str]]) -> bool:
    """True if `start`, or any function it calls, can call itself."""
    candidates = {start} | _reachable_from(start, calls)
    return any(name in _reachable_from(name, calls) for name in candidates)


def _reaches_loop(start: str, calls: dict[str, set[str]], functions: dict) -> bool:
    candidates = {start} | _reachable_from(start, calls)
    return any(
        isinstance(node, _LOOPS)
        for name in candidates
        for node in ast.walk(functions[name])
    )


def classify_choice(answer: str) -> str:
    """Label a one-word strategy answer. Anything other than exactly one of the two words is OTHER."""
    words = set(re.findall(r"[a-z]+", answer.lower()))
    said_recursive = bool(words & {"recursive", "recursion"})
    said_iterative = bool(words & {"iterative", "iteration"})
    if said_recursive and not said_iterative:
        return RECURSIVE
    if said_iterative and not said_recursive:
        return ITERATIVE
    return OTHER

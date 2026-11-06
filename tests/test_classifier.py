import pytest

from steering.classifier import ITERATIVE, OTHER, RECURSIVE, classify, classify_choice, extract_code

DIRECT_RECURSION = """
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
"""

MEMOIZED_RECURSION = """
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
"""

NESTED_HELPER = """
def inorder(root):
    result = []
    def visit(node):
        if node:
            visit(node.left)
            result.append(node.val)
            visit(node.right)
    visit(root)
    return result
"""

METHOD_RECURSION = """
class Solution:
    def inorder(self, root):
        if not root:
            return []
        return self.inorder(root.left) + [root.val] + self.inorder(root.right)
"""

GENERATOR_RECURSION = """
def inorder(root):
    if root:
        yield from inorder(root.left)
        yield root.val
        yield from inorder(root.right)
"""

MUTUAL_RECURSION = """
def is_even(n):
    return True if n == 0 else is_odd(n - 1)

def is_odd(n):
    return False if n == 0 else is_even(n - 1)
"""

RECURSION_WITH_LOOP = """
def dfs(graph, start, visited=None):
    visited = visited or []
    visited.append(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
"""

LOOP = """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
"""

EXPLICIT_STACK = """
def inorder(root):
    result, stack = [], []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        result.append(root.val)
        root = root.right
    return result
"""

MORRIS_TRAVERSAL = """
def inorder(root):
    result = []
    while root:
        if root.left:
            node = root.left
            while node.right and node.right is not root:
                node = node.right
            if node.right is None:
                node.right = root
                root = root.left
            else:
                node.right = None
                result.append(root.val)
                root = root.right
        else:
            result.append(root.val)
            root = root.right
    return result
"""

LOOP_WITH_UNRELATED_POP_METHOD = """
def pop(items):
    return items[-1]

def solve(stack):
    while stack:
        stack.pop()
"""

BOTH_VERSIONS = """
def fib_recursive(n):
    return n if n < 2 else fib_recursive(n - 1) + fib_recursive(n - 2)

def fib_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
"""

CLOSED_FORM = """
def fib(n):
    return round(1.618033988749895 ** n / 5 ** 0.5)
"""


@pytest.mark.parametrize(
    "code, expected",
    [
        (DIRECT_RECURSION, RECURSIVE),
        (MEMOIZED_RECURSION, RECURSIVE),
        (NESTED_HELPER, RECURSIVE),
        (METHOD_RECURSION, RECURSIVE),
        (GENERATOR_RECURSION, RECURSIVE),
        (MUTUAL_RECURSION, RECURSIVE),
        (RECURSION_WITH_LOOP, RECURSIVE),
        (LOOP, ITERATIVE),
        (EXPLICIT_STACK, ITERATIVE),
        (MORRIS_TRAVERSAL, ITERATIVE),
        (LOOP_WITH_UNRELATED_POP_METHOD, ITERATIVE),
        (BOTH_VERSIONS, OTHER),
        (CLOSED_FORM, OTHER),
        ("", OTHER),
        ("def broken(:", OTHER),
        ("for i in range(3):\n    print(i)", OTHER),
    ],
)
def test_classify(code, expected):
    assert classify(code) == expected


def test_extract_code_removes_fences():
    assert extract_code("```python\nx = 1\n```") == "x = 1"


def test_extract_code_prefers_block_that_defines_a_function():
    response = "```python\nfrom math import sqrt\n```\ntext\n```python\ndef f():\n    pass\n```"
    assert extract_code(response) == "def f():\n    pass"


def test_extract_code_handles_a_fence_cut_off_at_the_token_limit():
    assert extract_code("```python\ndef f():\n    pass") == "def f():\n    pass"


def test_extract_code_accepts_text_without_fences():
    assert extract_code("def f():\n    pass\n") == "def f():\n    pass"


@pytest.mark.parametrize(
    "answer, label",
    [
        ("iterative", ITERATIVE),
        ("Recursive.", RECURSIVE),
        ("**Iterative**", ITERATIVE),
        ("I would use recursion", RECURSIVE),
        ("recursive or iterative", OTHER),
        ("Hmm", OTHER),
    ],
)
def test_classify_choice(answer, label):
    assert classify_choice(answer) == label

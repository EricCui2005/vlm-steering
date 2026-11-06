"""Small statistics helpers: confidence intervals, Fisher's exact test, and FDR control."""

import math


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the table [[a, b], [c, d]].

    Sums the probability of every table with the same margins that is no more
    likely than the observed one.
    """
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2

    def probability(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(total, col1)

    observed = probability(a)
    lowest, highest = max(0, col1 - row2), min(row1, col1)
    p_value = sum(
        probability(x)
        for x in range(lowest, highest + 1)
        if probability(x) <= observed * (1 + 1e-9)
    )
    return min(1.0, p_value)


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Which hypotheses are rejected while controlling the false discovery rate at q."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    largest_passing_rank = 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= q * rank / m:
            largest_passing_rank = rank
    rejected = set(order[:largest_passing_rank])
    return [i in rejected for i in range(m)]

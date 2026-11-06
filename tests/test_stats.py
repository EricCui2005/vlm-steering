import pytest

from steering.stats import benjamini_hochberg, fisher_exact_two_sided, wilson_interval


def test_wilson_interval_for_zero_of_eight():
    low, high = wilson_interval(0, 8)
    assert low == 0
    assert high == pytest.approx(0.324, abs=0.001)


def test_fisher_matches_known_value():
    # scipy.stats.fisher_exact([[8, 2], [1, 5]]) gives p = 0.0350
    assert fisher_exact_two_sided(8, 2, 1, 5) == pytest.approx(0.0350, abs=0.0001)


def test_fisher_with_no_difference_is_one():
    assert fisher_exact_two_sided(5, 5, 5, 5) == pytest.approx(1.0)


def test_benjamini_hochberg():
    assert benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.6], q=0.05) == [
        True, True, False, False, False,
    ]

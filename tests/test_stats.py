"""wilson_interval 邊界與已知值測試(Phase 5,docs/PLAN2.md)。"""

from __future__ import annotations

import pytest

from fallguard.stats import wilson_interval


def test_wilson_interval_zero_successes():
    lo, hi = wilson_interval(0, 6)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.390, abs=0.01)


def test_wilson_interval_all_successes():
    lo, hi = wilson_interval(6, 6)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert lo == pytest.approx(0.610, abs=0.01)


def test_wilson_interval_known_value():
    # 5/6 ≈ 0.833,手算過的已知值(見 docs/PLAN2.md Phase 5)
    lo, hi = wilson_interval(5, 6)
    assert lo == pytest.approx(0.436, abs=0.01)
    assert hi == pytest.approx(0.970, abs=0.01)


def test_wilson_interval_zero_trials():
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize("successes,n", [(0, 1), (1, 1), (3, 10), (10, 10)])
def test_wilson_interval_bounds_always_valid(successes, n):
    lo, hi = wilson_interval(successes, n)
    assert 0.0 <= lo <= hi <= 1.0

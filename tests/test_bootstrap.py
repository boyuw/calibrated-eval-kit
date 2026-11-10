"""Tests for the percentile-bootstrap confidence intervals.

The anchors are exact where the method is exact:

* with ``n_boot=1`` the interval is the single resampled statistic, so a
  hand-rolled ``random.Random(seed).choices`` replicate must match;
* a degenerate item set (no variation at all) collapses the interval to
  the point estimate — every resample is the same set;
* intervals are deterministic for a fixed seed and move with the seed;
* the interval contains the point estimate and narrows as the item set
  grows, on the bundled items and on a repeated copy of them.
"""

import random
from pathlib import Path

import pytest

from evalkit.bootstrap import bootstrap_ci, ece_ci, kappa_ci
from evalkit.calibration import agreement_rate
from evalkit.data import Item, load_items

EXAMPLES = Path(__file__).parent.parent / "evalkit" / "examples" / "items.jsonl"


def _item(i: int, human: str, judge: str, conf: float) -> Item:
    return Item(f"t{i}", "prompt", None, human, judge, conf)


@pytest.fixture(scope="module")
def bundled():
    return load_items(EXAMPLES)


def test_single_replicate_matches_manual_resample(bundled):
    """n_boot=1: the interval is one resample, reproducible by hand."""
    rng = random.Random(42)
    resample = rng.choices(bundled, k=len(bundled))
    interval = bootstrap_ci(agreement_rate, bundled, n_boot=1, seed=42)
    expected = agreement_rate(resample)
    assert interval.lower == pytest.approx(expected)
    assert interval.upper == pytest.approx(expected)
    assert interval.estimate == pytest.approx(agreement_rate(bundled))


def test_deterministic_for_fixed_seed(bundled):
    first = kappa_ci(bundled, n_boot=500, seed=0)
    second = kappa_ci(bundled, n_boot=500, seed=0)
    assert first == second


def test_different_seeds_move_the_interval(bundled):
    a = kappa_ci(bundled, n_boot=1000, seed=0)
    b = kappa_ci(bundled, n_boot=1000, seed=1)
    assert (a.lower, a.upper) != (b.lower, b.upper)


def test_interval_contains_estimate_and_stays_in_range(bundled):
    kappa = kappa_ci(bundled, n_boot=1000, seed=0)
    assert -1.0 <= kappa.lower <= kappa.estimate <= kappa.upper <= 1.0
    ece = ece_ci(bundled, n_boot=1000, seed=0)
    assert 0.0 <= ece.lower <= ece.estimate <= ece.upper <= 1.0


def test_bundled_intervals_are_wide(bundled):
    """24 items cannot pin kappa down: the 95% interval spans ~0.8.

    This is the point of the module — the numbers below are the actual
    seed-0 values, recorded so a silent change in the resampling shows
    up as a test failure rather than a shrug.
    """
    kappa = kappa_ci(bundled, n_boot=1000, seed=0)
    assert kappa.estimate == pytest.approx(0.364, abs=0.001)
    assert kappa.lower == pytest.approx(-0.059, abs=0.01)
    assert kappa.upper == pytest.approx(0.731, abs=0.01)
    ece = ece_ci(bundled, n_boot=1000, seed=0)
    assert ece.estimate == pytest.approx(0.161, abs=0.001)
    assert ece.lower == pytest.approx(0.054, abs=0.01)
    assert ece.upper == pytest.approx(0.353, abs=0.01)


def test_degenerate_set_collapses_the_interval():
    items = [_item(i, "correct", "correct", 0.9) for i in range(10)]
    kappa = kappa_ci(items, n_boot=100, seed=0)
    assert kappa.estimate == 1.0
    assert kappa.lower == 1.0 and kappa.upper == 1.0
    ece = ece_ci(items, n_boot=100, seed=0)
    assert ece.estimate == pytest.approx(0.1)
    assert ece.lower == pytest.approx(0.1) and ece.upper == pytest.approx(0.1)


def test_interval_narrows_with_more_items(bundled):
    """Ten copies of the same mix: same estimate, much tighter interval."""
    small = kappa_ci(bundled, n_boot=500, seed=0)
    big = kappa_ci(bundled * 10, n_boot=500, seed=0)
    assert big.estimate == pytest.approx(small.estimate, abs=0.01)
    assert (big.upper - big.lower) < 0.5 * (small.upper - small.lower)


def test_rejects_bad_inputs(bundled):
    with pytest.raises(ValueError):
        bootstrap_ci(agreement_rate, [], n_boot=100)
    with pytest.raises(ValueError):
        bootstrap_ci(agreement_rate, bundled, n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_ci(agreement_rate, bundled, alpha=0.0)
    with pytest.raises(ValueError):
        bootstrap_ci(agreement_rate, bundled, alpha=1.0)
    with pytest.raises(ValueError):
        kappa_ci([])
    with pytest.raises(ValueError):
        ece_ci(bundled, n_bins=0)

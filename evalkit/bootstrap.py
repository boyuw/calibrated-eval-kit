"""Bootstrap confidence intervals for the calibration metrics.

Point estimates on a few dozen items are conversation starters, not
results: on 24 items a kappa of 0.36 could easily be 0.1 or 0.6. This
module wraps any item-level statistic in a percentile bootstrap — resample
the items with replacement, recompute the statistic on each resample, and
read the interval off the quantiles of the resampled statistics.

Method: percentile bootstrap, seeded, standard library only. The
resampling uses one ``random.Random(seed)`` and ``rng.choices``, so the
same seed always gives the same interval. Percentile intervals are the
simplest honest choice at this scale; they are not bias-corrected (no
BCa) and they assume the resampled statistics are a fair picture of the
sampling distribution, which needs a non-tiny item set to be believable.
Degenerate resamples are handled by the statistics themselves — e.g.
Cohen's kappa returns 1.0/0.0 when a resample collapses to one label —
so a degenerate item set yields a collapsed interval, which is the
correct answer for a set with no variation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .calibration import cohens_kappa, expected_calibration_error
from .data import Item

Statistic = Callable[[list[Item]], float]


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile-bootstrap interval for one statistic."""

    estimate: float  # the statistic on the full item set
    lower: float
    upper: float
    n_boot: int
    alpha: float  # e.g. 0.05 for a 95% interval
    method: str = "percentile"


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted list."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap_ci(
    statistic: Statistic,
    items: list[Item],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile-bootstrap interval for ``statistic`` on ``items``.

    ``statistic`` is any function from an item list to a float (kappa,
    ECE, agreement, ...). Each bootstrap replicate draws ``len(items)``
    items with replacement from ``items`` and recomputes the statistic;
    the interval is the ``[alpha/2, 1 - alpha/2]`` quantiles of the
    replicates. Deterministic for a fixed ``seed``.
    """
    if not items:
        raise ValueError("at least one item is required")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rng = random.Random(seed)
    n = len(items)
    estimate = statistic(items)
    replicates = [statistic(rng.choices(items, k=n)) for _ in range(n_boot)]
    replicates.sort()
    return BootstrapInterval(
        estimate=estimate,
        lower=_percentile(replicates, alpha / 2.0),
        upper=_percentile(replicates, 1.0 - alpha / 2.0),
        n_boot=n_boot,
        alpha=alpha,
    )


def kappa_ci(
    items: list[Item],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap interval for Cohen's kappa on ``items``."""
    return bootstrap_ci(cohens_kappa, items, n_boot, alpha, seed)


def ece_ci(
    items: list[Item],
    n_bins: int = 5,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap interval for ECE (``n_bins`` reliability bins) on ``items``."""
    return bootstrap_ci(
        lambda it: expected_calibration_error(it, n_bins), items, n_boot, alpha, seed
    )

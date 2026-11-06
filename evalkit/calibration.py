"""Judge calibration against human labels.

A judge that says "90% confident" should be right about 90% of the time.
These functions measure how far a judge is from that ideal: raw agreement,
chance-corrected agreement (Cohen's kappa), reliability binning with an
expected calibration error (ECE) summary, and the Brier score — the mean
squared error of the stated confidence against correctness.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .data import Item


def _require(items: list[Item]) -> None:
    if not items:
        raise ValueError("at least one item is required")


def agreement_rate(items: list[Item]) -> float:
    """Fraction of items where judge_label == human_label."""
    _require(items)
    return sum(it.judge_correct for it in items) / len(items)


def cohens_kappa(items: list[Item]) -> float:
    """Cohen's kappa between judge and human labels.

    kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and
    p_e is the agreement expected from each side's label marginals.
    """
    _require(items)
    n = len(items)
    p_o = agreement_rate(items)
    human = Counter(it.human_label for it in items)
    judge = Counter(it.judge_label for it in items)
    p_e = sum((human[label] / n) * (judge[label] / n) for label in set(human) | set(judge))
    if p_e == 1.0:  # both sides use a single label; kappa is degenerate
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


@dataclass(frozen=True)
class Bin:
    """One reliability bin over judge confidence."""

    lower: float
    upper: float
    count: int
    mean_confidence: float | None  # None when the bin is empty
    accuracy: float | None  # fraction correct in bin; None when empty


def reliability_bins(items: list[Item], n_bins: int = 5) -> list[Bin]:
    """Group items into equal-width confidence bins over [0, 1].

    The top bin is inclusive of confidence exactly 1.0.
    """
    _require(items)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    buckets: list[list[Item]] = [[] for _ in range(n_bins)]
    for it in items:
        idx = min(int(it.judge_confidence * n_bins), n_bins - 1)
        buckets[idx].append(it)
    bins: list[Bin] = []
    for i, bucket in enumerate(buckets):
        if bucket:
            mean_conf = sum(it.judge_confidence for it in bucket) / len(bucket)
            accuracy = sum(it.judge_correct for it in bucket) / len(bucket)
        else:
            mean_conf = accuracy = None
        bins.append(Bin(i / n_bins, (i + 1) / n_bins, len(bucket), mean_conf, accuracy))
    return bins


def expected_calibration_error(items: list[Item], n_bins: int = 5) -> float:
    """ECE: count-weighted mean |accuracy - mean confidence| across bins."""
    _require(items)
    n = len(items)
    return sum(
        (b.count / n) * abs(b.accuracy - b.mean_confidence)
        for b in reliability_bins(items, n_bins)
        if b.count
    )


def brier_score(items: list[Item]) -> float:
    """Brier score: mean squared error of confidence against correctness.

    (1/n) * sum((confidence_i - correct_i)^2), where correct_i is 1 when
    the judge's label matches the human label and 0 otherwise. A proper
    scoring rule: it is minimised in expectation only by stating the
    true probability of being right. Unlike ECE it needs no binning and
    it penalises overconfidence on individual items, not just on bin
    averages — but it mixes calibration with resolution, so read it
    alongside ECE, not instead of it. 0 is perfect, 1 is worst.
    """
    _require(items)
    return sum(
        (it.judge_confidence - (1.0 if it.judge_correct else 0.0)) ** 2
        for it in items
    ) / len(items)


@dataclass(frozen=True)
class CalibrationReport:
    """Summary of a judge's calibration on one item set."""

    n_items: int
    agreement: float
    kappa: float
    ece: float
    brier: float
    bins: list[Bin]


def calibration_report(items: list[Item], n_bins: int = 5) -> CalibrationReport:
    """Compute the full calibration summary for a set of items."""
    _require(items)
    return CalibrationReport(
        n_items=len(items),
        agreement=agreement_rate(items),
        kappa=cohens_kappa(items),
        ece=expected_calibration_error(items, n_bins),
        brier=brier_score(items),
        bins=reliability_bins(items, n_bins),
    )

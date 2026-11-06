"""Brier score on fixtures whose values are computable by hand.

The Brier score is the mean squared error of the stated confidence
against correctness (1 when the judge's label matches the human label,
0 otherwise). Tests check the perfect / worst endpoints, a hand-computed
mixed case, input validation, and that the full calibration report
carries the same value.
"""

import pytest

from evalkit.calibration import brier_score, calibration_report
from evalkit.data import Item


def make(human: str, judge: str, conf: float) -> Item:
    return Item(id="x", prompt="p", reference=None,
                human_label=human, judge_label=judge, judge_confidence=conf)


def test_brier_perfect_confident_judge_is_zero():
    items = [make("pass", "pass", 1.0), make("fail", "fail", 1.0)]
    assert brier_score(items) == pytest.approx(0.0)


def test_brier_perfect_hedged_judge():
    # Right every time but only 50% confident: (0.5 - 1)^2 = 0.25 each.
    items = [make("pass", "pass", 0.5), make("fail", "fail", 0.5)]
    assert brier_score(items) == pytest.approx(0.25)


def test_brier_always_wrong_and_confident_is_one():
    items = [make("pass", "fail", 1.0), make("fail", "pass", 1.0)]
    assert brier_score(items) == pytest.approx(1.0)


def test_brier_hand_computed_mixed_case():
    # (0.9 - 1)^2 = 0.01, (0.9 - 0)^2 = 0.81, (0.2 - 0)^2 = 0.04,
    # (0.2 - 1)^2 = 0.64; mean = 1.50 / 4 = 0.375.
    items = [
        make("pass", "pass", 0.9),
        make("pass", "fail", 0.9),
        make("fail", "fail", 0.2),
        make("fail", "pass", 0.2),
    ]
    assert brier_score(items) == pytest.approx(0.375)


def test_brier_requires_items():
    with pytest.raises(ValueError):
        brier_score([])


def test_brier_in_unit_interval_and_in_report():
    items = [
        make("pass", "pass", 0.9),
        make("pass", "fail", 0.9),
        make("fail", "fail", 0.2),
        make("fail", "pass", 0.2),
    ]
    score = brier_score(items)
    assert 0.0 <= score <= 1.0
    assert calibration_report(items).brier == pytest.approx(score)

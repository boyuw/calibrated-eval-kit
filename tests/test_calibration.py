"""Calibration metrics on fixtures whose values are computable by hand."""

import pytest

from evalkit.calibration import (
    agreement_rate,
    cohens_kappa,
    expected_calibration_error,
    reliability_bins,
)
from evalkit.data import Item


def make(human: str, judge: str, conf: float = 0.9) -> Item:
    return Item(id="x", prompt="p", reference=None,
                human_label=human, judge_label=judge, judge_confidence=conf)


# (human, judge) pairs: 3 of 4 agree.
# Marginals: human pass 2/4, fail 2/4; judge pass 3/4, fail 1/4.
# p_o = 0.75, p_e = 0.5*0.75 + 0.5*0.25 = 0.5, kappa = (0.75-0.5)/(1-0.5) = 0.5.
FIXTURE = [
    make("pass", "pass"),
    make("pass", "pass"),
    make("fail", "fail"),
    make("fail", "pass"),
]


def test_agreement_rate():
    assert agreement_rate(FIXTURE) == pytest.approx(0.75)


def test_cohens_kappa_hand_computed():
    assert cohens_kappa(FIXTURE) == pytest.approx(0.5)


def test_kappa_perfect_and_chance():
    perfect = [make("pass", "pass"), make("fail", "fail")]
    assert cohens_kappa(perfect) == pytest.approx(1.0)
    # Judge echoes the majority label: agreement is all chance-level here.
    chance = [make("pass", "pass"), make("fail", "pass")]
    assert cohens_kappa(chance) == pytest.approx(0.0)


def test_ece_single_bin():
    # Two items at confidence 0.9 in one bin, one right and one wrong:
    # accuracy 0.5, mean confidence 0.9, ECE = |0.5 - 0.9| = 0.4.
    items = [make("pass", "pass", 0.9), make("pass", "fail", 0.9)]
    assert expected_calibration_error(items, n_bins=1) == pytest.approx(0.4)


def test_reliability_bins_cover_all_items():
    items = [make("pass", "pass", c) for c in (0.05, 0.35, 0.55, 0.95, 1.0)]
    bins = reliability_bins(items, n_bins=5)
    assert sum(b.count for b in bins) == len(items)
    assert bins[-1].count == 2  # 0.95 and the inclusive 1.0 land in the top bin

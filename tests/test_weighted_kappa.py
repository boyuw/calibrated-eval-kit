"""Weighted kappa on a fixture whose values are computable by hand."""

import pytest

from evalkit.calibration import calibration_report, cohens_kappa, weighted_kappa
from evalkit.data import Item


def make(human: str, judge: str, conf: float = 0.9) -> Item:
    return Item(id="x", prompt="p", reference=None,
                human_label=human, judge_label=judge, judge_confidence=conf)


LABELS = ["low", "mid", "high"]

# 10 items, 8 exact agreements and 2 off-by-one misses:
# 4x (low, low), 2x (mid, mid), 2x (high, high), (low, mid), (mid, high).
# Human marginals: low .5, mid .3, high .2. Judge: low .4, mid .3, high .3.
# Diagonal expected agreement: .5*.4 + .3*.3 + .2*.3 = .35.
# Adjacent expected mass: .5*.3 + .3*.4 + .3*.3 + .2*.3 = .42; far misses
# carry weight 0 under both schemes.
# Linear (adjacent weight .5):   p_o_w = .90, p_e_w = .35 + .21 = .56,
#   kappa = .34 / .44 = 0.7727...
# Quadratic (adjacent weight .75): p_o_w = .95, p_e_w = .35 + .315 = .665,
#   kappa = .285 / .335 = 0.8507...
# Plain kappa on the same items: p_o = .8, p_e = .35, kappa = .45/.65.
FIXTURE = (
    [make("low", "low")] * 4
    + [make("mid", "mid")] * 2
    + [make("high", "high")] * 2
    + [make("low", "mid"), make("mid", "high")]
)


def test_weighted_kappa_linear_hand_computed():
    assert weighted_kappa(FIXTURE, LABELS, weights="linear") == pytest.approx(0.34 / 0.44)


def test_weighted_kappa_quadratic_hand_computed():
    assert weighted_kappa(FIXTURE, LABELS, weights="quadratic") == pytest.approx(0.285 / 0.335)


def test_weighted_kappa_rewards_near_misses():
    # Partial credit lifts kappa above the plain value, and quadratic —
    # which forgives a one-step miss more — lifts it above linear.
    plain = cohens_kappa(FIXTURE)
    linear = weighted_kappa(FIXTURE, LABELS, weights="linear")
    quadratic = weighted_kappa(FIXTURE, LABELS, weights="quadratic")
    assert plain == pytest.approx(0.45 / 0.65)
    assert plain < linear < quadratic


def test_weighted_kappa_perfect_and_degenerate():
    perfect = [make("low", "low"), make("mid", "mid"), make("high", "high")]
    assert weighted_kappa(perfect, LABELS) == pytest.approx(1.0)
    assert weighted_kappa(perfect, LABELS, weights="quadratic") == pytest.approx(1.0)
    # Both sides use a single label: degenerate, same rule as plain kappa.
    single_agree = [make("low", "low"), make("low", "low")]
    assert weighted_kappa(single_agree, LABELS) == pytest.approx(1.0)
    single_disagree = [make("low", "mid"), make("low", "mid")]
    assert weighted_kappa(single_disagree, LABELS) == pytest.approx(0.0)


def test_weighted_kappa_two_labels_is_plain_kappa():
    # With two labels the weights are 1 on the diagonal and 0 off it, so
    # weighted kappa collapses to plain kappa under either scheme.
    items = [make("low", "low"), make("low", "high"), make("high", "high")]
    labels = ["low", "high"]
    assert weighted_kappa(items, labels) == pytest.approx(cohens_kappa(items))
    assert weighted_kappa(items, labels, weights="quadratic") == pytest.approx(
        cohens_kappa(items)
    )


def test_weighted_kappa_rejects_bad_inputs():
    with pytest.raises(ValueError):
        weighted_kappa(FIXTURE, LABELS, weights="cubic")
    with pytest.raises(ValueError):
        weighted_kappa(FIXTURE, ["low", "low", "high"])
    with pytest.raises(ValueError):
        weighted_kappa(FIXTURE, ["low"])
    with pytest.raises(ValueError):
        weighted_kappa(FIXTURE, ["low", "mid"])  # "high" items not listed
    with pytest.raises(ValueError):
        weighted_kappa([], LABELS)


def test_calibration_report_carries_weighted_kappa():
    report = calibration_report(FIXTURE, labels=LABELS)
    assert report.weighted_kappa == pytest.approx(0.34 / 0.44)
    assert report.kappa == pytest.approx(0.45 / 0.65)
    # Without labels the field stays None; the rest of the report is intact.
    assert calibration_report(FIXTURE).weighted_kappa is None

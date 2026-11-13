"""Edge cases: empty bins, single-class labels, perfect/adversarial judges.

The metrics are defined on ordinary item sets, but the degenerate
corners are where a silent wrong number is easiest to ship:

- empty bins must stay in the table (count 0, metrics None) and must
  not contribute to ECE;
- single-class label sets make the chance agreement p_e = 1, so kappa
  is degenerate by construction — the kit returns 1.0 when the single
  label is always right and 0.0 when it is always wrong, never NaN;
- a perfect judge and an always-wrong judge pin the ends of every
  metric, and an always-wrong judge that *says* confidence 0.0 is
  perfectly calibrated (ECE/Brier 0) while still scoring kappa -1:
  calibration measures honesty about being right, not accuracy.
"""

import pytest

from evalkit.bootstrap import ece_ci, kappa_ci
from evalkit.calibration import (
    agreement_rate,
    brier_score,
    calibration_report,
    cohens_kappa,
    expected_calibration_error,
    reliability_bins,
)
from evalkit.data import Item
from evalkit.export import reliability_table
from evalkit.judges import MockJudge, judge_items


def make(human: str, judge: str, conf: float = 0.9) -> Item:
    return Item(id="x", prompt="p", reference=None,
                human_label=human, judge_label=judge, judge_confidence=conf)


# ---------------------------------------------------------------- empty bins

def test_empty_bins_stay_null_and_do_not_feed_ece():
    # All confidences 0.9 land in the top bin of 5; bins 0-3 are empty.
    items = [make("pass", "pass", 0.9), make("pass", "fail", 0.9)]
    bins = reliability_bins(items, n_bins=5)
    assert [b.count for b in bins] == [0, 0, 0, 0, 2]
    for b in bins[:4]:
        assert b.mean_confidence is None
        assert b.accuracy is None
    # ECE is computed over the one non-empty bin only: |0.5 - 0.9| = 0.4,
    # the same value a single-bin call gives.
    assert expected_calibration_error(items, n_bins=5) == pytest.approx(0.4)
    assert expected_calibration_error(items, n_bins=1) == pytest.approx(0.4)


def test_empty_bins_kept_in_exported_table():
    items = [make("pass", "pass", 0.9)]
    rows = reliability_table(items, n_bins=5)
    assert len(rows) == 5
    assert rows[0]["count"] == 0
    assert rows[0]["mean_confidence"] is None
    assert rows[4]["count"] == 1


# --------------------------------------------------------- single-class labels

def test_single_class_all_agree_kappa_is_one_not_nan():
    # Both sides use only "pass" and always agree: p_e = 1 by construction.
    items = [make("pass", "pass", 0.9) for _ in range(3)]
    assert cohens_kappa(items) == pytest.approx(1.0)
    assert agreement_rate(items) == pytest.approx(1.0)
    assert brier_score(items) == pytest.approx(0.01)  # (0.9 - 1)^2


def test_single_class_all_disagree_kappa_is_zero_not_nan():
    # Both sides use only "pass" and never agree: p_e = 1, p_o = 0.
    items = [make("pass", "fail", 0.9) for _ in range(3)]
    assert cohens_kappa(items) == pytest.approx(0.0)
    assert agreement_rate(items) == pytest.approx(0.0)
    assert brier_score(items) == pytest.approx(0.81)  # (0.9 - 0)^2
    assert expected_calibration_error(items, n_bins=1) == pytest.approx(0.9)


def test_single_class_human_mixed_judge_kappa_is_zero():
    # Human always "pass"; judge says "pass" half the time.
    # p_o = 0.5, p_e = 1.0 * 0.5 = 0.5, so kappa = 0: the 50%
    # agreement is exactly what the marginals predict by chance.
    items = [make("pass", "pass"), make("pass", "pass"),
             make("pass", "fail"), make("pass", "fail")]
    assert cohens_kappa(items) == pytest.approx(0.0)
    assert agreement_rate(items) == pytest.approx(0.5)


# ------------------------------------------------- perfect and adversarial judges

def test_perfect_judge_pins_every_metric():
    items = [make("pass", "pass", 1.0), make("fail", "fail", 1.0)]
    report = calibration_report(items, n_bins=1)
    assert report.agreement == pytest.approx(1.0)
    assert report.kappa == pytest.approx(1.0)
    assert report.ece == pytest.approx(0.0)
    assert report.brier == pytest.approx(0.0)


def test_adversarial_judge_confident_and_wrong():
    # Always the wrong label, stated with confidence 1.0.
    # kappa = (0 - 0.5) / (1 - 0.5) = -1; ECE and Brier both hit 1.
    items = [make("pass", "fail", 1.0), make("fail", "pass", 1.0)]
    report = calibration_report(items, n_bins=1)
    assert report.agreement == pytest.approx(0.0)
    assert report.kappa == pytest.approx(-1.0)
    assert report.ece == pytest.approx(1.0)
    assert report.brier == pytest.approx(1.0)


def test_adversarial_judge_honest_about_being_wrong():
    # Same always-wrong labels, but stated with confidence 0.0: the
    # judge is perfectly calibrated (it knows it is wrong) even though
    # its accuracy is 0. ECE and Brier reward the honesty; kappa does
    # not — it still reads -1.
    items = [make("pass", "fail", 0.0), make("fail", "pass", 0.0)]
    report = calibration_report(items, n_bins=1)
    assert report.agreement == pytest.approx(0.0)
    assert report.kappa == pytest.approx(-1.0)
    assert report.ece == pytest.approx(0.0)
    assert report.brier == pytest.approx(0.0)


def test_adversarial_mock_judge_end_to_end():
    # An adapter that always flips the human label, via judge_items.
    cases = [("a", "prompt a", None, "pass"), ("b", "prompt b", None, "fail")]
    judge = MockJudge(
        verdicts={"prompt a": ("fail", 1.0), "prompt b": ("pass", 1.0)}
    )
    report = calibration_report(judge_items(cases, judge), n_bins=1)
    assert report.agreement == pytest.approx(0.0)
    assert report.kappa == pytest.approx(-1.0)
    assert report.brier == pytest.approx(1.0)


# ------------------------------------------------------------- confidence bounds

def test_confidence_zero_and_one_land_in_outer_bins():
    items = [make("pass", "pass", 0.0), make("pass", "pass", 1.0)]
    bins = reliability_bins(items, n_bins=5)
    assert bins[0].count == 1  # 0.0 is the bottom bin's lower edge
    assert bins[-1].count == 1  # 1.0 is inclusive in the top bin
    assert bins[0].mean_confidence == pytest.approx(0.0)
    assert bins[-1].mean_confidence == pytest.approx(1.0)


# ------------------------------------------------------------- empty input / bootstrap

def test_empty_input_raises_everywhere():
    for fn in (agreement_rate, cohens_kappa, brier_score,
               expected_calibration_error, calibration_report,
               reliability_bins, reliability_table):
        with pytest.raises(ValueError):
            fn([])
    with pytest.raises(ValueError):
        kappa_ci([])
    with pytest.raises(ValueError):
        ece_ci([])


def test_bootstrap_on_degenerate_set_collapses():
    # Every resample of a single-class perfect set is the same set, so
    # the kappa interval collapses to the point [1, 1] — the correct
    # answer for a set with no variation, not an error.
    items = [make("pass", "pass", 0.9) for _ in range(4)]
    interval = kappa_ci(items, n_boot=20, seed=0)
    assert interval.estimate == pytest.approx(1.0)
    assert interval.lower == pytest.approx(1.0)
    assert interval.upper == pytest.approx(1.0)
    # Same for ECE on an always-wrong, fully confident judge: [1, 1].
    adversarial = [make("pass", "fail", 1.0), make("fail", "pass", 1.0)]
    interval = ece_ci(adversarial, n_bins=1, n_boot=20, seed=0)
    assert interval.estimate == pytest.approx(1.0)
    assert interval.lower == pytest.approx(1.0)
    assert interval.upper == pytest.approx(1.0)

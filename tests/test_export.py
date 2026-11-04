"""Reliability-diagram export: hand-computed rows, CSV, and JSON."""

import csv
import io
import json

import pytest

from evalkit.calibration import reliability_bins
from evalkit.data import Item
from evalkit.export import (
    reliability_table,
    reliability_table_csv,
    reliability_table_json,
    rows_from_bins,
)


def make(human: str, judge: str, conf: float) -> Item:
    return Item(id="x", prompt="p", reference=None,
                human_label=human, judge_label=judge, judge_confidence=conf)


# Two items in bin 0 of a 2-bin split: confidences 0.1 (right) and 0.3
# (wrong) -> mean confidence 0.2, accuracy 0.5, gap +0.3. Bin 1 is empty.
ITEMS = [make("pass", "pass", 0.1), make("pass", "fail", 0.3)]


def test_table_hand_computed():
    rows = reliability_table(ITEMS, n_bins=2)
    assert rows[0] == {
        "bin": 0, "lower": 0.0, "upper": 0.5, "count": 2,
        "mean_confidence": pytest.approx(0.2),
        "accuracy": pytest.approx(0.5),
        "gap": pytest.approx(0.3),
    }
    assert rows[1] == {
        "bin": 1, "lower": 0.5, "upper": 1.0, "count": 0,
        "mean_confidence": None, "accuracy": None, "gap": None,
    }


def test_rows_from_bins_matches_table():
    assert rows_from_bins(reliability_bins(ITEMS, n_bins=2)) == reliability_table(ITEMS, n_bins=2)


def test_csv_round_trips_and_keeps_empty_bin():
    parsed = list(csv.DictReader(io.StringIO(reliability_table_csv(ITEMS, n_bins=2))))
    assert [r["bin"] for r in parsed] == ["0", "1"]
    assert parsed[0]["count"] == "2"
    assert float(parsed[0]["mean_confidence"]) == pytest.approx(0.2)
    assert float(parsed[0]["gap"]) == pytest.approx(0.3)
    # Empty bin: count 0, blank metric fields (not a silently dropped row).
    assert parsed[1]["count"] == "0"
    assert parsed[1]["mean_confidence"] == ""
    assert parsed[1]["accuracy"] == ""
    assert parsed[1]["gap"] == ""


def test_json_round_trips_with_nulls():
    rows = json.loads(reliability_table_json(ITEMS, n_bins=2))
    assert rows[0]["count"] == 2
    assert rows[0]["gap"] == pytest.approx(0.3)
    assert rows[1]["count"] == 0
    assert rows[1]["mean_confidence"] is None
    assert rows[1]["accuracy"] is None
    assert rows[1]["gap"] is None


def test_export_rejects_empty_and_bad_bins():
    with pytest.raises(ValueError):
        reliability_table([], n_bins=5)
    with pytest.raises(ValueError):
        reliability_table(ITEMS, n_bins=0)

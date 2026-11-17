"""Per-slice calibration: slices match whole-set metrics, partition, edges."""

import pytest

from evalkit.calibration import calibration_report
from evalkit.data import Item, load_items
from evalkit.slices import UNTAGGED, per_slice_calibration


def make(id, human, judge, conf=0.9, tags=()):
    return Item(id=id, prompt="p", reference=None,
                human_label=human, judge_label=judge,
                judge_confidence=conf, tags=tags)


def test_single_slice_matches_whole_set():
    # One slice holding every item must reproduce the whole-set report.
    items = [
        make("1", "pass", "pass", 0.9, ("a",)),
        make("2", "pass", "pass", 0.8, ("a",)),
        make("3", "fail", "fail", 0.7, ("a",)),
        make("4", "fail", "pass", 0.6, ("a",)),
    ]
    (slice_report,) = per_slice_calibration(items, min_n=1)
    whole = calibration_report(items)
    assert slice_report.name == "a"
    assert slice_report.n_items == len(items)
    assert slice_report.agreement == pytest.approx(whole.agreement)
    assert slice_report.kappa == pytest.approx(whole.kappa)
    assert slice_report.ece == pytest.approx(whole.ece)
    assert slice_report.brier == pytest.approx(whole.brier)
    assert slice_report.reliable


def test_slices_partition_and_metrics_differ():
    # One well-calibrated slice, one badly calibrated slice, same
    # confidence: the split must surface what the aggregate hides.
    good = [make(f"g{i}", "pass", "pass", 0.9, ("good",)) for i in range(4)]
    bad = [make(f"b{i}", "pass", "fail", 0.9, ("bad",)) for i in range(4)]
    reports = per_slice_calibration(good + bad, min_n=1)
    assert [r.name for r in reports] == ["bad", "good"]  # sorted by name
    assert sum(r.n_items for r in reports) == 8
    by_name = {r.name: r for r in reports}
    assert by_name["good"].agreement == pytest.approx(1.0)
    assert by_name["good"].ece == pytest.approx(0.1)  # |1.0 - 0.9|
    assert by_name["good"].kappa == pytest.approx(1.0)
    assert by_name["good"].brier == pytest.approx(0.01)
    assert by_name["bad"].agreement == pytest.approx(0.0)
    assert by_name["bad"].ece == pytest.approx(0.9)  # |0.0 - 0.9|
    assert by_name["bad"].brier == pytest.approx(0.81)
    assert by_name["good"].ece < by_name["bad"].ece


def test_untagged_items_grouped_not_dropped():
    items = [
        make("1", "pass", "pass", 0.9, ("a",)),
        make("2", "pass", "pass", 0.9),  # no tags argument at all
        make("3", "pass", "fail", 0.9, ()),  # explicitly empty tags
    ]
    reports = per_slice_calibration(items, min_n=1)
    by_name = {r.name: r for r in reports}
    assert by_name[UNTAGGED].n_items == 2
    assert by_name["a"].n_items == 1
    assert sum(r.n_items for r in reports) == 3


def test_single_item_slice_reported_but_flagged():
    (report,) = per_slice_calibration(
        [make("1", "pass", "pass", 0.9, ("solo",))])
    assert report.n_items == 1
    assert report.agreement == pytest.approx(1.0)
    assert report.ece == pytest.approx(0.1)
    assert not report.reliable  # default min_n is 5
    # ... unless the caller says one item is enough.
    (report,) = per_slice_calibration(
        [make("1", "pass", "pass", 0.9, ("solo",))], min_n=1)
    assert report.reliable


def test_empty_input_raises():
    with pytest.raises(ValueError):
        per_slice_calibration([])


def test_bad_min_n_raises():
    with pytest.raises(ValueError):
        per_slice_calibration([make("1", "pass", "pass")], min_n=0)


def test_multi_tag_item_counted_in_each_slice():
    items = [
        make("1", "pass", "pass", 0.9, ("a", "b")),
        make("2", "pass", "pass", 0.9, ("a",)),
        make("3", "pass", "pass", 0.9, ("b",)),
    ]
    reports = per_slice_calibration(items, min_n=1)
    by_name = {r.name: r for r in reports}
    assert by_name["a"].n_items == 2
    assert by_name["b"].n_items == 2
    # Slice counts exceed the item count here by design: the shared item
    # belongs to both slices, so the slices overlap rather than partition.
    assert sum(r.n_items for r in reports) == 4


def test_custom_key_groups_by_anything():
    items = [
        make("1", "pass", "pass", 0.95),
        make("2", "pass", "fail", 0.95),
        make("3", "pass", "pass", 0.5),
    ]
    reports = per_slice_calibration(
        items, key=lambda it: "high" if it.judge_confidence >= 0.9 else "low",
        min_n=1)
    by_name = {r.name: r for r in reports}
    assert by_name["high"].n_items == 2
    assert by_name["low"].n_items == 1
    # A key returning None sends the item to "<untagged>".
    reports = per_slice_calibration(items, key=lambda it: None, min_n=1)
    assert len(reports) == 1
    assert reports[0].name == UNTAGGED
    assert reports[0].n_items == 3


def test_tags_load_from_jsonl(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(
        '{"id": "1", "prompt": "p", "human_label": "pass",'
        ' "judge_label": "pass", "judge_confidence": 0.9,'
        ' "tags": ["math", "hard"]}\n'
        '{"id": "2", "prompt": "p", "human_label": "pass",'
        ' "judge_label": "pass", "judge_confidence": 0.9}\n',
        encoding="utf-8",
    )
    tagged, untagged = load_items(path)
    assert tagged.tags == ("math", "hard")
    assert untagged.tags == ()


def test_malformed_tags_rejected(tmp_path):
    for bad_tags in ('"math"', '[1, 2]', '["math", 3]'):
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"id": "1", "prompt": "p", "human_label": "pass",'
            f' "judge_label": "pass", "judge_confidence": 0.9,'
            f' "tags": {bad_tags}}}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_items(path)

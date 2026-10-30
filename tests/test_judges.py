"""Judge adapters: the mock judge and the Item conversion."""

import pytest

from evalkit.calibration import agreement_rate
from evalkit.data import Item
from evalkit.judges import JudgeOutput, MockJudge, item_from_judge_output, judge_items


def test_judge_output_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        JudgeOutput(label="correct", confidence=1.5)
    with pytest.raises(ValueError):
        JudgeOutput(label="correct", confidence=-0.1)


def test_mock_judge_returns_canned_verdict():
    judge = MockJudge({"What is 2 + 2?": ("correct", 0.9)})
    out = judge.judge("What is 2 + 2?", "4")
    assert out.label == "correct"
    assert out.confidence == 0.9


def test_mock_judge_falls_back_instead_of_reusing_a_canned_answer():
    judge = MockJudge({"known?": ("correct", 0.9)},
                      default_label="incorrect", default_confidence=0.5)
    out = judge.judge("unseen?", None)
    assert (out.label, out.confidence) == ("incorrect", 0.5)


def test_item_from_judge_output():
    item = item_from_judge_output("c1", "p", "ref", "correct",
                                  JudgeOutput(label="correct", confidence=0.9))
    assert item == Item(id="c1", prompt="p", reference="ref",
                        human_label="correct", judge_label="correct",
                        judge_confidence=0.9)
    assert item.judge_correct


def test_judge_items_accepts_a_plain_function_adapter():
    def fn(prompt: str, reference: str | None) -> JudgeOutput:
        return JudgeOutput(label="correct", confidence=0.8)

    cases = [("a", "p1", None, "correct"), ("b", "p2", None, "incorrect")]
    items = judge_items(cases, fn)
    assert [it.id for it in items] == ["a", "b"]
    assert agreement_rate(items) == pytest.approx(0.5)


def test_judge_items_end_to_end_with_mock_judge():
    judge = MockJudge({
        "p1": ("correct", 0.9),
        "p2": ("incorrect", 0.9),
    })
    cases = [("a", "p1", None, "correct"), ("b", "p2", None, "incorrect"),
             ("c", "p3", None, "correct")]
    items = judge_items(cases, judge)
    # First two match their human labels; the unseen third falls back to
    # "incorrect" at 0.5, so it disagrees with its "correct" human label.
    assert agreement_rate(items) == pytest.approx(2 / 3)

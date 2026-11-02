"""Bundled example sets: both domains load, and each judge is imperfect.

The general-knowledge set (items.jsonl) and the code-review set
(code_review.jsonl) share the same Item schema; the second domain exists
so the calibration report can be exercised on judged outputs that are
not trivia answers.
"""

from pathlib import Path

from evalkit.data import load_items

EXAMPLES = Path(__file__).parent.parent / "evalkit" / "examples"


def test_code_review_examples_load():
    items = load_items(EXAMPLES / "code_review.jsonl")
    assert len(items) == 12
    assert all(0.0 <= it.judge_confidence <= 1.0 for it in items)
    assert all(it.id.startswith("cr-") for it in items)
    # The fixture judge is imperfect on purpose, in both directions:
    # overconfident on an error (cr-09) and unsure on a miss (cr-12).
    assert any(not it.judge_correct for it in items)
    assert any(it.judge_correct for it in items)


def test_example_sets_share_ids_space():
    general = load_items(EXAMPLES / "items.jsonl")
    review = load_items(EXAMPLES / "code_review.jsonl")
    ids = [it.id for it in general] + [it.id for it in review]
    assert len(ids) == len(set(ids))

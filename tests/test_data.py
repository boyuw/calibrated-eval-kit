"""JSONL loading: the bundled dataset parses and bad input is rejected."""

import pytest

from evalkit.data import load_items

from pathlib import Path

EXAMPLES = Path(__file__).parent.parent / "evalkit" / "examples" / "items.jsonl"


def test_bundled_examples_load():
    items = load_items(EXAMPLES)
    assert len(items) == 24
    assert all(0.0 <= it.judge_confidence <= 1.0 for it in items)
    # The fixture judge is imperfect on purpose.
    assert any(not it.judge_correct for it in items)


def test_missing_field_rejected(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "prompt": "p"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_items(bad)


def test_confidence_out_of_range_rejected(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"id": "x", "prompt": "p", "human_label": "correct",'
        ' "judge_label": "correct", "judge_confidence": 1.5}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_items(bad)

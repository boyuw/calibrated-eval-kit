"""Split determinism, disjointness, and the leakage guard."""

import pytest

from evalkit.data import Item
from evalkit.splits import LeakageError, check_no_leakage, split_items


def make_items(n: int = 40) -> list[Item]:
    return [
        Item(id=f"item-{i:02d}", prompt=f"Question number {i}?", reference=None,
             human_label="correct", judge_label="correct", judge_confidence=0.9)
        for i in range(n)
    ]


def test_split_is_deterministic():
    items = make_items()
    dev1, held1 = split_items(items, heldout_fraction=0.25, seed=7)
    dev2, held2 = split_items(items, heldout_fraction=0.25, seed=7)
    assert [it.id for it in dev1] == [it.id for it in dev2]
    assert [it.id for it in held1] == [it.id for it in held2]


def test_split_is_disjoint_and_complete():
    items = make_items()
    dev, held_out = split_items(items, heldout_fraction=0.25, seed=7)
    dev_ids = {it.id for it in dev}
    held_ids = {it.id for it in held_out}
    assert dev_ids.isdisjoint(held_ids)
    assert dev_ids | held_ids == {it.id for it in items}
    assert 0 < len(held_out) < len(items)
    check_no_leakage(dev, held_out)  # must not raise


def test_leakage_by_id_fires():
    items = make_items(4)
    with pytest.raises(LeakageError):
        check_no_leakage(items[:2], items[1:])


def test_leakage_by_text_fires():
    a = Item(id="a", prompt="What is 2 + 2?", reference=None,
             human_label="correct", judge_label="correct", judge_confidence=0.9)
    b = Item(id="b", prompt="  what IS 2+2?? ", reference=None,
             human_label="correct", judge_label="correct", judge_confidence=0.9)
    with pytest.raises(LeakageError):
        check_no_leakage([a], [b])

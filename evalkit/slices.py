"""Per-slice calibration: the same report, broken down by item group.

An aggregate ECE of 0.16 can hide a slice that is badly calibrated —
a judge that is honest on trivia and wildly overconfident on code review
looks fine until you split the set. These functions group items by a
key (by default the item's tags) and compute the full calibration
summary — agreement, Cohen's kappa, ECE, Brier score — for each slice.

Small slices are still reported, with their n, but flagged: a slice
below ``min_n`` items gets ``reliable=False`` so a kappa of 1.0 on two
items is never mistaken for evidence. The flag is a warning, not a
suppression — dropping small slices would hide exactly the groups
where the judge is most likely to be untested.

An item with several tags belongs to each of its tag slices, so slice
counts add up to the total only when the key assigns each item to
exactly one slice. Items with no tags (or a key that returns nothing
for them) land in the ``"<untagged>"`` slice rather than vanishing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .calibration import calibration_report
from .data import Item

UNTAGGED = "<untagged>"

#: A grouping key: given an item, return one slice name, several names,
#: or None when the item belongs to no slice (it lands in "<untagged>").
SliceKey = Callable[[Item], str | Iterable[str] | None]


def _slice_names(key: SliceKey, item: Item) -> list[str]:
    value = key(item)
    if value is None:
        names: list[str] = []
    elif isinstance(value, str):
        names = [value]
    else:
        names = list(value)
    # Empty names carry no information; duplicates would double-count.
    names = sorted({name for name in names if name})
    return names or [UNTAGGED]


@dataclass(frozen=True)
class SliceReport:
    """Calibration summary for one slice of the item set."""

    name: str
    n_items: int
    agreement: float
    kappa: float
    ece: float
    brier: float
    reliable: bool  # True when n_items >= min_n; see module docstring


def per_slice_calibration(
    items: list[Item],
    key: SliceKey | None = None,
    n_bins: int = 5,
    min_n: int = 5,
) -> list[SliceReport]:
    """Compute the calibration summary for each slice of ``items``.

    ``key`` groups items into slices; the default groups by item tags,
    so an item tagged ("math", "hard") is counted in both the "math"
    and the "hard" slices. Pass a custom key for any other grouping —
    e.g. ``lambda it: "high" if it.judge_confidence >= 0.9 else "low"``,
    or a source field via ``lambda it: it.tags[0]``. Slices are returned
    sorted by name. Raises ValueError on an empty item set, as the
    whole-set metrics do, and on ``min_n < 1``.
    """
    if not items:
        raise ValueError("at least one item is required")
    if min_n < 1:
        raise ValueError("min_n must be >= 1")
    group: SliceKey = key if key is not None else (lambda it: it.tags)
    grouped: dict[str, list[Item]] = {}
    for it in items:
        for name in _slice_names(group, it):
            grouped.setdefault(name, []).append(it)
    reports: list[SliceReport] = []
    for name in sorted(grouped):
        subset = grouped[name]
        report = calibration_report(subset, n_bins)
        reports.append(
            SliceReport(
                name=name,
                n_items=len(subset),
                agreement=report.agreement,
                kappa=report.kappa,
                ece=report.ece,
                brier=report.brier,
                reliable=len(subset) >= min_n,
            )
        )
    return reports

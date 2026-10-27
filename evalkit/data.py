"""Loading evaluation items from JSONL.

An evaluation item is one judged output: a prompt, an optional reference,
a human (gold) label, and a judge label with a confidence in [0, 1].
Labels are categorical strings, e.g. "correct" / "incorrect".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Item:
    """One judged output."""

    id: str
    prompt: str
    reference: str | None
    human_label: str
    judge_label: str
    judge_confidence: float

    @property
    def judge_correct(self) -> bool:
        """True when the judge's label matches the human label."""
        return self.judge_label == self.human_label


_REQUIRED = ("id", "prompt", "human_label", "judge_label", "judge_confidence")


def _parse_item(raw: dict, lineno: int) -> Item:
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"line {lineno}: missing field(s): {', '.join(missing)}")
    confidence = raw["judge_confidence"]
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"line {lineno}: judge_confidence must be in [0, 1]")
    return Item(
        id=str(raw["id"]),
        prompt=str(raw["prompt"]),
        reference=None if raw.get("reference") is None else str(raw["reference"]),
        human_label=str(raw["human_label"]),
        judge_label=str(raw["judge_label"]),
        judge_confidence=float(confidence),
    )


def load_items(path: str | Path) -> list[Item]:
    """Load items from a JSONL file (one JSON object per line)."""
    items: list[Item] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            items.append(_parse_item(json.loads(line), lineno))
    if not items:
        raise ValueError(f"no items found in {path}")
    return items

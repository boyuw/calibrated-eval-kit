"""Deterministic dev / held-out splits with a leakage guard.

If you tune prompts, thresholds, or judges on the same items you later
report scores on, the scores are optimistic. This module splits items
into a dev set (tune freely) and a held-out set (touch only for final
numbers), and can verify that two sets do not overlap — by id or by
near-identical prompt text.
"""

from __future__ import annotations

import hashlib
import re

from .data import Item


class LeakageError(ValueError):
    """Raised when dev and held-out sets share an item or a prompt."""


def _bucket(item_id: str, seed: int) -> int:
    """Stable bucket in [0, 100) for an item id.

    Uses SHA-256 rather than built-in hash(), whose result for strings is
    randomized per process (PYTHONHASHSEED) and would break determinism.
    """
    digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def split_items(
    items: list[Item], heldout_fraction: float = 0.25, seed: int = 0
) -> tuple[list[Item], list[Item]]:
    """Split items into (dev, held_out), deterministically by item id."""
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    threshold = round(heldout_fraction * 100)
    dev = [it for it in items if _bucket(it.id, seed) >= threshold]
    held_out = [it for it in items if _bucket(it.id, seed) < threshold]
    return dev, held_out


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace/punctuation for duplicate checks."""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _fingerprint(item: Item) -> str:
    return hashlib.sha256(normalize_text(item.prompt).encode("utf-8")).hexdigest()


def check_no_leakage(dev: list[Item], held_out: list[Item]) -> None:
    """Raise LeakageError if the sets share an id or a normalized prompt."""
    shared_ids = {it.id for it in dev} & {it.id for it in held_out}
    if shared_ids:
        raise LeakageError(f"item id(s) in both sets: {sorted(shared_ids)}")
    dev_texts = {_fingerprint(it) for it in dev}
    leaked = [it.id for it in held_out if _fingerprint(it) in dev_texts]
    if leaked:
        raise LeakageError(f"prompt text appears in both sets, held-out id(s): {leaked}")

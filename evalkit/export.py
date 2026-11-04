"""Reliability-diagram data export.

The demo prints a per-bin calibration table for humans; these functions
produce the same table as machine-readable rows, CSV, or JSON, so it can
be embedded in a README, a report, or a plotting script. Standard library
only, matching the rest of the kit.

Empty bins are kept, not dropped: a bin with no items has count 0 and
null (JSON) / blank (CSV) mean confidence, accuracy, and gap, so a judge
that never uses low confidences stays visible in the exported table.
The gap column is accuracy minus mean confidence: negative means the
judge was overconfident in that bin, positive means underconfident.
"""

from __future__ import annotations

import csv
import io
import json

from .calibration import Bin, reliability_bins
from .data import Item

_COLUMNS = ("bin", "lower", "upper", "count", "mean_confidence", "accuracy", "gap")


def rows_from_bins(bins: list[Bin]) -> list[dict]:
    """Convert reliability bins to plain rows (one dict per bin)."""
    rows: list[dict] = []
    for i, b in enumerate(bins):
        gap = None
        if b.count:
            gap = b.accuracy - b.mean_confidence
        rows.append({
            "bin": i,
            "lower": b.lower,
            "upper": b.upper,
            "count": b.count,
            "mean_confidence": b.mean_confidence,
            "accuracy": b.accuracy,
            "gap": gap,
        })
    return rows


def reliability_table(items: list[Item], n_bins: int = 5) -> list[dict]:
    """Per-bin table (mean confidence, accuracy, count, gap) as rows."""
    return rows_from_bins(reliability_bins(items, n_bins))


def reliability_table_csv(items: list[Item], n_bins: int = 5) -> str:
    """Per-bin table as CSV text (empty fields for empty bins)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_COLUMNS)
    writer.writeheader()
    for row in reliability_table(items, n_bins):
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return buffer.getvalue()


def reliability_table_json(items: list[Item], n_bins: int = 5) -> str:
    """Per-bin table as JSON text (nulls for empty bins)."""
    return json.dumps(reliability_table(items, n_bins), indent=2)

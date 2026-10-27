"""calibrated-eval-kit: minimal tools for calibrated LLM/agent evaluation."""

from .calibration import (
    CalibrationReport,
    agreement_rate,
    calibration_report,
    cohens_kappa,
    expected_calibration_error,
    reliability_bins,
)
from .data import Item, load_items
from .splits import LeakageError, check_no_leakage, split_items
from .trajectory import ExpectedStep, TrajectoryScore, score_trajectory

__version__ = "0.1.0"

__all__ = [
    "CalibrationReport",
    "ExpectedStep",
    "Item",
    "LeakageError",
    "TrajectoryScore",
    "agreement_rate",
    "calibration_report",
    "check_no_leakage",
    "cohens_kappa",
    "expected_calibration_error",
    "load_items",
    "reliability_bins",
    "score_trajectory",
    "split_items",
]

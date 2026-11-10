"""calibrated-eval-kit: minimal tools for calibrated LLM/agent evaluation."""

from .bootstrap import BootstrapInterval, bootstrap_ci, ece_ci, kappa_ci
from .calibration import (
    CalibrationReport,
    agreement_rate,
    brier_score,
    calibration_report,
    cohens_kappa,
    expected_calibration_error,
    reliability_bins,
)
from .data import Item, load_items
from .export import (
    reliability_table,
    reliability_table_csv,
    reliability_table_json,
    rows_from_bins,
)
from .judges import JudgeAdapter, JudgeOutput, MockJudge, item_from_judge_output, judge_items
from .splits import LeakageError, check_no_leakage, split_items
from .trajectory import ExpectedStep, TrajectoryScore, score_trajectory

__version__ = "0.1.0"

__all__ = [
    "BootstrapInterval",
    "CalibrationReport",
    "ExpectedStep",
    "Item",
    "JudgeAdapter",
    "JudgeOutput",
    "LeakageError",
    "MockJudge",
    "TrajectoryScore",
    "agreement_rate",
    "bootstrap_ci",
    "brier_score",
    "calibration_report",
    "check_no_leakage",
    "cohens_kappa",
    "ece_ci",
    "expected_calibration_error",
    "item_from_judge_output",
    "judge_items",
    "kappa_ci",
    "load_items",
    "reliability_bins",
    "reliability_table",
    "reliability_table_csv",
    "reliability_table_json",
    "rows_from_bins",
    "score_trajectory",
    "split_items",
]

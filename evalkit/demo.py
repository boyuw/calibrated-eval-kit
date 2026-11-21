"""Demo: calibration report on the bundled items + one trajectory score.

Run with:  python -m evalkit.demo
"""

from __future__ import annotations

from pathlib import Path

from .bootstrap import ece_ci, kappa_ci
from .calibration import calibration_report, cohens_kappa, weighted_kappa
from .data import Item, load_items
from .export import reliability_table_csv
from .slices import per_slice_calibration
from .trajectory import ExpectedStep, Trace, score_trajectory

EXAMPLES = Path(__file__).parent / "examples" / "items.jsonl"
CODE_REVIEW = Path(__file__).parent / "examples" / "code_review.jsonl"


def print_calibration() -> None:
    items = load_items(EXAMPLES)
    report = calibration_report(items, n_bins=5)
    print("== Judge calibration (bundled example items) ==")
    print(f"items:            {report.n_items}")
    print(f"agreement:        {report.agreement:.3f}")
    print(f"Cohen's kappa:    {report.kappa:.3f}")
    print(f"ECE (5 bins):     {report.ece:.3f}")
    print(f"Brier score:      {report.brier:.3f}")
    print()
    print(f"{'conf bin':<12}{'n':>4}{'mean conf':>11}{'accuracy':>10}")
    for b in report.bins:
        if not b.count:
            continue
        close = "]" if b.upper == 1.0 else ")"  # top bin includes confidence 1.0
        print(f"[{b.lower:.1f}, {b.upper:.1f}{close}{'':<5}{b.count:>4}"
              f"{b.mean_confidence:>11.2f}{b.accuracy:>10.2f}")
    print()


def print_per_slice() -> None:
    for path, label in ((EXAMPLES, "general knowledge"),
                        (CODE_REVIEW, "code review")):
        items = load_items(path)
        reports = per_slice_calibration(items)
        print(f"== Per-slice calibration ({label}: {len(items)} items) ==")
        print(f"{'slice':<12}{'n':>4}{'agreement':>11}{'kappa':>8}"
              f"{'ECE':>8}{'Brier':>8}")
        for r in reports:
            flag = "" if r.reliable else "  (n < 5: small slice, treat with caution)"
            print(f"{r.name:<12}{r.n_items:>4}{r.agreement:>11.3f}"
                  f"{r.kappa:>8.3f}{r.ece:>8.3f}{r.brier:>8.3f}{flag}")
        print()


def print_bootstrap_cis() -> None:
    items = load_items(EXAMPLES)
    kappa = kappa_ci(items, n_boot=1000, seed=0)
    ece = ece_ci(items, n_boot=1000, seed=0)
    print("== Bootstrap 95% intervals (percentile, 1000 replicates, seed 0) ==")
    print(f"Cohen's kappa:    {kappa.estimate:.3f}  "
          f"[{kappa.lower:.3f}, {kappa.upper:.3f}]")
    print(f"ECE (5 bins):     {ece.estimate:.3f}  "
          f"[{ece.lower:.3f}, {ece.upper:.3f}]")
    print("On 24 items the kappa interval spans zero: the point estimate")
    print("0.364 does not establish that the judge beats chance.")
    print()


def print_weighted_kappa() -> None:
    # A 1–5 star rating judge: the labels are ordered, so a miss by one
    # star is a smaller error than a miss across the scale. Plain kappa
    # counts both as failures; weighted kappa gives partial credit.
    labels = ["1", "2", "3", "4", "5"]
    pairs = [
        ("5", "5"), ("5", "5"), ("4", "4"), ("4", "5"), ("3", "3"),
        ("3", "4"), ("2", "2"), ("2", "3"), ("1", "1"), ("1", "2"),
        ("5", "3"), ("2", "4"),
    ]
    items = [
        Item(id=f"rating-{i}", prompt="rate this answer", reference=None,
             human_label=human, judge_label=judge, judge_confidence=0.9)
        for i, (human, judge) in enumerate(pairs)
    ]
    print("== Weighted kappa (1–5 star ratings, 12 items) ==")
    print(f"plain kappa:        {cohens_kappa(items):.3f}")
    print(f"linear-weighted:    {weighted_kappa(items, labels):.3f}")
    print(f"quadratic-weighted: {weighted_kappa(items, labels, weights='quadratic'):.3f}")
    print("The two far misses (5 vs 3, 2 vs 4) are what keep the weighted")
    print("values below 1.0; the one-star misses cost little under the")
    print("quadratic weights and half a point under the linear ones.")
    print()


def print_reliability_export() -> None:
    items = load_items(EXAMPLES)
    print("== Reliability diagram data (CSV) ==")
    print(reliability_table_csv(items, n_bins=5), end="")


def print_trajectory() -> None:
    expected = [
        ExpectedStep("lookup_order", {"order_id": str}),
        ExpectedStep("check_refund_policy", {"order_id": str}),
        ExpectedStep("issue_refund", {"order_id": str, "amount": float}),
    ]
    trace: Trace = [
        {"tool": "lookup_order", "args": {"order_id": "A-1042"},
         "result": {"status": "found", "total": 49.99}},
        {"tool": "check_refund_policy", "args": {"order_id": "A-1042"},
         "result": {"refundable": True, "window_days": 30}},
        {"tool": "issue_refund", "args": {"order_id": "A-1042", "amount": 49.99},
         "result": {"status": "refunded"}},
    ]
    score = score_trajectory(
        trace, expected,
        final_predicate=lambda t: t[-1].get("result", {}).get("status") == "refunded",
    )
    print("== Trajectory score (refund agent, correct trace) ==")
    for s in score.steps:
        print(f"step {s.index}: expected {s.expected_tool:<18} "
              f"actual {s.actual_tool!s:<18} tool_ok={s.tool_ok} "
              f"args_ok={s.args_ok} score={s.score:.2f}")
    print(f"final state ok: {score.final_ok}")
    print(f"overall:        {score.overall:.3f}")


def main() -> None:
    print_calibration()
    print_weighted_kappa()
    print_per_slice()
    print_bootstrap_cis()
    print_reliability_export()
    print_trajectory()


if __name__ == "__main__":
    main()

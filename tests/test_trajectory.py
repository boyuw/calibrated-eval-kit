"""Trajectory scoring: full credit, partial credit, malformed args."""

import pytest

from evalkit.trajectory import ExpectedStep, score_trajectory

EXPECTED = [
    ExpectedStep("lookup_order", {"order_id": str}),
    ExpectedStep("check_refund_policy", {"order_id": str}),
    ExpectedStep("issue_refund", {"order_id": str, "amount": float}),
]

GOOD_TRACE = [
    {"tool": "lookup_order", "args": {"order_id": "A-1"},
     "result": {"status": "found"}},
    {"tool": "check_refund_policy", "args": {"order_id": "A-1"},
     "result": {"refundable": True}},
    {"tool": "issue_refund", "args": {"order_id": "A-1", "amount": 10.0},
     "result": {"status": "refunded"}},
]

REFUNDED = lambda t: t[-1].get("result", {}).get("status") == "refunded"  # noqa: E731


def test_correct_trace_gets_full_credit():
    score = score_trajectory(GOOD_TRACE, EXPECTED, REFUNDED)
    assert all(s.score == 1.0 for s in score.steps)
    assert score.final_ok is True
    assert score.overall == pytest.approx(1.0)


def test_one_wrong_tool_gets_partial_credit():
    trace = [dict(step) for step in GOOD_TRACE]
    trace[1] = {"tool": "send_email", "args": {}, "result": {}}
    score = score_trajectory(trace, EXPECTED, REFUNDED)
    assert score.steps[1].tool_ok is False
    assert score.steps[1].score == 0.0
    # mean step score 2/3, final state still reached:
    # overall = 0.8 * (2/3) + 0.2 * 1
    assert score.overall == pytest.approx(0.8 * (2 / 3) + 0.2)


def test_malformed_args_lose_the_args_half():
    trace = [dict(step) for step in GOOD_TRACE]
    trace[2] = {"tool": "issue_refund", "args": {"order_id": "A-1"},
                "result": {"status": "refunded"}}  # missing required "amount"
    score = score_trajectory(trace, EXPECTED, REFUNDED)
    assert score.steps[2].tool_ok is True
    assert score.steps[2].args_ok is False
    assert score.steps[2].score == pytest.approx(0.5)


def test_missing_step_scores_zero():
    score = score_trajectory(GOOD_TRACE[:2], EXPECTED)
    assert score.steps[2].actual_tool is None
    assert score.steps[2].score == 0.0
    assert score.final_ok is None

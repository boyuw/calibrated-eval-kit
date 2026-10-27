"""Scoring tool-use agent trajectories.

A final answer can look right while the trace is wrong: wrong tool,
malformed arguments, skipped steps. This module scores a trace step by
step against an expected tool sequence, plus an optional predicate on
the final state, with partial credit throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# A trace is a list of steps: {"tool": str, "args": dict, "result": Any}.
Trace = list[dict]
FinalPredicate = Callable[[Trace], bool]


@dataclass(frozen=True)
class ExpectedStep:
    """One expected step: a tool name and the argument schema it must satisfy.

    arg_schema maps argument names to a type (or tuple of types); a step's
    args are schema-valid when every named argument is present and an
    instance of the given type. Extra arguments are allowed.
    """

    tool: str
    arg_schema: dict[str, type | tuple[type, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class StepScore:
    index: int
    expected_tool: str
    actual_tool: str | None  # None when the trace has no step at this index
    tool_ok: bool
    args_ok: bool
    score: float


@dataclass(frozen=True)
class TrajectoryScore:
    steps: list[StepScore]
    final_ok: bool | None  # None when no final predicate was given
    overall: float


def _args_valid(args: object, schema: dict[str, type | tuple[type, ...]]) -> bool:
    if not isinstance(args, dict):
        return False
    return all(
        name in args and isinstance(args[name], typ) for name, typ in schema.items()
    )


def score_trajectory(
    trace: Trace,
    expected: list[ExpectedStep],
    final_predicate: FinalPredicate | None = None,
    *,
    step_weight: float = 0.8,
    final_weight: float = 0.2,
) -> TrajectoryScore:
    """Score a trace against the expected tool sequence.

    Steps are aligned positionally. Each expected step earns half credit
    for the right tool and half for schema-valid args; a missing step
    earns 0. Steps beyond the expected length are ignored. The overall
    score is step_weight * (mean step score) + final_weight * (final
    predicate), or just the mean step score when no predicate is given.
    """
    if not expected and final_predicate is None:
        raise ValueError("nothing to score: no expected steps and no predicate")
    if final_predicate is not None and abs(step_weight + final_weight - 1.0) > 1e-9:
        raise ValueError("step_weight + final_weight must equal 1")

    steps: list[StepScore] = []
    for i, exp in enumerate(expected):
        actual = trace[i] if i < len(trace) else None
        if actual is None:
            steps.append(StepScore(i, exp.tool, None, False, False, 0.0))
            continue
        tool_ok = actual.get("tool") == exp.tool
        args_ok = _args_valid(actual.get("args", {}), exp.arg_schema)
        steps.append(
            StepScore(i, exp.tool, actual.get("tool"), tool_ok, args_ok,
                      0.5 * tool_ok + 0.5 * args_ok)
        )

    step_mean = sum(s.score for s in steps) / len(steps) if steps else 0.0
    if final_predicate is None:
        return TrajectoryScore(steps, None, step_mean)
    final_ok = bool(final_predicate(trace))
    overall = step_weight * step_mean + final_weight * final_ok
    return TrajectoryScore(steps, final_ok, overall)

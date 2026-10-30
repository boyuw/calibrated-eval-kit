"""Judge adapters: how a judge's raw output becomes an :class:`Item`.

The kit evaluates judge *outputs*; it never calls a model. A real LLM
judge plugs in through one small interface — any callable (or object
with a ``judge`` method) taking a prompt and an optional reference and
returning a :class:`JudgeOutput`:

.. code-block:: python

    class MyLlmJudge:
        \"\"\"Adapter over a real judge model (sketch, not run here).\"\"\"

        def judge(self, prompt: str, reference: str | None) -> JudgeOutput:
            # 1. Format a judging prompt: the question, the candidate
            #    answer, and (when available) the reference answer.
            # 2. Call the model and parse its verdict + confidence.
            # 3. Return JudgeOutput(label=..., confidence=...).
            #    Confidence must be in [0, 1]; the kit validates it.
            ...

    items = judge_items(cases, MyLlmJudge(), human_labels)

That is the whole contract: the kit never sees prompts, API keys, or
model responses — only the parsed label and confidence, which is what
the calibration metrics consume. :class:`MockJudge` below is the offline
stand-in for that adapter, useful in tests and in the README example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .data import Item


@dataclass(frozen=True)
class JudgeOutput:
    """One parsed verdict from a judge.

    ``label`` is the judge's categorical verdict (opaque string, same
    vocabulary as the human labels it is compared against); ``confidence``
    is the judge's stated probability that the verdict is right, in
    [0, 1]; ``rationale`` is free text the kit ignores, kept so an
    adapter can carry the model's reasoning through for inspection.
    """

    label: str
    confidence: float
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.confidence, (int, float)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"judge confidence must be in [0, 1], got {self.confidence!r}")


class JudgeAdapter(Protocol):
    """The interface a real judge adapter implements."""

    def judge(self, prompt: str, reference: str | None) -> JudgeOutput:
        """Return the judge's verdict for one prompt/reference pair."""
        ...


# A plain callable with the same signature is also an adapter; this alias
# is the type to annotate when a function, not a class, is supplied.
JudgeFn = Callable[[str, str | None], JudgeOutput]


def item_from_judge_output(
    id: str,
    prompt: str,
    reference: str | None,
    human_label: str,
    output: JudgeOutput,
) -> Item:
    """Build the kit's :class:`Item` from one adapter verdict."""
    return Item(
        id=id,
        prompt=prompt,
        reference=reference,
        human_label=human_label,
        judge_label=output.label,
        judge_confidence=float(output.confidence),
    )


class MockJudge:
    """A deterministic, offline stand-in for a real judge adapter.

    Verdicts are canned per prompt: ``verdicts`` maps a prompt to a
    ``(label, confidence)`` pair (or a :class:`JudgeOutput`). Any prompt
    not in the map gets ``(default_label, default_confidence)`` — the
    honest stand-in for "the model said something unparseable", rather
    than silently reusing a canned answer. No network, no keys, no RNG.
    """

    def __init__(
        self,
        verdicts: dict[str, JudgeOutput | tuple[str, float]] | None = None,
        default_label: str = "incorrect",
        default_confidence: float = 0.5,
    ) -> None:
        self._verdicts: dict[str, JudgeOutput] = {}
        for prompt, verdict in (verdicts or {}).items():
            if isinstance(verdict, JudgeOutput):
                self._verdicts[prompt] = verdict
            else:
                label, confidence = verdict
                self._verdicts[prompt] = JudgeOutput(label=label, confidence=confidence)
        self._default = JudgeOutput(label=default_label, confidence=default_confidence)

    def judge(self, prompt: str, reference: str | None) -> JudgeOutput:
        """Return the canned verdict for ``prompt``, or the default."""
        return self._verdicts.get(prompt, self._default)


def judge_items(
    cases: list[tuple[str, str, str | None, str]],
    adapter: JudgeAdapter | JudgeFn,
) -> list[Item]:
    """Run an adapter over cases and return kit :class:`Item` objects.

    Each case is ``(id, prompt, reference, human_label)``. ``adapter``
    may be an object with a ``judge`` method or a plain function of
    ``(prompt, reference)``; a function is wrapped, not special-cased
    anywhere downstream.
    """
    judge = adapter.judge if hasattr(adapter, "judge") else adapter  # type: ignore[union-attr]
    return [
        item_from_judge_output(id, prompt, reference, human_label, judge(prompt, reference))
        for id, prompt, reference, human_label in cases
    ]

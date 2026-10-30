# calibrated-eval-kit

A minimal, dependency-free toolkit for evaluating LLM and agent outputs
with **calibrated judges** — judges whose stated confidence means what it
says. Pure Python 3.11+, standard library only.

It answers three questions that come up in every serious evaluation setup:

1. **Can I trust the judge?** Agreement with human labels, Cohen's kappa,
   and calibration: reliability bins by judge confidence plus expected
   calibration error (ECE). A judge that is 95% confident and 70% right
   is not a judge — it is a coin flip with good PR.
2. **Am I tuning on my test set?** Deterministic dev / held-out splits,
   plus a leakage guard that fails loudly if an item — or a near-duplicate
   of its prompt — appears in both sets.
3. **Was the trajectory right, or just the answer?** Step-level scoring
   of tool-use traces against an expected tool sequence (right tool?
   schema-valid args?) and a final-state predicate, with partial credit.

## Quickstart

```bash
pip install -e .
python -m evalkit.demo   # calibration report + trajectory example
pytest                   # run the tests
```

No installation is needed to try the demo if you run it from the repo
root — `python -m evalkit.demo` works in place.

## The bundled example

`evalkit/examples/items.jsonl` holds 24 synthetic judged items. The
fixture judge agrees with humans on 17 of 24 — and is *overconfident on
several of its errors*, which is exactly the failure mode the calibration
report exists to surface:

```
== Judge calibration (bundled example items) ==
items:            24
agreement:        0.708
Cohen's kappa:    0.364
ECE (5 bins):     0.161
```

Note the story the numbers tell: 71% agreement sounds acceptable, but
chance-corrected agreement (kappa 0.364) is weak, and the highest
confidence bin — mean confidence 0.92 — is only 75% accurate. That gap
is overconfidence, and it is invisible if you only track agreement.
(Run the demo for the full per-bin table.)

Input format is JSONL, one item per line:

```json
{"id": "item-01", "prompt": "...", "reference": "Canberra",
 "human_label": "correct", "judge_label": "correct", "judge_confidence": 0.98}
```

Labels are categorical strings; the human label is the gold standard and
the judge label is the judge's verdict on the same output.

## Modules

| Module | What it does |
|---|---|
| `evalkit.data` | Load and validate items from JSONL. |
| `evalkit.judges` | Judge-adapter interface (`JudgeOutput`, `MockJudge`, `judge_items`): how a real judge's verdict becomes an `Item`. |
| `evalkit.calibration` | Agreement, Cohen's kappa, reliability bins, ECE, full report. |
| `evalkit.splits` | Hash-based deterministic dev/held-out split; leakage check by id and by normalized prompt text. |
| `evalkit.trajectory` | Positional step scoring of tool-use traces against an expected sequence + final-state predicate. |

## Plugging in a real judge

The kit never calls a model. Implement the `JudgeAdapter` interface —
`judge(prompt, reference) -> JudgeOutput(label, confidence)` — around
your judge: format a judging prompt, call the model, parse its verdict
and confidence, and return them. `judge_items` turns a list of cases
into `Item`s that the calibration metrics consume directly. `MockJudge`
is the offline stand-in: canned verdicts per prompt, with an explicit
default for unseen prompts (it never silently reuses a canned answer),
so tests and examples run with no network and no keys. See
`evalkit/judges.py` for the full contract and a sketch of a real adapter.

## Scope

This is a toy-scale reference implementation, deliberately small enough
to read in one sitting. It is not a serving system, a benchmark suite, or
a judge model. The design decisions — and what is deliberately left out —
are documented in [docs/DESIGN.md](docs/DESIGN.md).

## License

MIT — see [LICENSE](LICENSE).

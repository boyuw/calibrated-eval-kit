# Design notes

Why the kit is built the way it is, and what it deliberately does not do.
The goal is a reference implementation small enough to review in ~20
minutes and honest enough to argue about in an interview.

## Decisions

**Standard library only.** Calibration math, hashing, and JSONL parsing
need nothing beyond the stdlib. Zero dependencies means zero install
friction and nothing between the reader and the logic.

**Labels are opaque strings.** The kit never assumes "correct"/"incorrect"
or "pass"/"fail"; it only compares judge labels to human labels for
equality and counts marginals. Any categorical scheme works, including
multi-class.

**ECE uses equal-width bins over [0, 1] and per-bin mean confidence.**
Accuracy in a bin is compared against the bin's *mean stated confidence*,
not the bin midpoint — the standard formulation, and the one that matches
the claim being tested ("when I say 0.9, am I right 90% of the time?").
Empty bins are reported with count 0 rather than silently dropped, so a
judge that never uses low confidences is visible in the table.

**Splits hash with SHA-256, never `hash()`.** Python randomizes string
hashing per process (PYTHONHASHSEED), so a split built on `hash()` would
not be reproducible across runs — fatal for a held-out set. The assignment
is `sha256(seed:id) mod 100 < threshold`, which is deterministic,
order-independent, and stable across machines and Python versions.

**Leakage is checked by id *and* by normalized prompt text.** Renaming an
item's id does not decontaminate it. Normalization is intentionally
simple — casefold, collapse non-alphanumerics to single spaces — enough
to catch copy-paste and light-edit duplicates, which are the realistic
leakage mode at this scale.

**Trajectory alignment is positional, not edit-distance.** Step *i* of
the trace is scored against expected step *i*. This is stricter and
simpler than sequence alignment, and it matches how agent harnesses are
usually specified (a fixed playbook). Extra steps earn no credit rather
than negative credit; a missing step earns 0. Per-step credit is split
evenly between "right tool" and "schema-valid args" so a trace that
calls the right tool with malformed arguments is visibly half-wrong.

**Argument schemas are Python types.** `{"amount": float}` reads well and
needs no schema language. Known quirk, accepted at this scale:
`isinstance(True, int)` is `True`, so a boolean would pass an `int`
check. A production system would want a stricter validator.

**Brier score sits alongside ECE in the report.** ECE measures
calibration only, and only on bin averages; the Brier score is the
mean squared error of each stated confidence against correctness, so
it needs no binning and penalises a confident error on a single item.
It is a proper scoring rule — minimised in expectation only by stating
the true probability of being right — but it mixes calibration with
resolution, so the report carries both numbers and the demo prints
both.

**Exports keep empty bins, with a signed gap.** The reliability-diagram
export (`evalkit.export`) is the same binning as the report, serialized
for embedding or plotting. Empty bins stay in the table with null/blank
metrics rather than vanishing, and the gap column is accuracy minus mean
confidence, so a negative gap reads as overconfidence in that bin.

**Weights are parameters with stated defaults.** The trajectory default
(0.8 steps / 0.2 final state) reflects a belief: process matters more
than outcome, but outcome still matters. It is a parameter precisely
because reasonable teams disagree.

## Deliberately out of scope

- **No judge clients — but a defined adapter seam.** The kit evaluates
  judge *outputs*; calling an LLM to produce them is the caller's job.
  `evalkit/judges.py` defines exactly where that job plugs in
  (`judge(prompt, reference) -> JudgeOutput`) and ships `MockJudge` as
  the offline stand-in, so the boundary is a documented interface
  rather than an ad-hoc conversion in each caller. This keeps the kit
  model-agnostic and testable offline.
- **No statistics beyond point estimates.** No confidence intervals, no
  bootstrap, no significance tests. On 24 items a kappa without an
  interval is a conversation starter, not a result — the kit does not
  pretend otherwise.
- **No semantic duplicate detection.** Embedding-based near-duplicate
  search would catch more leakage but needs a model and a vector store;
  normalized-text matching covers the common cases.
- **No persistence, serving, or dashboards.** Results are dataclasses and
  printed tables. Anything more belongs in the system that embeds the kit.
- **No async / batching / parallelism.** Item sets at this scale load in
  milliseconds.

## When to outgrow it

If you need judge ensembles, per-slice calibration (by task type, by
model version), drift monitoring over time, or trace alignment under
reordering and retries, you have left toy scale — take the data model
(`Item`, `ExpectedStep`) and the metric definitions with you; they are
the parts designed to survive.

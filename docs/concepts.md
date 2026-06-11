# Concepts

FieldLoop ties each decision a deployed policy made to the real-world outcome it caused. Failures
become replayable evidence. Each version is gated on proving it is safer before reaching a robot.

```
capture → join → curate → train → gate → deploy → (repeat)
```

The `(repeat)` is the data flywheel: each deployment surfaces new incidents and edge cases that feed the next training cycle.

## Rollout — a decision

A **rollout** is one decision a policy made on a robot: which policy version ran, when (on a
skew-free monotonic clock), and pointers to the observation/action payloads. Rollouts are recorded
on the robot and shipped by a sidecar; the heavy sensor bytes stay in your own storage.

## Outcome — what happened, later

An **outcome** is a real-world signal that arrives *after* the decision and usually carries no
reference back to it: a teleop takeover, an e-stop, a collision, a downstream failure. The set of
outcome kinds is fixed so analytics share one vocabulary.

## The join — reconstructing the link

The hard part: nothing attaches an outcome to the decision that caused it — the failure could
have happened well before the signal arrives. The **attribution cascade** reconstructs the link by escalating strategies — explicit reference, temporal proximity
(weighted by recency), spatial co-location, causal chain, and *synthetic absence* (a window proven
covered by heartbeats with no adverse outcome is a confident success). Every binding carries an
**explicit confidence**. The default is the raw recency/coverage score; a calibration curve fit from
curator-confirmed labels replaces it once enough labels exist. An inferred binding is never treated
as ground truth. The result is a typed `Feedback` row: which outcome bound to which rollout, by what
method, how sure. See `docs/quickstart.md` for this on in-memory data.

## Curate — a frozen dataset

A **curation** pins the exact `(rollout, feedback)` pairs a dataset is built from, content-hashed so
it reproduces byte-for-byte. Synthetic-provenance data can train and pre-screen but is flagged so it
can never become the evidence that ships a policy to a real robot.

## Train — trait-backed

Fieldloop orchestrates training (the job state machine, lineage, provenance carry-through) but does
not *do* it: the trainer sits behind a trait, so a team plugs in its own (behavior cloning, RL, a VLA
fine-tune). See `docs/adr/0002-seams-and-stubs.md`.

## Gate — prove it is safer

The **gate** is tiered and fail-closed. An offline tier can pass on its own, but shipping requires a
real-robot tier: a candidate-vs-incumbent A/B that a sequential test scores a genuine win. A blocked
ship is rigor, not an error — it states the missing evidence.

## Deploy — staged, with rollback

A deploy is a gated, staged assignment a robot pulls: `shadow → canary → fleet`, advanced on a
healthy live signal and rolled back on a regression. Fieldloop publishes the assignment and records
the decision; the robot's runtime fetches, verifies, and applies the weights.

## Evidence — three states, never blurred

Every binding behind a verdict is **Confirmed** (gate-grade), **Hypothesis** (an inferred lead), or
**Excluded** (with a reason: synthetic, untrusted version, retracted, …). Correlation is surfaced as
a lead to confirm by replay, never as a proven cause.

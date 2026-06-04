# Incident workbench

The workbench answers, for a failure: *what happened, is it recurring, and how do I turn it into a
useful artifact?* It works retrospectively — point it at historical logs plus incident timestamps; it
does not require adopting instrumentation first.

## The investigation

1. **Cohort.** The failures are split from a healthy baseline (rollouts bound to a failure vs. the
   rest), so contributors can be ranked against a comparison group.
2. **Likely contributors.** Signals that moved most in the failures versus the baseline are ranked —
   *correlation, leads to confirm by replay, never a proven cause.* The caveat travels with the data.
3. **Replay.** An ordered timeline of the failing run, frame by frame, with the message-index windows
   into the recorded payloads. (Decoding the payloads into video is a viewer-side step.)
4. **Confirm.** A human confirms a root cause. This writes an authoritative manual binding into the
   *same supersession slot* as the inferred failure, so it replaces the hypothesis rather than adding
   a parallel one — and the rollout must exist, so confirming never fabricates evidence.

## Recurrence

The retrospective report ranks the **top recurring failure modes**, the failure-class breakdown
(custom labels roll up here — see `docs/adapter.md`), and which **policy version and robot** are
overrepresented. Counts are over real bindings; when the corpus exceeds the read sample, the report
says so (`sampled`) rather than presenting a sample as the whole fleet.

## Honesty rails

- Confirmed evidence, an inferred hypothesis, and an excluded binding are three distinct states,
  never blurred.
- Ranked contributors are leads, not causes.
- An inferred binding never feeds the safety gate; only confirmed evidence does.

These are enforced in the data the API returns, not just the prose. See `docs/concepts.md`.

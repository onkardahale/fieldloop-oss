# ADR 0002 — Trait seams and fail-fast binaries for external systems

Status: accepted

## Context

Several capabilities depend on systems that cannot run in a unit test or CI image: GPU policy
training, a physics simulator, a live ClickHouse/Postgres/object-store, a robot's OTA runtime, a
durable workflow engine. The whole loop's correctness must still be verifiable offline, and a binary
must never *pretend* to have wired infrastructure it has not.

## Decision

- **Each external system sits behind a trait** (`TrainingBackend`, `SimEvaluator`, `Store`,
  `RobotPolicyRuntime`, …). A deterministic in-crate implementation lets the surrounding logic —
  state machines, lineage, provenance — run end to end in tests. The real implementation is a
  drop-in at the same trait; no other code changes.
- **A binary that lacks its real backend fails fast** rather than starting with placeholder wiring.
  The serving and ingest binaries refuse to run without a durable token store, signer, and real
  databases — running them against in-memory fakes would serve real operators against nothing.

## Consequences

- The loop is verified up to — and including — each boundary; only the external system itself is
  unverified, and that boundary is a contract test on the trait, not a guess.
- A test backend is **not** a stand-in for the missing system: it pins the exact input/output shape
  the real one must honor.
- Reviewers must read these as *extension points*, not unfinished work: the seam is the deliverable.

## See also

`docs/adr/0003-clickhouse-and-postgres.md` (the two-store split), the per-crate `//!` docs for which
trait each crate owns.

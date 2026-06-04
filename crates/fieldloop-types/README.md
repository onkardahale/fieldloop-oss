# `fieldloop-types`

The **FROZEN canonical shared data schema** — the Rust expression of
[`ARCHITECTURE.md` Part 2](../../../ARCHITECTURE.md). Every Fieldloop component
agrees on these types: the capture SDK mints them, the sidecar ships them, the
ingest gateway writes them, the JOIN/attribution layer binds them, and
curation/training/eval consume them.

**M0 imperative:** freeze this first. Re-instrumenting a fielded fleet is the most
expensive mistake. Additions are append-only; breaking changes are a fleet-wide
migration event.

This crate is **types + minimal constructors only** — no business logic, no I/O,
no DB code.

## Type → ARCHITECTURE.md map

| Type | Module | ARCHITECTURE.md | Notes |
|---|---|---|---|
| `RolloutId`, `EpisodeId`, `OutcomeId`, `FeedbackId`, `BootId`, `EvalRunId` | `ids` | §2.1/§2.2/§2.3/§2.4, **invariant 2** | UUIDv7 newtypes; embedded ts is **advisory only**. Distinct newtypes (improves on TensorZero's bare `Uuid`) so a swapped id is a compile error. |
| `IdParseError` | `ids` | — | `FromStr` error for the id newtypes. |
| `MonoClock` `{ boot_id, mono_ns, ts_wall_ns }` | `clock` | §2.1/§2.2, **invariant 2**, Part 5 §2 | `(boot_id, mono_ns)` is the **attribution authority**. `mono_delta_ns` returns `None` across boots. |
| `ServerAnchor` `{ ts_ingest_ns, server_anchor_offset_ns }` | `clock` | §2.1, **invariant 2** | Server-set trusted time; drives cooldown + cross-boot alignment. |
| `TenantId`, `RobotId`, `RobotIdentity` | `tenant` | §2.1/§2.2, **invariant 5**, §6.2, Part 5 §7 | Identity is always `(tenant_id, robot_id)`, bundled so a bare `robot_id` can't travel alone. |
| `PayloadRef`, `ByteRange`, `BoundedBlob` | `payload` | **invariant 3 (BYOS)** & **6**, §6.1 | Rows carry pointers, never sensor bytes; inline fields are byte-bounded + flagged for re-validation. |
| `PolicyVersion` | `policy` | **invariant 4**, §6.3 | Content-hash string newtype (`name@vsemver+sha256[:12]`), **not** a UUID. Threads through every row. |
| `PolicyVersionRecord`, `PolicyProvenance`, `ArtifactFormat` | `policy` | §2.4, **invariant 4 & 6**, §6.1/§6.3 | Registry record + provenance (closed bidirectional lineage). `safetensors`-only trusted format. |
| `Rollout` | `rollout` | **§2.1** | The atom. Identity/ordering · two clocks · policy-claim-until-reconciled · payload-as-pointers. |
| `Trust`, `SchemaConformance`, `EvalContext` | `rollout` | §2.1, **invariant 4**, §6.5 | Server-derived trust (ledger reconciliation); validation state; blinded A/B eval context (`Option`, so arm-without-run is unrepresentable). |
| `OutcomeEvent`, `OutcomeKind` | `outcome` | **§2.2** | Immutable raw outcome log *before* attribution. `explicit_rollout_id: Option` — `None` is the implicit case the cascade must resolve. |
| `Feedback` | `feedback` | **§2.3**, Part 5 | The attributed JOIN output. Append-only; supersession/retraction by appending newer rows. |
| `FeedbackTarget` | `feedback` | §2.3 | `target_id` + `target_type` as one sum type (kills the polymorphic-column double-count bug). |
| `LabelKind` | `feedback` | §2.3 | The supersession slot — distinguishes supersession (within a slot) from multiplicity (across slots). |
| `FeedbackValue` | `feedback` | §2.3 | Typed value sum: `Boolean` / `Float` / `FailureClass` / `DemonstrationRef`. |
| `FailureClass` | `feedback` | typed-config skill | Closed failure taxonomy (perception/planning/manipulation/hardware/network/environment/operator). |
| `JoinMethod` | `feedback` | §2.3, Part 5 §1 | The attribution cascade method (explicit→temporal→spatial→causal→manual + synthetic-absence). |
| `FeedbackSource` | `feedback` | §2.3 | Origin of the signal (detector/teleop/curator/sim/absence-sweeper). |
| `Feedback.dedup_key` (field) | `feedback` | Part 5 §4 | A **string**, NOT a hash-derived UUIDv7 (the red-team fix); the `id` is a fresh v7 per row. |

## The seven invariants, where they live

1. **50Hz loop never blocks** — `Rollout` is a flat, cheap record; ids are
   lock/alloc-free `now_v7()` mints.
2. **Two clocks, one id** — `ids` (advisory v7 ts) + `clock::MonoClock` /
   `ServerAnchor` (the real authority).
3. **Metadata ≠ data plane (BYOS)** — `payload::PayloadRef` (pointers) /
   `BoundedBlob` (small inline only).
4. **`policy_version` threads everywhere** — `policy::PolicyVersion` on every
   `Rollout`; `rollout::Trust` set only after ledger reconciliation.
5. **`(tenant_id, robot_id)` everywhere** — `tenant::RobotIdentity`.
6. **Untrusted-by-default** — closed "kind" enums, `BoundedBlob` re-validation
   flags, explicit `Trust`/`SchemaConformance`.
7. **GATE co-located with the fleet** — topology concern; the data the gate needs
   rides on `rollout::EvalContext` + `Feedback::is_safety_eligible_confidence`.

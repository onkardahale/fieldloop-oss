# ADR 0003 — Two stores: ClickHouse for the log, Postgres for the control plane

Status: accepted

## Context

The system holds two very different kinds of data: a high-volume, append-heavy log of rollouts,
outcomes, and attributed feedback (read with wide analytical scans), and a small transactional
control plane — policy registry, deployment ledger, per-robot rollout state, ingest idempotency —
that needs exact, concurrent, point updates and constraints.

## Decision

- **ClickHouse** for the log: append-only base tables plus `*ByTargetId` lookup tables, with a
  `UInt128` re-key so a UUIDv7 in a sort key still orders chronologically. Outcomes attach to rollouts
  by a later-arriving JOIN, which ClickHouse's scan model serves cheaply.
- **Postgres** for the control plane: the deploy ledger, `rollout_state`, and idempotency live where
  transactions, foreign keys, and row-level security (see ADR 0001) belong.

The raw sensor payloads (MCAP) stay in the customer's object store and are referenced by pointer —
neither database holds the bytes.

## Consequences

- Each store does what it is good at; neither is forced into the other's workload.
- The cost is two systems to operate and a write path that fans out to both (kept off the request hot
  path — see the gateway/sidecar design).
- The storage layer's schema/query/serialization code is pure (generates and validates SQL with no
  DB), so it is unit-tested offline; only the thin live clients open a socket.

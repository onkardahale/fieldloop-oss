//! The ClickHouse side of the store: append-heavy schema, idempotent DDL, row
//! serialization, and the canonical rollout->outcome join read.
//!
//! ClickHouse holds the three append-only streams only — [`fieldloop_types::Rollout`]
//! (the deployed-policy inference log), [`fieldloop_types::OutcomeEvent`] (the raw
//! observed/synthesized outcome log), and the per-grain attributed feedback. The
//! transactional control-plane (policy registry, deployment ledger, work queue,
//! idempotency, episode rollup) deliberately lives in Postgres, not in a ClickHouse
//! materialized view, because that data needs row-level updates and read-your-write
//! that an append-only analytics store does not provide.
//!
//! Everything here is SQL/JSON *generation*: migrations are `String`s you assert on
//! and serialization is a pure function. Nothing in this module connects to a
//! database.

pub mod migrations;
pub mod queries;
pub mod rows;

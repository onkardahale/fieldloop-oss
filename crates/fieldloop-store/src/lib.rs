//! # `fieldloop-store` — storage layer: schema, migrations, rows, the join read, tenant safety
//!
//! Owns how data is laid out and read back across two stores: ClickHouse (the append-heavy log of
//! rollouts, outcomes, and attributed feedback) and Postgres (the transactional control plane:
//! registry, deploy ledger, pending-outcome queue, ingest idempotency, rollups).
//!
//! Pure by default: it GENERATES and VALIDATES SQL and serializes rows — every migration and the
//! canonical join read is a `String` you can assert on, unit-tested with no database. Only the
//! `live-db` feature ([`live`]) opens a socket (its integration tests are `#[ignore]`d).
//!
//! Modules:
//! - [`clickhouse::migrations`] — idempotent DDL for the base + `*ByTargetId`/`*ById` lookup tables
//!   (UInt128 re-key so a UUIDv7 sort key still orders chronologically).
//! - [`clickhouse::rows`] — pure [`fieldloop_types`] → `JSONEachRow` serializers.
//! - [`clickhouse::queries`] — the canonical reward/outcome join read, parameterized + tenant-scoped.
//! - [`tenant`] — the fail-closed query builder: every query is bound to one tenant by parameter,
//!   building one with no tenant is an error, and values are parameterized (never interpolated).
//! - [`postgres::migrations`] — control-plane DDL + per-tenant row-level-security policies.
//!
//! SECURITY-CRITICAL: Postgres RLS is bypassed by `SUPERUSER`/`BYPASSRLS` roles even with
//! `FORCE ROW LEVEL SECURITY`. The data-plane/tenant-scoped connection MUST be a non-superuser,
//! non-`BYPASSRLS` role or per-tenant isolation is silently void; migrations may run as the owner.

// A storage-schema crate should be loud about under-documentation but never panic in
// library code: a malformed query is an error value, not a crash.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod clickhouse;
pub mod postgres;
pub mod tenant;

#[cfg(feature = "live-db")]
pub mod live;

pub use tenant::{BuildError, ParamValue, TenantQuery, TenantQueryBuilder};

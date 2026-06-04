//! The Postgres side of the store: the transactional control-plane.
//!
//! ClickHouse holds the append-only analytics streams. Postgres holds everything
//! that needs row-level updates, read-your-write, and per-tenant row-level security —
//! data an append-only analytics store deliberately does not provide:
//!   * the policy-version **registry** (the immutable record of each trained policy),
//!   * the **deployment ledger** (which policy was actually deployed where — the
//!     server-side truth a robot's self-report is reconciled against),
//!   * the **pending-outcome work queue** (mutable attribution bookkeeping kept out
//!     of the immutable outcome log),
//!   * the **ingest idempotency** table (which batch ids have been seen), and
//!   * the **episode/outcome rollup** (the per-episode aggregate; this lives in
//!     Postgres, not a ClickHouse materialized view, because the rollup is updated as
//!     late feedback arrives).
//!
//! As with the ClickHouse side, everything here is DDL *generation* — strings you can
//! assert on — so nothing in the default build connects to a database.

pub mod migrations;

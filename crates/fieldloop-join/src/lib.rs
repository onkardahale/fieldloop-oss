//! # `fieldloop-join` — decision→outcome attribution
//!
//! Binds each policy decision ([`fieldloop_types::Rollout`]) to the real-world outcome it caused
//! ([`fieldloop_types::OutcomeEvent`]) — which usually carries no reference back to it — and emits
//! [`fieldloop_types::Feedback`]: which decision an outcome belongs to, and how confident.
//!
//! Pure and deterministic: logic only, no I/O, so the engine is exhaustively testable on fixed
//! clocks. Persistence and the live join worker are elsewhere (the `worker` feature).
//!
//! The cascade, most-certain first:
//! - **Explicit** — the outcome carried a real rollout id → confidence `1.0`.
//! - **Temporal** — bind the latest rollout at/before the outcome within the embodiment window,
//!   confidence ramping with recency; a tight-timing kind across a boot boundary is refused as
//!   ambiguous rather than bound on a drifting wall clock.
//! - **Synthetic-absence** — a window provably blanketed by heartbeats with no failure → a
//!   success, scaled by coverage.
//!
//! Contract:
//! - a `Manual` binding has the highest read precedence and is the ground truth confidence is fit
//!   from; the engine respects it as input and never mints one ([`engine::winning_binding`]);
//! - cross-tenant binding is a hard skip, never a low-confidence row ([`engine::SkipReason`]);
//! - confidence is calibrated, not asserted: a raw recency/coverage score mapped per
//!   `(join_method, embodiment)` by a [`calibrator::Calibrator`] (default: identity);
//! - dedup is idempotent, latest-wins via a `dedup_key` over the attribution inputs;
//! - a failure's kind comes from the outcome or a human label — never invented here.
//!
//! Modules: [`engine`] (the cascade), [`confidence`] (raw scores), [`calibrator`] (score→confidence).

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod calibrator;
pub mod confidence;
pub mod engine;

// The live join WORKER: subscribes to the gateway's NATS landed events and runs the
// canonical reward join against a real ClickHouse for each. Behind the optional `worker`
// feature so the default pure-engine build (and the offline closed-loop gate) never
// compiles it and never pulls `async-nats` or the live DB client.
#[cfg(feature = "worker")]
pub mod worker;

pub use calibrator::{Calibrator, FittedCalibrator, IdentityCalibrator};
pub use confidence::{synthetic_absence_raw_score, temporal_raw_score};
pub use engine::{
    AttributeOptions, AttributionReport, Heartbeat, SkipReason, Skipped, attribute,
    attribute_report, attribute_with, winning_binding,
};
